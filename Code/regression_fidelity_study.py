#!/usr/bin/env python3
"""Diagnose and improve output fidelity in the regression experiment.

The original experiment reports only test MSE and R².  This study keeps the
same synthetic data, sparse teacher, student architecture, and output-only
distillation objective, but adds:

* train/test fidelity metrics and an affine-calibration diagnostic;
* a same-function oracle proving that the student class can represent the
  amplified teacher target exactly;
* controlled optimizer, duration, target-scale, batch-size, and capacity
  trials;
* ensemble metrics across independent student initializations.

The paper's main experiment remains unchanged; this is a separate exploratory
entry point.
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import time
from dataclasses import asdict, dataclass, replace
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from pathway_tasks_complete import (
    Config,
    RegressionStudent,
    RegressionTeacher,
    add_sparse_recovery_metrics,
    build_sparse_overlapping_mask,
    get_device,
    set_all_seeds,
    synth_X_with_latents,
)


@dataclass(frozen=True)
class Trial:
    name: str
    epochs: int = 120
    lr: float = 2e-3
    batch: int = 512
    hidden: int = 64
    schedule: str = "constant"
    train_gamma: float = 6.0
    fine_tune_epochs: int = 0
    fine_tune_lr: float = 5e-4
    lbfgs_steps: int = 0
    weight_decay: float = 0.0
    dropout: float = 0.0


TRIALS: Dict[str, Trial] = {
    "baseline": Trial("baseline"),
    "early_60": Trial("early_60", epochs=60),
    "longer": Trial("longer", epochs=360),
    "cosine_240": Trial("cosine_240", epochs=240, schedule="cosine"),
    "cosine": Trial("cosine", epochs=360, schedule="cosine"),
    "small_batch": Trial("small_batch", batch=128),
    "narrow_16": Trial("narrow_16", hidden=16),
    "narrow_32": Trial("narrow_32", hidden=32),
    "wide": Trial("wide", epochs=240, hidden=128, schedule="cosine"),
    "weight_decay_1e4": Trial("weight_decay_1e4", weight_decay=1e-4),
    "weight_decay_1e3": Trial("weight_decay_1e3", weight_decay=1e-3),
    "dropout_005": Trial("dropout_005", dropout=0.05),
    "dropout_005_90": Trial("dropout_005_90", epochs=90, dropout=0.05),
    "dropout_015": Trial("dropout_015", dropout=0.15),
    "wd_dropout": Trial("wd_dropout", weight_decay=1e-3, dropout=0.05),
    # Train on a unit-variance target and analytically rescale the output head.
    "unit_scale": Trial("unit_scale", train_gamma=1.0),
    # Unit-scale pretraining followed by output-head rescaling and fine-tuning.
    "curriculum": Trial(
        "curriculum",
        epochs=120,
        schedule="cosine",
        train_gamma=1.0,
        fine_tune_epochs=120,
        fine_tune_lr=5e-4,
    ),
    "adam_lbfgs": Trial("adam_lbfgs", lbfgs_steps=30),
}


def r2_score(target: torch.Tensor, prediction: torch.Tensor) -> float:
    target = target.detach().double().cpu()
    prediction = prediction.detach().double().cpu()
    residual = torch.sum((target - prediction) ** 2)
    total = torch.sum((target - target.mean()) ** 2)
    return float((1.0 - residual / (total + 1e-12)).item())


def pearson_score(target: torch.Tensor, prediction: torch.Tensor) -> float:
    target = target.detach().double().cpu()
    prediction = prediction.detach().double().cpu()
    target = target - target.mean()
    prediction = prediction - prediction.mean()
    denominator = torch.linalg.norm(target) * torch.linalg.norm(prediction)
    if denominator <= 1e-12:
        return float("nan")
    return float(torch.dot(target, prediction).div(denominator).item())


def fidelity_metrics(target: torch.Tensor, prediction: torch.Tensor, prefix: str) -> Dict[str, float]:
    target_cpu = target.detach().float().cpu()
    prediction_cpu = prediction.detach().float().cpu()
    error = prediction_cpu - target_cpu
    correlation = pearson_score(target_cpu, prediction_cpu)
    return {
        f"{prefix}_mse": float(torch.mean(error.square()).item()),
        f"{prefix}_mae": float(torch.mean(error.abs()).item()),
        f"{prefix}_r2": r2_score(target_cpu, prediction_cpu),
        f"{prefix}_pearson": correlation,
        f"{prefix}_corr2": correlation * correlation,
    }


def fit_affine_calibration(target: torch.Tensor, prediction: torch.Tensor) -> Tuple[float, float]:
    """Fit target ~= slope * prediction + intercept by least squares."""
    target = target.detach().double().cpu()
    prediction = prediction.detach().double().cpu()
    centered_prediction = prediction - prediction.mean()
    denominator = torch.dot(centered_prediction, centered_prediction)
    if denominator <= 1e-12:
        return 0.0, float(target.mean().item())
    slope = torch.dot(centered_prediction, target - target.mean()) / denominator
    intercept = target.mean() - slope * prediction.mean()
    return float(slope.item()), float(intercept.item())


def predict(model: nn.Module, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    with torch.no_grad():
        return model(x)


def train_adam(
    model: nn.Module,
    x_train: torch.Tensor,
    target_train: torch.Tensor,
    epochs: int,
    lr: float,
    batch_size: int,
    schedule: str,
    weight_decay: float,
    curve_every: int,
    curve_rows: List[Dict[str, float]],
    trial_name: str,
    student_seed: int,
    phase: str,
    x_test: torch.Tensor,
    target_test: torch.Tensor,
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = None
    if schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, epochs),
            eta_min=lr * 0.01,
        )
    elif schedule != "constant":
        raise ValueError(f"Unknown schedule: {schedule}")

    loader = DataLoader(
        TensorDataset(x_train, target_train.detach()),
        batch_size=batch_size,
        shuffle=True,
    )
    checkpoints = {1, epochs}
    checkpoints.update(range(curve_every, epochs + 1, curve_every))

    for epoch in range(1, epochs + 1):
        model.train()
        for x_batch, target_batch in loader:
            prediction, _ = model(x_batch)
            loss = nn.functional.mse_loss(prediction, target_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if epoch in checkpoints:
            train_prediction, _ = predict(model, x_train)
            test_prediction, _ = predict(model, x_test)
            curve_rows.append(
                {
                    "trial": trial_name,
                    "student_seed": student_seed,
                    "phase": phase,
                    "epoch": epoch,
                    "lr": optimizer.param_groups[0]["lr"],
                    "train_mse": float(
                        nn.functional.mse_loss(train_prediction, target_train).item()
                    ),
                    "train_r2": r2_score(target_train, train_prediction),
                    "test_mse": float(
                        nn.functional.mse_loss(test_prediction, target_test).item()
                    ),
                    "test_r2": r2_score(target_test, test_prediction),
                }
            )


def train_lbfgs(
    model: nn.Module,
    x_train: torch.Tensor,
    target_train: torch.Tensor,
    steps: int,
) -> None:
    if steps <= 0:
        return
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=0.5,
        max_iter=steps,
        history_size=20,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        prediction, _ = model(x_train)
        loss = nn.functional.mse_loss(prediction, target_train)
        loss.backward()
        return loss

    model.train()
    optimizer.step(closure)


def rescale_output_head(model: RegressionStudent, factor: float) -> None:
    with torch.no_grad():
        model.out.weight.mul_(factor)
        model.out.bias.mul_(factor)


def build_oracle_student(
    teacher: RegressionTeacher,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    gamma: float,
) -> RegressionStudent:
    """Copy the teacher and transform its output head to the amplified target."""
    mask = teacher.backbone.pathway.mask.detach().clone()
    hidden = teacher.out.in_features
    student = RegressionStudent(mask, hidden=hidden, p_drop=0.0).to(mask.device)
    student.load_state_dict(copy.deepcopy(teacher.state_dict()))
    scale = gamma / float(target_std.item() + 1e-8)
    with torch.no_grad():
        student.out.weight.mul_(scale)
        student.out.bias.sub_(target_mean).mul_(scale)
    return student


def make_fixed_problem(args: argparse.Namespace, device: str):
    config = Config(
        G=args.G,
        P=args.P,
        size_min=args.size_min,
        size_max=args.size_max,
        overlap=args.overlap,
        n_train=args.n_train,
        n_test=args.n_test,
        hidden=args.teacher_hidden,
        batch=args.batch,
        epochs=args.epochs,
        lr=args.lr,
        gamma_y=args.gamma_y,
        teacher_seed=args.teacher_seed,
        data_seed=args.data_seed,
    )
    set_all_seeds(config.teacher_seed)
    mask, pathway_to_genes = build_sparse_overlapping_mask(
        G=config.G,
        P=config.P,
        size_range=(config.size_min, config.size_max),
        overlap_frac=config.overlap,
        seed=config.teacher_seed,
    )
    x_all = synth_X_with_latents(
        n=config.n_train + config.n_test,
        G=config.G,
        ptg=pathway_to_genes,
        beta=0.7,
        noise=0.6,
        seed=config.data_seed,
    ).to(device)
    x_train = x_all[: config.n_train]
    x_test = x_all[config.n_train :]

    set_all_seeds(config.teacher_seed)
    teacher = RegressionTeacher(
        mask,
        hidden=config.hidden,
        p_drop=config.drop_teacher,
    ).to(device)
    teacher.eval()
    with torch.no_grad():
        raw_train, _ = teacher(x_train)
        raw_test, teacher_test_activations = teacher(x_test)
    target_mean = raw_train.mean()
    target_std = raw_train.std()
    standardized_train = (raw_train - target_mean) / (target_std + 1e-8)
    standardized_test = (raw_test - target_mean) / (target_std + 1e-8)
    return (
        config,
        mask,
        pathway_to_genes,
        x_train,
        x_test,
        teacher,
        teacher_test_activations,
        target_mean,
        target_std,
        standardized_train,
        standardized_test,
    )


def evaluate_student(
    trial: Trial,
    student_seed: int,
    student: RegressionStudent,
    teacher: RegressionTeacher,
    pathway_to_genes,
    teacher_test_activations: torch.Tensor,
    x_train: torch.Tensor,
    x_test: torch.Tensor,
    final_train_target: torch.Tensor,
    final_test_target: torch.Tensor,
    elapsed_seconds: float,
) -> Tuple[Dict[str, float], torch.Tensor]:
    train_prediction, _ = predict(student, x_train)
    test_prediction, student_test_activations = predict(student, x_test)
    slope, intercept = fit_affine_calibration(final_train_target, train_prediction)
    calibrated_test = slope * test_prediction + intercept

    row: Dict[str, float] = {
        "trial": trial.name,
        "student_seed": student_seed,
        "epochs": trial.epochs,
        "fine_tune_epochs": trial.fine_tune_epochs,
        "lr": trial.lr,
        "batch": trial.batch,
        "hidden": trial.hidden,
        "schedule": trial.schedule,
        "train_gamma": trial.train_gamma,
        "lbfgs_steps": trial.lbfgs_steps,
        "weight_decay": trial.weight_decay,
        "dropout": trial.dropout,
        "elapsed_seconds": elapsed_seconds,
        "calibration_slope": slope,
        "calibration_intercept": intercept,
        "test_r2_affine": r2_score(final_test_target, calibrated_test),
    }
    row.update(fidelity_metrics(final_train_target, train_prediction, "train"))
    row.update(fidelity_metrics(final_test_target, test_prediction, "test"))
    add_sparse_recovery_metrics(
        row,
        teacher.backbone,
        student.backbone,
        pathway_to_genes,
        teacher_test_activations,
        student_test_activations,
    )
    return row, test_prediction.detach().cpu()


def run_trial(
    trial: Trial,
    student_seed: int,
    problem,
    final_gamma: float,
    curve_every: int,
    curve_rows: List[Dict[str, float]],
) -> Tuple[Dict[str, float], torch.Tensor]:
    (
        _,
        mask,
        pathway_to_genes,
        x_train,
        x_test,
        teacher,
        teacher_test_activations,
        _,
        _,
        standardized_train,
        standardized_test,
    ) = problem
    final_train_target = final_gamma * standardized_train
    final_test_target = final_gamma * standardized_test
    phase_one_target = trial.train_gamma * standardized_train

    set_all_seeds(student_seed)
    student = RegressionStudent(mask, hidden=trial.hidden, p_drop=trial.dropout).to(x_train.device)
    started = time.perf_counter()
    train_adam(
        student,
        x_train,
        phase_one_target,
        epochs=trial.epochs,
        lr=trial.lr,
        batch_size=trial.batch,
        schedule=trial.schedule,
        weight_decay=trial.weight_decay,
        curve_every=curve_every,
        curve_rows=curve_rows,
        trial_name=trial.name,
        student_seed=student_seed,
        phase="initial",
        x_test=x_test,
        target_test=trial.train_gamma * standardized_test,
    )

    if not math.isclose(trial.train_gamma, final_gamma):
        rescale_output_head(student, final_gamma / trial.train_gamma)

    if trial.fine_tune_epochs:
        train_adam(
            student,
            x_train,
            final_train_target,
            epochs=trial.fine_tune_epochs,
            lr=trial.fine_tune_lr,
            batch_size=trial.batch,
            schedule="cosine",
            weight_decay=trial.weight_decay,
            curve_every=curve_every,
            curve_rows=curve_rows,
            trial_name=trial.name,
            student_seed=student_seed,
            phase="fine_tune",
            x_test=x_test,
            target_test=final_test_target,
        )

    train_lbfgs(student, x_train, final_train_target, trial.lbfgs_steps)
    elapsed = time.perf_counter() - started
    return evaluate_student(
        trial,
        student_seed,
        student,
        teacher,
        pathway_to_genes,
        teacher_test_activations,
        x_train,
        x_test,
        final_train_target,
        final_test_target,
        elapsed,
    )


def add_oracle_row(problem, final_gamma: float) -> Tuple[Dict[str, float], torch.Tensor]:
    (
        _,
        _,
        pathway_to_genes,
        x_train,
        x_test,
        teacher,
        teacher_test_activations,
        target_mean,
        target_std,
        standardized_train,
        standardized_test,
    ) = problem
    trial = Trial("oracle_clone", epochs=0)
    oracle = build_oracle_student(teacher, target_mean, target_std, final_gamma)
    return evaluate_student(
        trial,
        teacher.backbone.pathway.weight.numel(),
        oracle,
        teacher,
        pathway_to_genes,
        teacher_test_activations,
        x_train,
        x_test,
        final_gamma * standardized_train,
        final_gamma * standardized_test,
        0.0,
    )


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        "train_r2",
        "test_r2",
        "test_corr2",
        "test_r2_affine",
        "test_mse",
        "relL2_mean",
        "cos_std_mean",
        "act_corr_mean",
        "act_r2",
        "elapsed_seconds",
    ]
    return (
        results.groupby("trial", sort=False)[numeric]
        .agg(["mean", "std"])
        .reset_index()
    )


def ensemble_rows(
    predictions: Dict[str, List[torch.Tensor]],
    final_test_target: torch.Tensor,
) -> pd.DataFrame:
    rows = []
    for trial_name, trial_predictions in predictions.items():
        for ensemble_size in range(1, len(trial_predictions) + 1):
            ensemble_prediction = torch.stack(trial_predictions[:ensemble_size]).mean(dim=0)
            row = {
                "trial": trial_name,
                "ensemble_size": ensemble_size,
            }
            row.update(fidelity_metrics(final_test_target.cpu(), ensemble_prediction, "test"))
            rows.append(row)
    return pd.DataFrame(rows)


def parse_trials(raw: str, args: argparse.Namespace) -> List[Trial]:
    names = [name.strip() for name in raw.split(",") if name.strip()]
    unknown = sorted(set(names).difference(TRIALS))
    if unknown:
        raise ValueError(f"Unknown trials {unknown}; choose from {sorted(TRIALS)}")
    trials = [TRIALS[name] for name in names]
    return [
        replace(trial, epochs=args.epochs, lr=args.lr, batch=args.batch)
        if trial.name == "baseline"
        else trial
        for trial in trials
    ]


def flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [
        "_".join(str(part) for part in column if str(part))
        if isinstance(column, tuple)
        else column
        for column in frame.columns
    ]
    return frame


def main(args: argparse.Namespace) -> None:
    device = get_device(args.device)
    os.makedirs(args.outdir, exist_ok=True)
    torch.set_num_threads(args.threads)
    problem = make_fixed_problem(args, device)
    config = problem[0]
    standardized_test = problem[-1]
    trials = parse_trials(args.trials, args)
    seeds = [args.student_seed_base + index for index in range(args.students)]

    print(f"device={device} threads={torch.get_num_threads()} students={len(seeds)}")
    print(f"problem={asdict(config)}")
    print(f"trials={[trial.name for trial in trials]}")

    result_rows: List[Dict[str, float]] = []
    curve_rows: List[Dict[str, float]] = []
    predictions: Dict[str, List[torch.Tensor]] = {trial.name: [] for trial in trials}

    oracle_row, _ = add_oracle_row(problem, args.gamma_y)
    result_rows.append(oracle_row)
    print(f"oracle_clone test_r2={oracle_row['test_r2']:.9f}")

    for trial in trials:
        for seed in seeds:
            row, prediction = run_trial(
                trial,
                seed,
                problem,
                final_gamma=args.gamma_y,
                curve_every=args.curve_every,
                curve_rows=curve_rows,
            )
            result_rows.append(row)
            predictions[trial.name].append(prediction)
            print(
                f"{trial.name:12s} seed={seed} "
                f"train_r2={row['train_r2']:.4f} test_r2={row['test_r2']:.4f} "
                f"corr2={row['test_corr2']:.4f} affine_r2={row['test_r2_affine']:.4f} "
                f"seconds={row['elapsed_seconds']:.1f}",
                flush=True,
            )

    results = pd.DataFrame(result_rows)
    curves = pd.DataFrame(curve_rows)
    summary = flatten_columns(summarize(results))
    ensembles = ensemble_rows(predictions, args.gamma_y * standardized_test)

    results_path = os.path.join(args.outdir, "regression_fidelity_trials.csv")
    summary_path = os.path.join(args.outdir, "regression_fidelity_summary.csv")
    curves_path = os.path.join(args.outdir, "regression_training_curves.csv")
    ensembles_path = os.path.join(args.outdir, "regression_ensembles.csv")
    results.to_csv(results_path, index=False)
    summary.to_csv(summary_path, index=False)
    curves.to_csv(curves_path, index=False)
    ensembles.to_csv(ensembles_path, index=False)
    print(summary.to_string(index=False))
    print(f"saved: {results_path}")
    print(f"saved: {summary_path}")
    print(f"saved: {curves_path}")
    print(f"saved: {ensembles_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trials",
        default="baseline,early_60,longer,cosine,small_batch,wide,weight_decay_1e4,weight_decay_1e3,dropout_005,dropout_015,unit_scale,curriculum,adam_lbfgs",
        help=f"Comma-separated subset of: {','.join(TRIALS)}",
    )
    parser.add_argument("--students", type=int, default=3)
    parser.add_argument("--student_seed_base", type=int, default=9000)
    parser.add_argument("--outdir", default="results_regression_fidelity")
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cpu", "cuda"])
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--curve_every", type=int, default=30)
    parser.add_argument("--G", type=int, default=400)
    parser.add_argument("--P", type=int, default=60)
    parser.add_argument("--size_min", type=int, default=14)
    parser.add_argument("--size_max", type=int, default=22)
    parser.add_argument("--overlap", type=float, default=0.55)
    parser.add_argument("--n_train", type=int, default=12000)
    parser.add_argument("--n_test", type=int, default=3000)
    parser.add_argument("--teacher_hidden", type=int, default=64)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--gamma_y", type=float, default=6.0)
    parser.add_argument("--teacher_seed", type=int, default=42)
    parser.add_argument("--data_seed", type=int, default=123)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
