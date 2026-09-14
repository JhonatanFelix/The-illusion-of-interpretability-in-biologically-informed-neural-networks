#!/usr/bin/env python3
"""Nested sample-size study with one fixed teacher, input pool, and test set."""

from __future__ import annotations

import argparse
import os
from dataclasses import replace
from typing import Dict, List

import pandas as pd
import torch

from pathway_tasks_complete import (
    Config,
    RegressionTeacher,
    build_sparse_overlapping_mask,
    get_device,
    set_all_seeds,
    synth_X_with_latents,
)
from regression_fidelity_study import TRIALS, ensemble_rows, run_trial


def parse_ints(raw: str) -> List[int]:
    values = sorted({int(value.strip()) for value in raw.split(",") if value.strip()})
    if not values or values[0] <= 0:
        raise ValueError("sample sizes must be positive integers")
    return values


def parse_trials(raw: str, args: argparse.Namespace):
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


def main(args: argparse.Namespace) -> None:
    device = get_device(args.device)
    torch.set_num_threads(args.threads)
    os.makedirs(args.outdir, exist_ok=True)
    sample_sizes = parse_ints(args.sample_sizes)
    trials = parse_trials(args.trials, args)
    max_train = max(sample_sizes)

    base_config = Config(
        G=args.G,
        P=args.P,
        size_min=args.size_min,
        size_max=args.size_max,
        overlap=args.overlap,
        n_train=max_train,
        n_test=args.n_test,
        hidden=args.hidden,
        batch=args.batch,
        epochs=args.epochs,
        lr=args.lr,
        gamma_y=args.gamma_y,
        teacher_seed=args.teacher_seed,
        data_seed=args.data_seed,
    )
    set_all_seeds(base_config.teacher_seed)
    mask, pathway_to_genes = build_sparse_overlapping_mask(
        base_config.G,
        base_config.P,
        (base_config.size_min, base_config.size_max),
        base_config.overlap,
        base_config.teacher_seed,
    )
    # Generate once so training sets are nested and every comparison uses the
    # identical held-out test examples.
    x_all = synth_X_with_latents(
        max_train + base_config.n_test,
        base_config.G,
        pathway_to_genes,
        seed=base_config.data_seed,
    ).to(device)
    train_pool = x_all[:max_train]
    x_test = x_all[max_train:]

    set_all_seeds(base_config.teacher_seed)
    teacher = RegressionTeacher(
        mask,
        hidden=base_config.hidden,
        p_drop=base_config.drop_teacher,
    ).to(device)
    teacher.eval()
    with torch.no_grad():
        raw_pool, _ = teacher(train_pool)
        raw_test, teacher_test_activations = teacher(x_test)

    rows = []
    curves = []
    ensemble_frames = []
    for n_train in sample_sizes:
        config = replace(base_config, n_train=n_train)
        x_train = train_pool[:n_train]
        raw_train = raw_pool[:n_train]
        target_mean = raw_train.mean()
        target_std = raw_train.std()
        standardized_train = (raw_train - target_mean) / (target_std + 1e-8)
        standardized_test = (raw_test - target_mean) / (target_std + 1e-8)
        problem = (
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
        predictions: Dict[str, List[torch.Tensor]] = {trial.name: [] for trial in trials}
        for trial in trials:
            for offset in range(args.students):
                seed = args.student_seed_base + offset
                curve_start = len(curves)
                row, prediction = run_trial(
                    trial,
                    seed,
                    problem,
                    final_gamma=args.gamma_y,
                    curve_every=args.curve_every,
                    curve_rows=curves,
                )
                row["n_train"] = n_train
                rows.append(row)
                predictions[trial.name].append(prediction)
                for curve_row in curves[curve_start:]:
                    curve_row["n_train"] = n_train
                print(
                    f"n_train={n_train:6d} trial={trial.name:12s} seed={seed} "
                    f"train_r2={row['train_r2']:.4f} test_r2={row['test_r2']:.4f}",
                    flush=True,
                )
        ensembles = ensemble_rows(predictions, args.gamma_y * standardized_test)
        ensembles.insert(0, "n_train", n_train)
        ensemble_frames.append(ensembles)

    results = pd.DataFrame(rows)
    curve_frame = pd.DataFrame(curves)
    ensemble_frame = pd.concat(ensemble_frames, ignore_index=True)
    summary = (
        results.groupby(["n_train", "trial"], sort=False)
        .agg(
            n=("student_seed", "size"),
            train_r2_mean=("train_r2", "mean"),
            train_r2_std=("train_r2", "std"),
            test_r2_mean=("test_r2", "mean"),
            test_r2_std=("test_r2", "std"),
            weight_cosine_mean=("cos_std_mean", "mean"),
            activation_correlation_mean=("act_corr_mean", "mean"),
            activation_r2_mean=("act_r2", "mean"),
        )
        .reset_index()
    )

    outputs = {
        "regression_sample_size_trials.csv": results,
        "regression_sample_size_summary.csv": summary,
        "regression_sample_size_curves.csv": curve_frame,
        "regression_sample_size_ensembles.csv": ensemble_frame,
    }
    for filename, frame in outputs.items():
        path = os.path.join(args.outdir, filename)
        frame.to_csv(path, index=False)
        print(f"saved: {path}")
    print(summary.to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample_sizes", default="12000,24000,48000,96000")
    parser.add_argument("--trials", default="baseline")
    parser.add_argument("--students", type=int, default=3)
    parser.add_argument("--student_seed_base", type=int, default=9000)
    parser.add_argument("--outdir", default="results_regression_fidelity/fixed_test_sample_size")
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cpu", "cuda"])
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--curve_every", type=int, default=30)
    parser.add_argument("--G", type=int, default=400)
    parser.add_argument("--P", type=int, default=60)
    parser.add_argument("--size_min", type=int, default=14)
    parser.add_argument("--size_max", type=int, default=22)
    parser.add_argument("--overlap", type=float, default=0.55)
    parser.add_argument("--n_test", type=int, default=3000)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--gamma_y", type=float, default=6.0)
    parser.add_argument("--teacher_seed", type=int, default=42)
    parser.add_argument("--data_seed", type=int, default=123)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
