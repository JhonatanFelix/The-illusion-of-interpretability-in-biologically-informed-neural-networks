#!/usr/bin/env python3
"""Audit teacher-student output fidelity with comparable continuous metrics.

Classification agreement and survival concordance are tolerant of numerical
output errors.  This script complements the original task metrics with R² and
correlation on logits, class probabilities, regression targets, and risks.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from pathway_tasks_complete import (
    BinaryStudent,
    BinaryTeacher,
    Config,
    MultiStudent,
    MultiTeacher,
    PathwayNet1L,
    RegressionStudent,
    RegressionTeacher,
    SurvivalStudent,
    SurvivalTeacher,
    build_sparse_overlapping_mask,
    concordance_index,
    get_device,
    set_all_seeds,
    simulate_times_from_risk,
    synth_X_with_latents,
)
from regression_fidelity_study import fidelity_metrics


def train_mse(
    model: nn.Module,
    x_train: torch.Tensor,
    target: torch.Tensor,
    config: Config,
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.wd)
    loader = DataLoader(
        TensorDataset(x_train, target.detach()),
        batch_size=config.batch,
        shuffle=True,
    )
    model.train()
    for _ in range(config.epochs):
        for x_batch, target_batch in loader:
            prediction, _ = model(x_batch)
            loss = F.mse_loss(prediction, target_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def make_problem(args: argparse.Namespace, device: str):
    config = Config(
        G=args.G,
        P=args.P,
        size_min=args.size_min,
        size_max=args.size_max,
        overlap=args.overlap,
        n_train=args.n_train,
        n_test=args.n_test,
        hidden=args.hidden,
        batch=args.batch,
        epochs=args.epochs,
        lr=args.lr,
        temp=args.temp,
        gamma_risk=args.gamma_risk,
        gamma_y=args.gamma_y,
        censor_rate=args.censor_rate,
        teacher_seed=args.teacher_seed,
        data_seed=args.data_seed,
    )
    set_all_seeds(config.teacher_seed)
    mask, pathway_to_genes = build_sparse_overlapping_mask(
        config.G,
        config.P,
        (config.size_min, config.size_max),
        config.overlap,
        config.teacher_seed,
    )
    x_all = synth_X_with_latents(
        config.n_train + config.n_test,
        config.G,
        pathway_to_genes,
        seed=config.data_seed,
    ).to(device)
    return config, mask, x_all[: config.n_train], x_all[config.n_train :]


def binary_audit(config: Config, mask, x_train, x_test, seed: int) -> Dict[str, float]:
    set_all_seeds(config.teacher_seed)
    teacher = BinaryTeacher(mask, hidden=config.hidden, p_drop=config.drop_teacher).to(x_train.device)
    teacher.eval()
    with torch.no_grad():
        target_train, _ = teacher(x_train)
        target_test, _ = teacher(x_test)

    set_all_seeds(seed)
    student = BinaryStudent(mask, hidden=config.hidden, p_drop=config.drop_student).to(x_train.device)
    train_mse(student, x_train, target_train, config)
    student.eval()
    with torch.no_grad():
        prediction, _ = student(x_test)
    hard_target = target_test > 0
    hard_prediction = prediction > 0
    margin = target_test.abs() >= 1e-2
    row = {"task": "binary", "student_seed": seed}
    row.update(fidelity_metrics(target_test, prediction, "output"))
    row.update(
        {
            "task_metric": float((hard_target == hard_prediction).float().mean().item()),
            "task_metric_name": "sign_accuracy",
            "margin_metric": float(
                (hard_target[margin] == hard_prediction[margin]).float().mean().item()
            ),
            "teacher_output_std": float(target_test.std().item()),
            "teacher_margin_median": float(target_test.abs().median().item()),
            "teacher_low_margin_fraction": float((~margin).float().mean().item()),
            "teacher_majority_baseline": float(
                max(hard_target.float().mean().item(), 1.0 - hard_target.float().mean().item())
            ),
        }
    )
    return row


def binary_one_layer_audit(config: Config, mask, x_train, x_test, seed: int) -> Dict[str, float]:
    set_all_seeds(config.teacher_seed)
    teacher = PathwayNet1L(mask, use_relu=False).to(x_train.device)
    teacher.eval()
    with torch.no_grad():
        target_train, _ = teacher(x_train)
        target_test, _ = teacher(x_test)

    set_all_seeds(seed)
    student = PathwayNet1L(mask, use_relu=False).to(x_train.device)
    train_mse(student, x_train, target_train, config)
    student.eval()
    with torch.no_grad():
        prediction, _ = student(x_test)
    hard_target = target_test > 0
    hard_prediction = prediction > 0
    margin = target_test.abs() >= 1e-2
    row = {"task": "binary_1layer", "student_seed": seed}
    row.update(fidelity_metrics(target_test, prediction, "output"))
    row.update(
        {
            "task_metric": float((hard_target == hard_prediction).float().mean().item()),
            "task_metric_name": "sign_accuracy",
            "margin_metric": float(
                (hard_target[margin] == hard_prediction[margin]).float().mean().item()
            ),
            "teacher_output_std": float(target_test.std().item()),
            "teacher_margin_median": float(target_test.abs().median().item()),
            "teacher_low_margin_fraction": float((~margin).float().mean().item()),
            "teacher_majority_baseline": float(
                max(hard_target.float().mean().item(), 1.0 - hard_target.float().mean().item())
            ),
        }
    )
    return row


def multiclass_audit(config: Config, mask, x_train, x_test, seed: int, classes: int) -> Dict[str, float]:
    set_all_seeds(config.teacher_seed)
    teacher = MultiTeacher(
        mask,
        K=classes,
        hidden=config.hidden,
        p_drop=config.drop_teacher,
    ).to(x_train.device)
    teacher.eval()
    with torch.no_grad():
        teacher_logits_train, _ = teacher(x_train)
        teacher_logits_test, _ = teacher(x_test)
        teacher_probability_train = F.softmax(teacher_logits_train / config.temp, dim=1)
        teacher_probability_test = F.softmax(teacher_logits_test / config.temp, dim=1)

    set_all_seeds(seed)
    student = MultiStudent(
        mask,
        K=classes,
        hidden=config.hidden,
        p_drop=config.drop_student,
    ).to(x_train.device)
    optimizer = torch.optim.Adam(student.parameters(), lr=config.lr, weight_decay=config.wd)
    loader = DataLoader(
        TensorDataset(x_train, teacher_probability_train),
        batch_size=config.batch,
        shuffle=True,
    )
    student.train()
    for _ in range(config.epochs):
        for x_batch, target_probability in loader:
            logits, _ = student(x_batch)
            loss = F.kl_div(
                F.log_softmax(logits / config.temp, dim=1),
                target_probability,
                reduction="batchmean",
            ) * (config.temp**2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    student.eval()
    with torch.no_grad():
        student_logits, _ = student(x_test)
        student_probability = F.softmax(student_logits / config.temp, dim=1)
        kd_kl = F.kl_div(
            F.log_softmax(student_logits / config.temp, dim=1),
            teacher_probability_test,
            reduction="batchmean",
        ) * (config.temp**2)
    # Softmax is invariant to adding a different constant to all classes in a
    # sample, so compare logits only after centering within each sample.
    centered_teacher = teacher_logits_test - teacher_logits_test.mean(dim=1, keepdim=True)
    centered_student = student_logits - student_logits.mean(dim=1, keepdim=True)
    dynamic_teacher = centered_teacher - centered_teacher.mean(dim=0, keepdim=True)
    dynamic_student = centered_student - centered_student.mean(dim=0, keepdim=True)
    probability_variation_teacher = (
        teacher_probability_test - teacher_probability_test.mean(dim=0, keepdim=True)
    )
    probability_variation_student = (
        student_probability - student_probability.mean(dim=0, keepdim=True)
    )
    sorted_logits = torch.sort(teacher_logits_test, dim=1).values
    top_margin = sorted_logits[:, -1] - sorted_logits[:, -2]
    hard_teacher = teacher_logits_test.argmax(dim=1)
    class_fractions = torch.bincount(hard_teacher, minlength=classes).float() / len(hard_teacher)

    row = {"task": "multiclass", "student_seed": seed}
    row.update(
        {
            key.replace("output_", "probability_"): value
            for key, value in fidelity_metrics(
                teacher_probability_test.flatten(),
                student_probability.flatten(),
                "output",
            ).items()
        }
    )
    row.update(
        {
            key.replace("output_", "dynamic_logit_"): value
            for key, value in fidelity_metrics(
                dynamic_teacher.flatten(),
                dynamic_student.flatten(),
                "output",
            ).items()
        }
    )
    row.update(
        {
            key.replace("output_", "probability_variation_"): value
            for key, value in fidelity_metrics(
                probability_variation_teacher.flatten(),
                probability_variation_student.flatten(),
                "output",
            ).items()
        }
    )
    row.update(
        {
            key.replace("output_", "centered_logit_"): value
            for key, value in fidelity_metrics(
                centered_teacher.flatten(),
                centered_student.flatten(),
                "output",
            ).items()
        }
    )
    row.update(
        {
            "task_metric": float(
                (student_logits.argmax(dim=1) == hard_teacher).float().mean().item()
            ),
            "task_metric_name": "top1_agreement",
            "kd_kl": float(kd_kl.item()),
            "teacher_margin_median": float(top_margin.median().item()),
            "teacher_majority_baseline": float(class_fractions.max().item()),
            "teacher_probability_std": float(teacher_probability_test.std().item()),
        }
    )
    return row


def scaled_target(raw_train: torch.Tensor, raw_test: torch.Tensor, gamma: float):
    mean = raw_train.mean()
    std = raw_train.std()
    return gamma * (raw_train - mean) / (std + 1e-8), gamma * (raw_test - mean) / (std + 1e-8)


def regression_audit(config: Config, mask, x_train, x_test, seed: int) -> Dict[str, float]:
    set_all_seeds(config.teacher_seed)
    teacher = RegressionTeacher(mask, hidden=config.hidden, p_drop=config.drop_teacher).to(x_train.device)
    teacher.eval()
    with torch.no_grad():
        raw_train, _ = teacher(x_train)
        raw_test, _ = teacher(x_test)
    target_train, target_test = scaled_target(raw_train, raw_test, config.gamma_y)
    set_all_seeds(seed)
    student = RegressionStudent(mask, hidden=config.hidden, p_drop=config.drop_student).to(x_train.device)
    train_mse(student, x_train, target_train, config)
    student.eval()
    with torch.no_grad():
        prediction, _ = student(x_test)
    row = {
        "task": "regression",
        "student_seed": seed,
        "task_metric_name": "r2",
    }
    row.update(fidelity_metrics(target_test, prediction, "output"))
    row["task_metric"] = row["output_r2"]
    row["teacher_output_std"] = float(target_test.std().item())
    return row


def survival_audit(config: Config, mask, x_train, x_test, seed: int) -> Dict[str, float]:
    set_all_seeds(config.teacher_seed)
    teacher = SurvivalTeacher(mask, hidden=config.hidden, p_drop=config.drop_teacher).to(x_train.device)
    teacher.eval()
    with torch.no_grad():
        raw_train, _ = teacher(x_train)
        raw_test, _ = teacher(x_test)
    target_train, target_test = scaled_target(raw_train, raw_test, config.gamma_risk)
    times, events = simulate_times_from_risk(target_test, censor_rate=config.censor_rate)
    times = times.to(x_train.device)
    events = events.to(x_train.device)
    set_all_seeds(seed)
    student = SurvivalStudent(mask, hidden=config.hidden, p_drop=config.drop_student).to(x_train.device)
    train_mse(student, x_train, target_train, config)
    student.eval()
    with torch.no_grad():
        prediction, _ = student(x_test)
    row = {
        "task": "survival",
        "student_seed": seed,
        "task_metric": concordance_index(prediction, times, events),
        "task_metric_name": "concordance_index",
        "teacher_task_metric": concordance_index(target_test, times, events),
    }
    row.update(fidelity_metrics(target_test, prediction, "output"))
    row["teacher_output_std"] = float(target_test.std().item())
    return row


def main(args: argparse.Namespace) -> None:
    device = get_device(args.device)
    torch.set_num_threads(args.threads)
    os.makedirs(args.outdir, exist_ok=True)
    problem = make_problem(args, device)
    config, mask, x_train, x_test = problem
    rows = []
    for offset in range(args.students):
        seed = args.student_seed_base + offset
        rows.extend(
            [
                binary_audit(config, mask, x_train, x_test, seed),
                binary_one_layer_audit(config, mask, x_train, x_test, seed),
                multiclass_audit(config, mask, x_train, x_test, seed, args.classes),
                regression_audit(config, mask, x_train, x_test, seed),
                survival_audit(config, mask, x_train, x_test, seed),
            ]
        )
    results = pd.DataFrame(rows)
    output = os.path.join(args.outdir, "task_fidelity_audit.csv")
    results.to_csv(output, index=False)
    columns = [
        column
        for column in [
            "task",
            "student_seed",
            "task_metric_name",
            "task_metric",
            "output_r2",
            "output_corr2",
            "probability_r2",
            "centered_logit_r2",
            "dynamic_logit_r2",
            "probability_variation_r2",
            "kd_kl",
            "margin_metric",
            "teacher_majority_baseline",
        ]
        if column in results
    ]
    print(results[columns].to_string(index=False))
    print(f"saved: {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--students", type=int, default=1)
    parser.add_argument("--student_seed_base", type=int, default=9000)
    parser.add_argument("--outdir", default="results_regression_fidelity/metric_audit")
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cpu", "cuda"])
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--G", type=int, default=400)
    parser.add_argument("--P", type=int, default=60)
    parser.add_argument("--size_min", type=int, default=14)
    parser.add_argument("--size_max", type=int, default=22)
    parser.add_argument("--overlap", type=float, default=0.55)
    parser.add_argument("--n_train", type=int, default=12000)
    parser.add_argument("--n_test", type=int, default=3000)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--temp", type=float, default=2.0)
    parser.add_argument("--classes", type=int, default=5)
    parser.add_argument("--gamma_y", type=float, default=6.0)
    parser.add_argument("--gamma_risk", type=float, default=6.0)
    parser.add_argument("--censor_rate", type=float, default=0.05)
    parser.add_argument("--teacher_seed", type=int, default=42)
    parser.add_argument("--data_seed", type=int, default=123)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
