import argparse
import itertools
import math
import os
import random
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ---------------------- utils ----------------------
def get_device(arg_device: str) -> str:
    if arg_device == "mps":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    if arg_device == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if arg_device == "cpu":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_csv_ints(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_csv_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_size_ranges(s: str) -> List[Tuple[int, int]]:
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.append((int(a), int(b)))
        else:
            v = int(part)
            out.append((v, v))
    return out


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# ---------------------- masks and data ----------------------
def build_sparse_overlapping_mask(
    G: int,
    P: int,
    size_range: Tuple[int, int] = (14, 22),
    overlap_frac: float = 0.55,
    seed: int = 0,
) -> Tuple[torch.Tensor, Dict[str, List[int]]]:
    """
    Original pathway-size based mask.
    Kept for compatibility with old experiments.
    """
    rng = np.random.default_rng(seed)
    all_genes = np.arange(G, dtype=int)
    gene_lists = []
    prev_pool = np.array([], dtype=int)

    for _ in range(P):
        size = int(rng.integers(size_range[0], size_range[1] + 1))
        size = max(1, min(size, G))

        target_overlap = int(math.floor(size * overlap_frac))
        k_overlap = min(target_overlap, prev_pool.size)

        if k_overlap > 0:
            overlap_part = rng.choice(prev_pool, size=k_overlap, replace=False)
        else:
            overlap_part = np.array([], dtype=int)

        remaining = np.setdiff1d(all_genes, overlap_part, assume_unique=False)
        k_fresh = min(size - k_overlap, remaining.size)

        if k_fresh > 0:
            fresh_part = rng.choice(remaining, size=k_fresh, replace=False)
        else:
            fresh_part = np.array([], dtype=int)

        genes = np.unique(np.concatenate([overlap_part, fresh_part]))

        if genes.size == 0:
            genes = rng.choice(all_genes, size=1, replace=False)

        gene_lists.append(genes)
        prev_pool = np.unique(np.concatenate([prev_pool, genes]))

    mask = torch.zeros(P, G, dtype=torch.float32)
    for i, genes in enumerate(gene_lists):
        mask[i, torch.as_tensor(genes, dtype=torch.long)] = 1.0

    ptg = {f"P{i}": list(map(int, gene_lists[i])) for i in range(P)}
    return mask, ptg


def build_fixed_density_mask(
    G: int,
    P: int,
    density: float,
    seed: int = 0,
    min_edges_per_pathway: int = 1,
) -> Tuple[torch.Tensor, Dict[str, List[int]]]:

    if not (0.0 < density <= 1.0):
        raise ValueError(f"mask density must be in (0, 1], got {density}")

    rng = np.random.default_rng(seed)
    total_possible = G * P
    target_edges = int(round(density * total_possible))

    min_required = P * min_edges_per_pathway
    if target_edges < min_required:
        target_edges = min_required

    target_edges = min(target_edges, total_possible)

    mask_np = np.zeros((P, G), dtype=np.float32)

    # Guarantee at least min_edges_per_pathway per pathway.
    for p in range(P):
        k = min(min_edges_per_pathway, G)
        genes = rng.choice(G, size=k, replace=False)
        mask_np[p, genes] = 1.0

    current_edges = int(mask_np.sum())
    remaining_edges = target_edges - current_edges

    if remaining_edges > 0:
        zero_positions = np.argwhere(mask_np == 0)
        chosen_idx = rng.choice(len(zero_positions), size=remaining_edges, replace=False)
        chosen_positions = zero_positions[chosen_idx]
        mask_np[chosen_positions[:, 0], chosen_positions[:, 1]] = 1.0

    mask = torch.tensor(mask_np, dtype=torch.float32)

    ptg = {}
    for p in range(P):
        genes = np.where(mask_np[p] > 0)[0]
        ptg[f"P{p}"] = list(map(int, genes))

    return mask, ptg


def mask_stats(mask: torch.Tensor) -> Dict[str, float]:
    P, G = mask.shape
    n_edges = int(mask.detach().cpu().sum().item())
    possible_edges = int(P * G)
    density_actual = n_edges / possible_edges
    sparsity_actual = 1.0 - density_actual

    row_degrees = mask.detach().cpu().sum(dim=1).numpy()
    col_degrees = mask.detach().cpu().sum(dim=0).numpy()

    return {
        "n_edges": n_edges,
        "possible_edges": possible_edges,
        "mask_density_actual": float(density_actual),
        "mask_sparsity_actual": float(sparsity_actual),
        "row_degree_mean": float(np.mean(row_degrees)),
        "row_degree_min": float(np.min(row_degrees)),
        "row_degree_max": float(np.max(row_degrees)),
        "col_degree_mean": float(np.mean(col_degrees)),
        "col_degree_min": float(np.min(col_degrees)),
        "col_degree_max": float(np.max(col_degrees)),
    }


def synth_X_with_latents(
    n: int,
    G: int,
    ptg: Dict[str, List[int]],
    beta: float = 0.7,
    noise: float = 0.6,
    seed: int = 123,
) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    P = len(ptg)

    latent_pathway_activity = rng.normal(size=(n, P))
    X = rng.normal(scale=noise, size=(n, G))

    for p, genes in enumerate(ptg.values()):
        if len(genes) > 0:
            X[:, genes] += beta * latent_pathway_activity[:, p:p + 1]

    X = (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-6)
    return torch.tensor(X, dtype=torch.float32)


# ---------------------- models ----------------------
class MaskedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, mask: torch.Tensor, bias: bool = True):
        super().__init__()
        assert mask.shape == (out_features, in_features)

        self.register_buffer("mask", mask.clone().float())
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight * self.mask, self.bias)


class PathwayBackbone(nn.Module):
    def __init__(self, mask: torch.Tensor, hidden: int = 64, p_drop: float = 0.15):
        super().__init__()
        P, G = mask.shape

        self.pathway = MaskedLinear(G, P, mask, bias=True)
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(P, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor):
        z = self.pathway(x)
        h = self.head(z)
        return h, z


class SurvivalNet(nn.Module):
    def __init__(self, mask, hidden=64, p_drop=0.15):
        super().__init__()
        self.backbone = PathwayBackbone(mask, hidden=hidden, p_drop=p_drop)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x):
        h, z = self.backbone(x)
        return self.out(h).squeeze(-1), z


class RegressionNet(nn.Module):
    def __init__(self, mask, hidden=64, p_drop=0.15):
        super().__init__()
        self.backbone = PathwayBackbone(mask, hidden=hidden, p_drop=p_drop)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x):
        h, z = self.backbone(x)
        return self.out(h).squeeze(-1), z


class BinaryNet(nn.Module):
    def __init__(self, mask, hidden=64, p_drop=0.15):
        super().__init__()
        self.backbone = PathwayBackbone(mask, hidden=hidden, p_drop=p_drop)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x):
        h, z = self.backbone(x)
        return self.out(h).squeeze(-1), z


class MultiNet(nn.Module):
    def __init__(self, mask, K=5, hidden=64, p_drop=0.15):
        super().__init__()
        self.backbone = PathwayBackbone(mask, hidden=hidden, p_drop=p_drop)
        self.out = nn.Linear(hidden, K)

    def forward(self, x):
        h, z = self.backbone(x)
        return self.out(h), z


# ---------------------- metrics ----------------------
@torch.no_grad()
def concordance_index(risk: torch.Tensor, time: torch.Tensor, event: torch.Tensor) -> float:
    r = risk.detach().cpu().numpy()
    t = time.detach().cpu().numpy()
    e = event.detach().cpu().numpy()

    num = 0.0
    den = 0.0
    n = len(r)

    for i in range(n):
        for j in range(n):
            if t[i] < t[j] and e[i] == 1:
                den += 1
                if r[i] > r[j]:
                    num += 1
                elif r[i] == r[j]:
                    num += 0.5

    return float(num / max(den, 1))


@torch.no_grad()
def simulate_times_from_risk(risk: torch.Tensor, censor_rate: float = 0.05):
    r = risk.detach().cpu().numpy()
    lam = np.exp(r)
    n = lam.shape[0]

    U = np.random.uniform(size=n)
    T = -np.log(U) / (lam + 1e-12)

    Uc = np.random.uniform(size=n)
    C = -np.log(Uc) / censor_rate

    time = np.minimum(T, C)
    event = (T <= C).astype(np.int64)

    return torch.tensor(time, dtype=torch.float32), torch.tensor(event, dtype=torch.long)


@torch.no_grad()
def best_positive_scale(wS, bS, wT, bT, lambda_b=1.0, eps=1e-12):
    num = (wS @ wT).item() + lambda_b * (bS * bT).item()
    den = (wS @ wS).item() + lambda_b * (bS * bS).item() + eps
    return max(0.0, num / den)


@torch.no_grad()
def pathway_weight_scores(teacher_backbone, student_backbone, ptg, lambda_b=1.0):
    tw = (teacher_backbone.pathway.weight * teacher_backbone.pathway.mask).detach().cpu()
    tb = teacher_backbone.pathway.bias.detach().cpu()

    sw = (student_backbone.pathway.weight * student_backbone.pathway.mask).detach().cpu()
    sb = student_backbone.pathway.bias.detach().cpu()

    rows = []

    for i, (pname, genes) in enumerate(ptg.items()):
        if len(genes) == 0:
            continue

        idx = torch.tensor(genes, dtype=torch.long)

        wT = tw[i, idx]
        wS = sw[i, idx]
        bT = tb[i]
        bS = sb[i]

        alpha = best_positive_scale(wS, bS, wT, bT, lambda_b=lambda_b)

        rel = torch.linalg.norm(alpha * wS - wT) / (torch.linalg.norm(wT) + 1e-12)
        cstd = (wT @ wS) / (torch.linalg.norm(wT) * torch.linalg.norm(wS) + 1e-12)

        rows.append({
            "pathway": pname,
            "pathway_index": i,
            "pathway_size": len(genes),
            "relL2": float(rel.item()),
            "cos_std": float(cstd.item()),
            "cos_pos": float(torch.clamp(cstd, min=0.0).item()),
        })

    return pd.DataFrame(rows)


@torch.no_grad()
def pathway_activation_scores(zT: torch.Tensor, zS: torch.Tensor):
    zT = zT.detach().cpu()
    zS = zS.detach().cpu()

    rows = []

    for p in range(zT.shape[1]):
        t = zT[:, p]
        s = zS[:, p]

        if torch.std(t) > 1e-6 and torch.std(s) > 1e-6:
            corr = torch.corrcoef(torch.stack([t, s]))[0, 1].item()
        else:
            corr = np.nan

        rows.append({
            "pathway_index": p,
            "act_corr": float(corr),
        })

    cos = F.cosine_similarity(zT, zS, dim=1)

    ss_res = torch.sum((zT - zS) ** 2)
    ss_tot = torch.sum((zT - zT.mean(dim=0)) ** 2)

    global_metrics = {
        "act_corr_mean": float(np.nanmean([r["act_corr"] for r in rows])),
        "act_cosine_mean": float(cos.mean().item()),
        "act_r2": float((1 - ss_res / (ss_tot + 1e-12)).item()),
    }

    return pd.DataFrame(rows), global_metrics


# ---------------------- configs ----------------------
@dataclass
class Config:
    G: int = 400
    P: int = 60
    size_min: int = 14
    size_max: int = 22
    overlap: float = 0.55
    mask_density: Optional[float] = None

    n_train: int = 12000
    n_test: int = 3000

    hidden: int = 64
    drop_teacher: float = 0.15
    drop_student: float = 0.0

    batch: int = 512
    epochs: int = 120
    lr: float = 2e-3
    wd: float = 0.0

    temp: float = 2.0
    gamma_risk: float = 6.0
    gamma_y: float = 6.0
    censor_rate: float = 0.05

    teacher_seed: int = 42
    data_seed: int = 123


@dataclass
class SupervisionConfig:
    mode: str = "none"
    lambda_supervision: float = 1.0
    sigma: float = 0.0
    rho: float = 1.0
    proxy_K: int = 30
    normalize_activations: bool = True
    include_bias_weight_loss: bool = True


TASKS4 = ["binary", "multiclass", "regression", "survival"]


# ---------------------- task helpers ----------------------
def make_model(task: str, mask, cfg: Config, teacher: bool, K: int = 5):
    p_drop = cfg.drop_teacher if teacher else cfg.drop_student

    if task == "binary":
        return BinaryNet(mask, hidden=cfg.hidden, p_drop=p_drop)

    if task == "multiclass":
        return MultiNet(mask, K=K, hidden=cfg.hidden, p_drop=p_drop)

    if task == "regression":
        return RegressionNet(mask, hidden=cfg.hidden, p_drop=p_drop)

    if task == "survival":
        return SurvivalNet(mask, hidden=cfg.hidden, p_drop=p_drop)

    raise ValueError(f"Unknown task: {task}")


def output_loss(task: str, pred, target, cfg: Config):
    if task == "multiclass":
        return F.kl_div(
            F.log_softmax(pred / cfg.temp, dim=1),
            target,
            reduction="batchmean",
        ) * (cfg.temp ** 2)

    return F.mse_loss(pred, target)


@torch.no_grad()
def teacher_targets(task: str, teacher, Xtr, Xte, cfg: Config, device: str):
    teacher.eval()

    if task == "multiclass":
        tlog_tr, _ = teacher(Xtr)
        tlog_te, _ = teacher(Xte)

        target_tr = F.softmax(tlog_tr / cfg.temp, dim=1).detach()
        target_te = F.softmax(tlog_te / cfg.temp, dim=1).detach()

        aux = {
            "teacher_hard": torch.argmax(tlog_te, dim=1),
        }

        teacher_ref = {
            "teacher_kd_kl": 0.0,
            "teacher_acc_vs_teacher": 1.0,
        }

        return target_tr, target_te, aux, teacher_ref

    raw_tr, _ = teacher(Xtr)
    raw_te, _ = teacher(Xte)

    if task == "binary":
        target_tr = raw_tr.detach()
        target_te = raw_te.detach()

        aux = {
            "teacher_hard": (raw_te > 0).long(),
        }

        teacher_ref = {
            "teacher_kd_mse_logit": 0.0,
            "teacher_acc_vs_teacher": 1.0,
        }

        return target_tr, target_te, aux, teacher_ref

    if task == "regression":
        mu = raw_tr.mean()
        sd = raw_tr.std()

        target_tr = (cfg.gamma_y * (raw_tr - mu) / (sd + 1e-8)).detach()
        target_te = (cfg.gamma_y * (raw_te - mu) / (sd + 1e-8)).detach()

        aux = {}

        teacher_ref = {
            "teacher_mse_amp": 0.0,
            "teacher_r2_amp": 1.0,
        }

        return target_tr, target_te, aux, teacher_ref

    if task == "survival":
        mu = raw_tr.mean()
        sd = raw_tr.std()

        target_tr = (cfg.gamma_risk * (raw_tr - mu) / (sd + 1e-8)).detach()
        target_te = (cfg.gamma_risk * (raw_te - mu) / (sd + 1e-8)).detach()

        t_te, e_te = simulate_times_from_risk(target_te, censor_rate=cfg.censor_rate)

        aux = {
            "time": t_te.to(device),
            "event": e_te.to(device),
        }

        teacher_ref = {
            "teacher_kd_mse_risk_amp": 0.0,
            "teacher_cindex": concordance_index(target_te, aux["time"], aux["event"]),
        }

        return target_tr, target_te, aux, teacher_ref

    raise ValueError(task)


@torch.no_grad()
def activation_normalizer(teacher, Xtr: torch.Tensor):
    teacher.eval()
    _, z = teacher(Xtr)
    mu = z.mean(dim=0, keepdim=True).detach()
    sd = z.std(dim=0, keepdim=True).detach().clamp_min(1e-6)
    return mu, sd
    
@torch.no_grad()
def precompute_teacher_activations(
    teacher,
    Xtr: torch.Tensor,
    z_mu: torch.Tensor,
    z_sd: torch.Tensor,
    normalize_activations: bool,
    sigma: float,
    seed: int,
):

    teacher.eval()

    _, zT = teacher(Xtr)
    zTn = normalize_z(zT, z_mu, z_sd, normalize_activations)

    if sigma > 0:
        gen = torch.Generator(device=zTn.device)
        gen.manual_seed(seed)
        noise = torch.randn(
            zTn.shape,
            generator=gen,
            device=zTn.device,
            dtype=zTn.dtype,
        )
        zTn = zTn + sigma * noise

    return zTn.detach()


def normalize_z(z, mu, sd, enabled=True):
    if enabled:
        return (z - mu) / sd
    return z


def sample_observed_pathways(P: int, rho: float, seed: int, device: str) -> Optional[torch.Tensor]:
    rho = max(0.0, min(1.0, rho))
    k = int(math.floor(rho * P))

    if k <= 0:
        return None

    rng = np.random.default_rng(seed)
    idx = rng.choice(np.arange(P), size=k, replace=False)
    idx.sort()

    return torch.tensor(idx, dtype=torch.long, device=device)


def make_proxy_matrix(K: int, P: int, seed: int, device: str):
    rng = np.random.default_rng(seed)

    A = rng.normal(
        0.0,
        1.0 / math.sqrt(max(K, 1)),
        size=(K, P),
    ).astype(np.float32)

    A /= (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)

    return torch.tensor(A, dtype=torch.float32, device=device)


def supervision_loss(
    sup: SupervisionConfig,
    teacher,
    student,
    xb: torch.Tensor,
    zS: torch.Tensor,
    z_mu: torch.Tensor,
    z_sd: torch.Tensor,
    obs_idx: Optional[torch.Tensor],
    proxy_A: Optional[torch.Tensor],
    shuffle_perm: Optional[torch.Tensor],
    fixed_zT_batch: Optional[torch.Tensor] = None,
):
    if sup.mode == "none" or sup.lambda_supervision <= 0:
        return zS.new_tensor(0.0)

    if sup.mode == "weights":
        wT = teacher.backbone.pathway.weight * teacher.backbone.pathway.mask
        wS = student.backbone.pathway.weight * student.backbone.pathway.mask

        loss = F.mse_loss(wS, wT.detach())

        if sup.include_bias_weight_loss:
            loss = loss + F.mse_loss(
                student.backbone.pathway.bias,
                teacher.backbone.pathway.bias.detach(),
            )

        return loss

    zSn = normalize_z(zS, z_mu, z_sd, sup.normalize_activations)

    if fixed_zT_batch is not None:
        zTn = fixed_zT_batch
    else:
        with torch.no_grad():
            _, zT = teacher(xb)
            zTn = normalize_z(zT, z_mu, z_sd, sup.normalize_activations)

    if sup.mode in ("activation", "random_activation", "shuffled_activation"):
        if obs_idx is None:
            return zS.new_tensor(0.0)

        target = zTn
        pred = zSn

        if sup.mode == "random_activation":
            target = torch.randn_like(zTn)

        elif sup.mode == "shuffled_activation":
            if shuffle_perm is None:
                raise RuntimeError("shuffle_perm required for shuffled_activation")
            target = zTn[:, shuffle_perm]

        target_obs = target[:, obs_idx]
        pred_obs = pred[:, obs_idx]

        return F.mse_loss(pred_obs, target_obs.detach())

    if sup.mode == "proxy":
        if proxy_A is None:
            raise RuntimeError("proxy_A required for proxy mode")

        y_proxy = zTn @ proxy_A.t()
        y_student = zSn @ proxy_A.t()

        return F.mse_loss(y_student, y_proxy.detach())

    raise ValueError(f"Unknown supervision mode: {sup.mode}")


@torch.no_grad()
def evaluate_task(task: str, pred, target_te, aux, cfg: Config):
    if task == "multiclass":
        kd = output_loss(task, pred, target_te, cfg).item()
        acc = (torch.argmax(pred, dim=1) == aux["teacher_hard"]).float().mean().item()

        return {
            "kd_kl": float(kd),
            "acc_vs_teacher": float(acc),
        }

    if task == "binary":
        mse = F.mse_loss(pred, target_te).item()
        acc = ((pred > 0).long() == aux["teacher_hard"]).float().mean().item()

        return {
            "kd_mse_logit": float(mse),
            "acc_vs_teacher": float(acc),
        }

    if task == "regression":
        mse = F.mse_loss(pred, target_te).item()

        ss_res = torch.sum((target_te - pred) ** 2)
        ss_tot = torch.sum((target_te - target_te.mean()) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-12)

        return {
            "mse_amp": float(mse),
            "r2_amp": float(r2.item()),
        }

    if task == "survival":
        mse = F.mse_loss(pred, target_te).item()
        cidx = concordance_index(pred, aux["time"], aux["event"])

        return {
            "kd_mse_risk_amp": float(mse),
            "cindex": float(cidx),
        }

    raise ValueError(task)


# ---------------------- core runner ----------------------
def run_one_task(
    task: str,
    mask: torch.Tensor,
    ptg: Dict[str, List[int]],
    Xtr: torch.Tensor,
    Xte: torch.Tensor,
    device: str,
    cfg: Config,
    sup: SupervisionConfig,
    n_students: int,
    seeds: List[int],
    K: int = 5,
    condition_id: str = "single",
    mask_metadata: Optional[Dict[str, float]] = None,
):
    teacher = make_model(task, mask, cfg, teacher=True, K=K).to(device)
    teacher.eval()

    target_tr, target_te, aux, teacher_ref = teacher_targets(task, teacher, Xtr, Xte, cfg, device)
    z_mu, z_sd = activation_normalizer(teacher, Xtr)
    
    fixed_noisy_zT_train = None

    if sup.mode in ("activation", "proxy", "shuffled_activation"):
        fixed_noisy_zT_train = precompute_teacher_activations(
            teacher=teacher,
            Xtr=Xtr,
            z_mu=z_mu,
            z_sd=z_sd,
            normalize_activations=sup.normalize_activations,
            sigma=sup.sigma,
            seed=cfg.teacher_seed + 777,
        )

    rows = []
    per_path_rows = []

    for s in seeds[:n_students]:
        set_all_seeds(s)

        student = make_model(task, mask, cfg, teacher=False, K=K).to(device)
        opt = torch.optim.Adam(student.parameters(), lr=cfg.lr, weight_decay=cfg.wd)

        sample_idx = torch.arange(Xtr.shape[0], device=Xtr.device)

        ds = DataLoader(
            TensorDataset(Xtr, target_tr, sample_idx),
            batch_size=cfg.batch,
            shuffle=True,
        )

        P = mask.shape[0]

        obs_idx = sample_observed_pathways(
            P,
            sup.rho,
            seed=s + 100_003,
            device=device,
        )

        if sup.mode == "proxy":
            proxy_A = make_proxy_matrix(
                sup.proxy_K,
                P,
                seed=s + 200_003,
                device=device,
            )
        else:
            proxy_A = None

        if sup.mode == "shuffled_activation":
            shuffle_perm = torch.randperm(P, device=device)
        else:
            shuffle_perm = None

        student.train()

        for _epoch in range(cfg.epochs):
            for xb, yb, idxb in ds:
                pred, zS = student(xb)

                loss_out = output_loss(task, pred, yb, cfg)

                if fixed_noisy_zT_train is not None:
                    fixed_zT_batch = fixed_noisy_zT_train[idxb]
                else:
                    fixed_zT_batch = None

                loss_sup = supervision_loss(
                    sup=sup,
                    teacher=teacher,
                    student=student,
                    xb=xb,
                    zS=zS,
                    z_mu=z_mu,
                    z_sd=z_sd,
                    obs_idx=obs_idx,
                    proxy_A=proxy_A,
                    shuffle_perm=shuffle_perm,
                    fixed_zT_batch=fixed_zT_batch,
                )

                loss = loss_out + sup.lambda_supervision * loss_sup

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

        student.eval()

        with torch.no_grad():
            pred_te, zS_te = student(Xte)
            _, zT_te = teacher(Xte)

            perf = evaluate_task(task, pred_te, target_te, aux, cfg)
            act_path_df, act_global = pathway_activation_scores(zT_te, zS_te)

        weight_path_df = pathway_weight_scores(
            teacher.backbone,
            student.backbone,
            ptg,
            lambda_b=1.0,
        )

        path_df = weight_path_df.merge(
            act_path_df,
            on="pathway_index",
            how="left",
        )

        path_df["task"] = task
        path_df["student_seed"] = s
        path_df["condition_id"] = condition_id

        path_df["mode"] = sup.mode
        path_df["lambda_supervision"] = sup.lambda_supervision
        path_df["sigma"] = sup.sigma
        path_df["rho"] = sup.rho
        path_df["proxy_K"] = sup.proxy_K

        path_df["P"] = cfg.P
        path_df["G"] = cfg.G
        path_df["size_min"] = cfg.size_min
        path_df["size_max"] = cfg.size_max
        path_df["overlap"] = cfg.overlap
        path_df["mask_density_requested"] = cfg.mask_density

        if mask_metadata:
            for k_meta, v_meta in mask_metadata.items():
                path_df[k_meta] = v_meta

        per_path_rows.append(path_df)

        summary = {
            "task": task,
            "student_seed": s,
            "condition_id": condition_id,

            "mode": sup.mode,
            "lambda_supervision": sup.lambda_supervision,
            "sigma": sup.sigma,
            "rho": sup.rho,
            "proxy_K": sup.proxy_K,

            "P": cfg.P,
            "G": cfg.G,
            "size_min": cfg.size_min,
            "size_max": cfg.size_max,
            "mean_pathway_size": float(np.mean([len(v) for v in ptg.values()])),
            "overlap": cfg.overlap,
            "mask_density_requested": cfg.mask_density,

            "n_observed_pathways": 0 if obs_idx is None else int(obs_idx.numel()),
            "epochs": cfg.epochs,
            "lr": cfg.lr,

            **teacher_ref,
            **perf,

            "relL2_mean": float(path_df["relL2"].mean()),
            "relL2_median": float(path_df["relL2"].median()),
            "cos_std_mean": float(path_df["cos_std"].mean()),
            "cos_std_median": float(path_df["cos_std"].median()),
            "cos_pos_mean": float(path_df["cos_pos"].mean()),
            "cos_pos_median": float(path_df["cos_pos"].median()),

            **act_global,
        }

        if mask_metadata:
            summary.update(mask_metadata)

        rows.append(summary)

    return pd.DataFrame(rows), pd.concat(per_path_rows, ignore_index=True)


# ---------------------- sweep construction ----------------------
def build_conditions(args):
    """
    Returns tuples:
      condition_id,
      SupervisionConfig,
      P_override,
      size_range_override,
      overlap_override,
      density_override
    """
    conditions = []

    if args.sweep == "none":
        sup = SupervisionConfig(
            mode=args.mode,
            lambda_supervision=args.lambda_supervision,
            sigma=args.sigma,
            rho=args.rho,
            proxy_K=args.proxy_K,
            normalize_activations=not args.no_normalize_activations,
            include_bias_weight_loss=not args.no_bias_weight_loss,
        )

        conditions.append((
            f"{args.mode}_lambda{args.lambda_supervision}_sigma{args.sigma}_rho{args.rho}_K{args.proxy_K}",
            sup,
            None,
            None,
            None,
            args.mask_density,
        ))

        return conditions

    if args.sweep == "phase":
        sigma_values = parse_csv_floats(args.sigma_values)
        rho_values = parse_csv_floats(args.rho_values)

        for sigma, rho in itertools.product(sigma_values, rho_values):
            mode = "none" if rho == 0.0 else "activation"

            sup = SupervisionConfig(
                mode=mode,
                lambda_supervision=args.lambda_supervision,
                sigma=sigma,
                rho=rho,
            )

            conditions.append((
                f"phase_sigma{sigma}_rho{rho}",
                sup,
                None,
                None,
                None,
                args.mask_density,
            ))

        return conditions

    if args.sweep == "noise":
        for sigma in [0.0, 0.1, 0.5, 1.0, 2.0]:
            sup = SupervisionConfig(
                mode="activation",
                lambda_supervision=args.lambda_supervision,
                sigma=sigma,
                rho=1.0,
            )

            conditions.append((
                f"noise_sigma{sigma}_rho1",
                sup,
                None,
                None,
                None,
                args.mask_density,
            ))

        conditions.append((
            "noise_output_only",
            SupervisionConfig(mode="none"),
            None,
            None,
            None,
            args.mask_density,
        ))

        return conditions

    if args.sweep == "coverage":
        for rho in [1.0, 0.75, 0.50, 0.25, 0.10, 0.0]:
            mode = "none" if rho == 0.0 else "activation"

            sup = SupervisionConfig(
                mode=mode,
                lambda_supervision=args.lambda_supervision,
                sigma=0.1,
                rho=rho,
            )

            conditions.append((
                f"coverage_sigma0.1_rho{rho}",
                sup,
                None,
                None,
                None,
                args.mask_density,
            ))

        return conditions

    if args.sweep == "proxy":
        for k in [60, 45, 30, 15, 6]:
            k_eff = min(k, args.P)

            sup = SupervisionConfig(
                mode="proxy",
                lambda_supervision=args.lambda_supervision,
                sigma=0.5,
                rho=1.0,
                proxy_K=k_eff,
            )

            conditions.append((
                f"proxy_K{k_eff}_sigma0.5",
                sup,
                None,
                None,
                None,
                args.mask_density,
            ))

        return conditions

    if args.sweep == "lambda":
        for lam in parse_csv_floats(args.lambda_values):
            selected_mode = args.mode if args.mode != "none" else "activation"

            sup = SupervisionConfig(
                mode=selected_mode,
                lambda_supervision=lam,
                sigma=args.sigma,
                rho=args.rho,
                proxy_K=args.proxy_K,
            )

            conditions.append((
                f"lambda_{lam}_{sup.mode}_sigma{sup.sigma}_rho{sup.rho}",
                sup,
                None,
                None,
                None,
                args.mask_density,
            ))

        return conditions

    if args.sweep == "main_sim":
        P_values = parse_csv_ints(args.P_values)
        density_values = parse_csv_floats(args.density_values)

        modes = [
            (
                "none",
                SupervisionConfig(
                    mode="none",
                    lambda_supervision=0.0,
                    sigma=0.0,
                    rho=0.0,
                ),
            ),
            (
                "weights",
                SupervisionConfig(
                    mode="weights",
                    lambda_supervision=args.lambda_supervision,
                    sigma=0.0,
                    rho=1.0,
                ),
            ),
            (
                "activation_perfect",
                SupervisionConfig(
                    mode="activation",
                    lambda_supervision=args.lambda_supervision,
                    sigma=0.0,
                    rho=1.0,
                ),
            ),
            (
                "activation_noisy",
                SupervisionConfig(
                    mode="activation",
                    lambda_supervision=args.lambda_supervision,
                    sigma=0.5,
                    rho=1.0,
                ),
            ),
            (
                "activation_partial",
                SupervisionConfig(
                    mode="activation",
                    lambda_supervision=args.lambda_supervision,
                    sigma=0.1,
                    rho=0.25,
                ),
            ),
            (
                "random_activation",
                SupervisionConfig(
                    mode="random_activation",
                    lambda_supervision=args.lambda_supervision,
                    sigma=0.0,
                    rho=1.0,
                ),
            ),
            (
                "shuffled_activation",
                SupervisionConfig(
                    mode="shuffled_activation",
                    lambda_supervision=args.lambda_supervision,
                    sigma=0.0,
                    rho=1.0,
                ),
            ),
        ]

        for P_val, density_val, (label, sup) in itertools.product(
            P_values,
            density_values,
            modes,
        ):
            conditions.append((
                f"main_P{P_val}_density{density_val}_{label}",
                sup,
                P_val,
                None,
                None,
                density_val,
            ))

        return conditions

    if args.sweep == "sparsity":
        density_values = parse_csv_floats(args.density_values)

        for density_val in density_values:
            for label, sup in [
                (
                    "best_perfect",
                    SupervisionConfig(
                        mode="activation",
                        lambda_supervision=args.lambda_supervision,
                        sigma=0.0,
                        rho=1.0,
                    ),
                ),
                (
                    "poor",
                    SupervisionConfig(
                        mode="activation",
                        lambda_supervision=args.lambda_supervision,
                        sigma=2.0,
                        rho=0.10,
                    ),
                ),
            ]:
                conditions.append((
                    f"sparsity_density{density_val}_{label}",
                    sup,
                    None,
                    None,
                    None,
                    density_val,
                ))

        return conditions

    raise ValueError(f"Unknown sweep: {args.sweep}")


def run_condition(args, base_cfg: Config, condition):
    condition_id, sup, P_override, size_range_override, overlap_override, density_override = condition

    cfg = Config(**asdict(base_cfg))

    if P_override is not None:
        cfg.P = P_override

    if size_range_override is not None:
        cfg.size_min, cfg.size_max = size_range_override

    if overlap_override is not None:
        cfg.overlap = overlap_override

    if density_override is not None:
        cfg.mask_density = density_override

    device = get_device(args.device)

    set_all_seeds(cfg.teacher_seed)

    if cfg.mask_density is not None:
        mask, ptg = build_fixed_density_mask(
            G=cfg.G,
            P=cfg.P,
            density=cfg.mask_density,
            seed=cfg.teacher_seed,
            min_edges_per_pathway=args.min_edges_per_pathway,
        )
    else:
        mask, ptg = build_sparse_overlapping_mask(
            G=cfg.G,
            P=cfg.P,
            size_range=(cfg.size_min, cfg.size_max),
            overlap_frac=cfg.overlap,
            seed=cfg.teacher_seed,
        )

    mask_metadata = mask_stats(mask)

    X_all = synth_X_with_latents(
        n=cfg.n_train + cfg.n_test,
        G=cfg.G,
        ptg=ptg,
        beta=args.beta,
        noise=args.noise,
        seed=cfg.data_seed,
    )

    Xtr = X_all[:cfg.n_train].to(device)
    Xte = X_all[cfg.n_train:].to(device)

    mask = mask.to(device)

    if args.task == "all":
        tasks = TASKS4
    else:
        tasks = [args.task]

    seeds = [args.student_seed_base + i for i in range(max(args.students, 1))]

    summary_frames = []
    pathway_frames = []

    for task in tasks:
        print(
            f">> {condition_id} | task={task} | mode={sup.mode} "
            f"| P={cfg.P} | density={cfg.mask_density} "
            f"| actual_density={mask_metadata['mask_density_actual']:.5f}",
            flush=True,
        )

        summary, per_path = run_one_task(
            task=task,
            mask=mask,
            ptg=ptg,
            Xtr=Xtr,
            Xte=Xte,
            device=device,
            cfg=cfg,
            sup=sup,
            n_students=args.students,
            seeds=seeds,
            K=args.K,
            condition_id=condition_id,
            mask_metadata=mask_metadata,
        )

        summary_frames.append(summary)
        pathway_frames.append(per_path)

    return (
        pd.concat(summary_frames, ignore_index=True),
        pd.concat(pathway_frames, ignore_index=True),
    )


# ---------------------- plotting ----------------------
def make_basic_figures(outdir: str, summary_df: pd.DataFrame):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"Skipping plots: matplotlib unavailable ({e})")
        return

    figdir = os.path.join(outdir, "figures")
    ensure_dir(figdir)

    phase = summary_df[
        summary_df["condition_id"].astype(str).str.startswith("phase_")
    ].copy()

    if not phase.empty:
        for task, df_task in phase.groupby("task"):
            grouped = (
                df_task
                .groupby(["sigma", "rho"], as_index=False)["cos_std_mean"]
                .mean()
            )

            piv = grouped.pivot(
                index="sigma",
                columns="rho",
                values="cos_std_mean",
            )

            fig = plt.figure(figsize=(7, 4.5))
            ax = plt.gca()

            im = ax.imshow(piv.values, aspect="auto", origin="lower")

            ax.set_xticks(range(len(piv.columns)))
            ax.set_xticklabels([str(c) for c in piv.columns])

            ax.set_yticks(range(len(piv.index)))
            ax.set_yticklabels([str(i) for i in piv.index])

            ax.set_xlabel("rho: observed pathway fraction")
            ax.set_ylabel("sigma: activation noise")
            ax.set_title(f"Phase diagram - {task}")

            fig.colorbar(im, ax=ax, label="mean first-layer cosine")
            fig.tight_layout()

            fig.savefig(
                os.path.join(figdir, f"phase_heatmap_{task}.png"),
                dpi=200,
            )
            plt.close(fig)

    main = summary_df[
        summary_df["condition_id"].astype(str).str.startswith("main_")
    ].copy()

    if not main.empty:
        for task, df_task in main.groupby("task"):
            fig = plt.figure(figsize=(8, 5))
            ax = plt.gca()

            for mode, df_m in df_task.groupby("mode"):
                agg = (
                    df_m
                    .groupby(["P", "mask_density_actual"], as_index=False)["cos_std_mean"]
                    .mean()
                    .sort_values(["mask_density_actual", "P"])
                )

                for density, df_d in agg.groupby("mask_density_actual"):
                    label = f"{mode}, density={density:.3f}"
                    ax.plot(
                        df_d["P"],
                        df_d["cos_std_mean"],
                        marker="o",
                        label=label,
                    )

            ax.set_xlabel("input -> h1 dimension P")
            ax.set_ylabel("mean first-layer cosine")
            ax.set_title(f"Recovery vs P and density - {task}")
            ax.legend(fontsize=7)

            fig.tight_layout()
            fig.savefig(
                os.path.join(figdir, f"main_recovery_vs_P_density_{task}.png"),
                dpi=200,
            )
            plt.close(fig)

        for task, df_task in main.groupby("task"):
            fig = plt.figure(figsize=(8, 5))
            ax = plt.gca()

            for mode, df_m in df_task.groupby("mode"):
                agg = (
                    df_m
                    .groupby("mask_density_actual", as_index=False)["cos_std_mean"]
                    .mean()
                    .sort_values("mask_density_actual")
                )

                ax.plot(
                    agg["mask_density_actual"],
                    agg["cos_std_mean"],
                    marker="o",
                    label=mode,
                )

            ax.set_xlabel("first-layer mask density")
            ax.set_ylabel("mean first-layer cosine")
            ax.set_title(f"Recovery vs mask density - {task}")
            ax.legend(fontsize=8)

            fig.tight_layout()
            fig.savefig(
                os.path.join(figdir, f"main_recovery_vs_density_{task}.png"),
                dpi=200,
            )
            plt.close(fig)


# ---------------------- main ----------------------
def main(args):
    ensure_dir(args.outdir)

    base_cfg = Config(
        G=args.G,
        P=args.P,
        size_min=args.size_min,
        size_max=args.size_max,
        overlap=args.overlap,
        mask_density=args.mask_density,

        n_train=args.n_train,
        n_test=args.n_test,

        hidden=args.hidden,
        drop_teacher=args.drop_teacher,
        drop_student=args.drop_student,

        batch=args.batch,
        epochs=args.epochs,
        lr=args.lr,
        wd=args.wd,

        temp=args.temp,
        gamma_risk=args.gamma_risk,
        gamma_y=args.gamma_y,
        censor_rate=args.censor_rate,

        teacher_seed=args.teacher_seed,
        data_seed=args.data_seed,
    )

    all_summary = []
    all_pathways = []

    conditions = build_conditions(args)

    print(
        f"Running {len(conditions)} condition(s) on device={get_device(args.device)}",
        flush=True,
    )

    for condition in conditions:
        summary, pathways = run_condition(args, base_cfg, condition)

        all_summary.append(summary)
        all_pathways.append(pathways)

        partial_summary = pd.concat(all_summary, ignore_index=True)
        partial_pathways = pd.concat(all_pathways, ignore_index=True)

        partial_summary.to_csv(
            os.path.join(args.outdir, "metrics_summary.csv"),
            index=False,
        )

        partial_pathways.to_csv(
            os.path.join(args.outdir, "metrics_per_pathway.csv"),
            index=False,
        )

    summary_df = pd.concat(all_summary, ignore_index=True)
    pathways_df = pd.concat(all_pathways, ignore_index=True)

    summary_path = os.path.join(args.outdir, "metrics_summary.csv")
    pathways_path = os.path.join(args.outdir, "metrics_per_pathway.csv")

    summary_df.to_csv(summary_path, index=False)
    pathways_df.to_csv(pathways_path, index=False)

    print("saved:", summary_path)
    print("saved:", pathways_path)

    if args.make_plots:
        make_basic_figures(args.outdir, summary_df)
        print("saved figures in:", os.path.join(args.outdir, "figures"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Activation-/weight-supervised BINN teacher-student experiments"
    )

    parser.add_argument(
        "--task",
        type=str,
        default="all",
        choices=["all", "binary", "multiclass", "regression", "survival"],
    )

    parser.add_argument(
        "--sweep",
        type=str,
        default="none",
        choices=[
            "none",
            "noise",
            "coverage",
            "proxy",
            "phase",
            "lambda",
            "sparsity",
            "main_sim",
        ],
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="none",
        choices=[
            "none",
            "weights",
            "activation",
            "proxy",
            "random_activation",
            "shuffled_activation",
        ],
    )

    parser.add_argument("--students", type=int, default=20)
    parser.add_argument("--outdir", type=str, default="results_supervision")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--make_plots", action="store_true")

    # Supervision parameters
    parser.add_argument("--lambda_supervision", type=float, default=1.0)
    parser.add_argument("--sigma", type=float, default=0.0)
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--proxy_K", type=int, default=30)
    parser.add_argument("--no_normalize_activations", action="store_true")
    parser.add_argument("--no_bias_weight_loss", action="store_true")

    # Data/model
    parser.add_argument("--G", type=int, default=400)
    parser.add_argument("--P", type=int, default=60)
    parser.add_argument("--P_values", type=str, default="20,40,60,100")

    # Old pathway-size options kept for compatibility.
    # They are not used by main_sim when density_values is provided.
    parser.add_argument("--size_min", type=int, default=14)
    parser.add_argument("--size_max", type=int, default=22)
    parser.add_argument("--pathway_size_ranges", type=str, default="6-10,14-22,30-45")
    parser.add_argument("--overlap", type=float, default=0.55)

    # Real sparsity/density controls
    parser.add_argument(
        "--mask_density",
        type=float,
        default=None,
        help="Global first-layer density: active_edges / (G * P). If set, overrides pathway-size mask.",
    )

    parser.add_argument(
        "--density_values",
        type=str,
        default="0.02,0.05,0.10,0.20",
        help="Comma-separated densities used by --sweep main_sim or --sweep sparsity.",
    )

    parser.add_argument(
        "--min_edges_per_pathway",
        type=int,
        default=1,
        help="Minimum active gene connections per pathway when using density-based masks.",
    )

    parser.add_argument("--n_train", type=int, default=12000)
    parser.add_argument("--n_test", type=int, default=3000)
    parser.add_argument("--beta", type=float, default=0.7)
    parser.add_argument("--noise", type=float, default=0.6)

    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--drop_teacher", type=float, default=0.15)
    parser.add_argument("--drop_student", type=float, default=0.0)

    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--wd", type=float, default=0.0)

    # Task specifics
    parser.add_argument("--temp", type=float, default=2.0)
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--gamma_risk", type=float, default=6.0)
    parser.add_argument("--gamma_y", type=float, default=6.0)
    parser.add_argument("--censor_rate", type=float, default=0.05)

    # Seeds
    parser.add_argument("--teacher_seed", type=int, default=42)
    parser.add_argument("--data_seed", type=int, default=123)
    parser.add_argument("--student_seed_base", type=int, default=9000)
    
    parser.add_argument(
    "--lambda_values",
    type=str,
    default="0.01,0.1,0.5,1,5,10",
    help="Comma-separated lambda values used by --sweep lambda.",
)

    parser.add_argument(
    "--sigma_values",
    type=str,
    default="0,0.1,0.5,1,2,5,10",
    help="Comma-separated sigma values used by --sweep phase.",
)

    parser.add_argument(
    "--rho_values",
    type=str,
    default="0,0.1,0.25,0.5,0.75,1",
    help="Comma-separated rho values used by --sweep phase.",
)

    main(parser.parse_args())
