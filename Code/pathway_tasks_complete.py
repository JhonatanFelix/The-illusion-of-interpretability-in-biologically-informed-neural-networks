import argparse, math, os, random
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd


# ---------------------- utils ----------------------
def get_device(arg_device: str) -> str:
    if arg_device == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if arg_device == "cpu":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"

def set_all_seeds(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


# ---------------------- data ----------------------
def build_sparse_overlapping_mask(
    G: int, P: int, size_range: Tuple[int, int] = (14, 22), overlap_frac: float = 0.55, seed: int = 0
) -> Tuple[torch.Tensor, Dict[str, List[int]]]:
    rng = np.random.default_rng(seed)
    all_genes = np.arange(G, dtype=int)
    gene_lists = []; prev_pool = np.array([], dtype=int)
    for _ in range(P):
        size = int(rng.integers(size_range[0], size_range[1] + 1))
        target_overlap = int(math.floor(size * overlap_frac))
        k_overlap = min(target_overlap, prev_pool.size)
        overlap_part = rng.choice(prev_pool, size=k_overlap, replace=False) if k_overlap>0 else np.array([], dtype=int)
        remaining = np.setdiff1d(all_genes, overlap_part, assume_unique=False)
        k_fresh = min(size - k_overlap, remaining.size)
        fresh_part = rng.choice(remaining, size=k_fresh, replace=False) if k_fresh>0 else np.array([], dtype=int)
        genes = np.unique(np.concatenate([overlap_part, fresh_part]))
        gene_lists.append(genes)
        prev_pool = np.unique(np.concatenate([prev_pool, genes]))
    mask = torch.zeros(P, G, dtype=torch.float32)
    for i, genes in enumerate(gene_lists):
        if genes.size > 0:
            mask[i, torch.as_tensor(genes, dtype=torch.long)] = 1.0
    ptg = {f"P{i}": list(map(int, gene_lists[i])) for i in range(P)}
    return mask, ptg

def synth_X_with_latents(
    n: int, G: int, ptg: Dict[str, List[int]], beta: float = 0.7, noise: float = 0.6, seed: int = 123
) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    P = len(ptg)
    A = rng.normal(size=(n, P))  # pathway activities
    X = rng.normal(scale=noise, size=(n, G))
    for p, genes in enumerate(ptg.values()):
        if len(genes)==0: continue
        X[:, genes] += beta * A[:, p:p+1]
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
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight * self.mask, self.bias)

class PathwayBackbone(nn.Module):
    def __init__(self, mask: torch.Tensor, hidden: int = 64, p_drop: float = 0.15):
        super().__init__()
        P, G = mask.shape
        self.pathway = MaskedLinear(G, P, mask, bias=True)
        self.head = nn.Sequential(
            nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(P, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
    def forward(self, x: torch.Tensor):
        z = self.pathway(x)
        h = self.head(z)
        return h, z

# Survival
class SurvivalTeacher(nn.Module):
    def __init__(self, mask, hidden=64, p_drop=0.15):
        super().__init__()
        self.backbone = PathwayBackbone(mask, hidden=hidden, p_drop=p_drop)
        self.risk_head = nn.Linear(hidden, 1)
    def forward(self, x):
        h, z = self.backbone(x)
        return self.risk_head(h).squeeze(-1), z

class SurvivalStudent(nn.Module):
    def __init__(self, mask, hidden=64, p_drop=0.0):
        super().__init__()
        self.backbone = PathwayBackbone(mask, hidden=hidden, p_drop=p_drop)
        self.risk_head = nn.Linear(hidden, 1)
    def forward(self, x):
        h, z = self.backbone(x)
        return self.risk_head(h).squeeze(-1), z

# Regression
class RegressionTeacher(nn.Module):
    def __init__(self, mask, hidden=64, p_drop=0.15):
        super().__init__()
        self.backbone = PathwayBackbone(mask, hidden=hidden, p_drop=p_drop)
        self.out = nn.Linear(hidden, 1)
    def forward(self, x):
        h, z = self.backbone(x)
        return self.out(h).squeeze(-1), z

class RegressionStudent(nn.Module):
    def __init__(self, mask, hidden=64, p_drop=0.0):
        super().__init__()
        self.backbone = PathwayBackbone(mask, hidden=hidden, p_drop=p_drop)
        self.out = nn.Linear(hidden, 1)
    def forward(self, x):
        h, z = self.backbone(x)
        return self.out(h).squeeze(-1), z

# Binary (nonlineare)
class BinaryTeacher(nn.Module):
    def __init__(self, mask, hidden=64, p_drop=0.15):
        super().__init__()
        self.backbone = PathwayBackbone(mask, hidden=hidden, p_drop=p_drop)
        self.out = nn.Linear(hidden, 1)
    def forward(self, x):
        h, z = self.backbone(x)
        return self.out(h).squeeze(-1), z

class BinaryStudent(nn.Module):
    def __init__(self, mask, hidden=64, p_drop=0.0):
        super().__init__()
        self.backbone = PathwayBackbone(mask, hidden=hidden, p_drop=p_drop)
        self.out = nn.Linear(hidden, 1)
    def forward(self, x):
        h, z = self.backbone(x)
        return self.out(h).squeeze(-1), z

# Multiclass
class MultiTeacher(nn.Module):
    def __init__(self, mask, K=5, hidden=64, p_drop=0.15):
        super().__init__()
        self.backbone = PathwayBackbone(mask, hidden=hidden, p_drop=p_drop)
        self.out = nn.Linear(hidden, K)
    def forward(self, x):
        h, z = self.backbone(x)
        return self.out(h), z

class MultiStudent(nn.Module):
    def __init__(self, mask, K=5, hidden=64, p_drop=0.0):
        super().__init__()
        self.backbone = PathwayBackbone(mask, hidden=hidden, p_drop=p_drop)
        self.out = nn.Linear(hidden, K)
    def forward(self, x):
        h, z = self.backbone(x)
        return self.out(h), z

# 1-layer lineare (media pathways -> logit)
class PathwayNet1L(nn.Module):
    def __init__(self, mask, use_relu=False):
        super().__init__()
        P, G = mask.shape
        self.pathway = MaskedLinear(G, P, mask, bias=True)
        self.use_relu = use_relu
        self.register_buffer("fixed_w", torch.ones(P, 1) / P)
        self.register_buffer("fixed_b", torch.zeros(1))
    def forward(self, x):
        z = self.pathway(x)
        if self.use_relu: z = F.relu(z)
        logit = z @ self.fixed_w + self.fixed_b
        return logit.squeeze(-1), z
        
@torch.no_grad()
def pathway_activation_alignment(zT: torch.Tensor, zS: torch.Tensor):
    """
    zT, zS: (N, P)

    Returns:
        dict with multiple alignment metrics
    """
    zT = zT.detach().cpu()
    zS = zS.detach().cpu()

    # --- 1. per-pathway correlation ---
    corrs = []
    for p in range(zT.shape[1]):
        t = zT[:, p]
        s = zS[:, p]
        if torch.std(t) > 1e-6 and torch.std(s) > 1e-6:
            c = torch.corrcoef(torch.stack([t, s]))[0, 1]
            corrs.append(c.item())
    corr_mean = float(np.mean(corrs)) if len(corrs) > 0 else np.nan

    # --- 2. cosine per sample ---
    cos = F.cosine_similarity(zT, zS, dim=1)
    cos_mean = float(cos.mean().item())

    # --- 3. global R2 ---
    ss_res = torch.sum((zT - zS) ** 2)
    ss_tot = torch.sum((zT - zT.mean(dim=0)) ** 2)
    r2 = float((1 - ss_res / (ss_tot + 1e-12)).item())

    return {
        "act_corr_mean": corr_mean,
        "act_cosine_mean": cos_mean,
        "act_r2": r2
    }


# ---------------------- metrics ----------------------
@torch.no_grad()
def concordance_index(risk: torch.Tensor, time: torch.Tensor, event: torch.Tensor) -> float:
    r = risk.detach().cpu().numpy()
    t = time.detach().cpu().numpy()
    e = event.detach().cpu().numpy()
    n = len(r); num = 0.0; den = 0.0
    for i in range(n):
        for j in range(n):
            if t[i] < t[j] and e[i]==1:
                den += 1
                if r[i] > r[j]: num += 1
                elif r[i] == r[j]: num += 0.5
    return float(num / max(den, 1))

@torch.no_grad()
def best_positive_scale(wS, bS, wT, bT, lambda_b=1.0, eps=1e-12):
    num = (wS @ wT).item() + lambda_b * (bS * bT).item()
    den = (wS @ wS).item() + lambda_b * (bS * bS).item() + eps
    return max(0.0, num / den)

@torch.no_grad()
def pathway_weight_scores_both(teacher_backbone, student_backbone, ptg, lambda_b=1.0):
    tw = (teacher_backbone.pathway.weight * teacher_backbone.pathway.mask).detach().cpu()
    tb = teacher_backbone.pathway.bias.detach().cpu()
    sw = (student_backbone.pathway.weight * student_backbone.pathway.mask).detach().cpu()
    sb = student_backbone.pathway.bias.detach().cpu()
    rel_errs, cos_std, cos_pos = [], [], []
    for i, genes in enumerate(ptg.values()):
        if len(genes)==0: continue
        idx = torch.tensor(genes, dtype=torch.long)
        wT, wS = tw[i, idx], sw[i, idx]
        bT, bS = tb[i], sb[i]
        alpha = best_positive_scale(wS, bS, wT, bT, lambda_b=lambda_b)
        rel = torch.linalg.norm(alpha*wS - wT) / (torch.linalg.norm(wT) + 1e-12)
        cstd = (wT @ wS) / (torch.linalg.norm(wT)*torch.linalg.norm(wS) + 1e-12)
        cpos = float(torch.clamp(cstd, min=0.0))
        rel_errs.append(rel.item()); cos_std.append(cstd.item()); cos_pos.append(cpos)
    return np.array(rel_errs), np.array(cos_std), np.array(cos_pos)


# ---------------------- survival sim ----------------------
def simulate_times_from_risk(risk: torch.Tensor, censor_rate: float = 0.05):
    r = risk.detach().cpu().numpy()
    lam = np.exp(r)
    n = lam.shape[0]
    U  = np.random.uniform(size=n);  T = -np.log(U)  / (lam + 1e-12)
    Uc = np.random.uniform(size=n);  C = -np.log(Uc) / censor_rate
    time  = np.minimum(T, C)
    event = (T <= C).astype(np.int64)
    return torch.tensor(time, dtype=torch.float32), torch.tensor(event, dtype=torch.long)


# ---------------------- config ----------------------
@dataclass
class Config:
    G: int = 400
    P: int = 60
    size_min: int = 14
    size_max: int = 22
    overlap: float = 0.55
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


# ---------------------- runners per task ----------------------
def run_multiclass(mask, ptg, Xtr, Xte, device, cfg: Config, n_students: int, seeds, K: int = 5):
    teacher = MultiTeacher(mask, K=K, hidden=cfg.hidden, p_drop=cfg.drop_teacher).to(device)
    teacher.eval()
    with torch.no_grad():
        tlog_tr, _ = teacher(Xtr.to(device))
        tlog_te, _ = teacher(Xte.to(device))
        tprob_tr = F.softmax(tlog_tr / cfg.temp, dim=1)
        tprob_te = F.softmax(tlog_te / cfg.temp, dim=1)
        yte_hard = torch.argmax(tlog_te, dim=1)

        # --- Teacher performance reference (vs its own hard labels) ---
        teacher_acc = 1.0  # by definition, argmax(tlog_te) == yte_hard
        teacher_kd_kl = 0.0  # KL(teacher||teacher)=0

    rows = []
    for s in seeds[:n_students]:
        set_all_seeds(s)
        student = MultiStudent(mask, K=K, hidden=cfg.hidden, p_drop=cfg.drop_student).to(device)
        opt = torch.optim.Adam(student.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
        ds = DataLoader(TensorDataset(Xtr.to(device), tprob_tr), batch_size=cfg.batch, shuffle=True)
        student.train()
        for _ in range(cfg.epochs):
            for xb, tpb in ds:
                slog, _ = student(xb)
                loss_kd = F.kl_div(F.log_softmax(slog / cfg.temp, dim=1), tpb, reduction="batchmean") * (cfg.temp**2)
                opt.zero_grad(); loss_kd.backward(); opt.step()
        student.eval()
        with torch.no_grad():
            slog_te, zS = student(Xte.to(device))
            _, zT = teacher(Xte.to(device))
            kd_te = F.kl_div(F.log_softmax(slog_te / cfg.temp, dim=1), tprob_te, reduction="batchmean") * (cfg.temp**2)
            acc = (torch.argmax(slog_te, dim=1) == yte_hard).float().mean()
            act_metrics = pathway_activation_alignment(zT, zS)

        rel, cs, cp = pathway_weight_scores_both(teacher.backbone, student.backbone, ptg, lambda_b=1.0)
        rows.append(dict(
            student_seed=s,

            # --- Teacher reference metrics ---
            teacher_kd_kl=float(teacher_kd_kl),
            teacher_acc_vs_teacher=float(teacher_acc),

            # --- Student metrics ---
            kd_kl=float(kd_te.item()),
            acc_vs_teacher=float(acc.item()),

            # --- Weight recovery (student vs teacher) ---
            relL2_mean=float(rel.mean()),
            cos_std_mean=float(cs.mean()),
            cos_pos_mean=float(cp.mean()),
            
            act_corr_mean=act_metrics["act_corr_mean"],
            act_cosine_mean=act_metrics["act_cosine_mean"],
            act_r2=act_metrics["act_r2"],
        ))
    return pd.DataFrame(rows)

def run_binary(mask, ptg, Xtr, Xte, device, cfg: Config, n_students: int, seeds):
    teacher = BinaryTeacher(mask, hidden=cfg.hidden, p_drop=cfg.drop_teacher).to(device)
    teacher.eval()
    with torch.no_grad():
        tlog_tr, _ = teacher(Xtr.to(device))
        tlog_te, _ = teacher(Xte.to(device))
        yte_hard = (tlog_te > 0).long()

        # --- Teacher performance reference (vs its own hard labels) ---
        teacher_acc = 1.0
        teacher_kd_mse = 0.0

    rows = []
    for s in seeds[:n_students]:
        set_all_seeds(s)
        student = BinaryStudent(mask, hidden=cfg.hidden, p_drop=cfg.drop_student).to(device)
        opt = torch.optim.Adam(student.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
        ds = DataLoader(TensorDataset(Xtr.to(device), tlog_tr), batch_size=cfg.batch, shuffle=True)
        student.train()
        for _ in range(cfg.epochs):
            for xb, tlog in ds:
                slog, _ = student(xb)
                loss = F.mse_loss(slog, tlog)
                opt.zero_grad(); loss.backward(); opt.step()
        student.eval()
        with torch.no_grad():
            slog_te, zS = student(Xte.to(device))
            _, zT = teacher(Xte.to(device))
            kd_mse = F.mse_loss(slog_te, tlog_te).item()
            acc = ((slog_te > 0).long() == yte_hard).float().mean().item()
            act_metrics = pathway_activation_alignment(zT, zS)

        rel, cs, cp = pathway_weight_scores_both(teacher.backbone, student.backbone, ptg, lambda_b=1.0)
        rows.append(dict(
            student_seed=s,

            # --- Teacher reference metrics ---
            teacher_kd_mse_logit=float(teacher_kd_mse),
            teacher_acc_vs_teacher=float(teacher_acc),

            # --- Student metrics ---
            kd_mse_logit=float(kd_mse),
            acc_vs_teacher=float(acc),

            # --- Weight recovery (student vs teacher) ---
            relL2_mean=float(rel.mean()),
            cos_std_mean=float(cs.mean()),
            cos_pos_mean=float(cp.mean()),
            
            act_corr_mean=act_metrics["act_corr_mean"],
            act_cosine_mean=act_metrics["act_cosine_mean"],
            act_r2=act_metrics["act_r2"],
        ))
    return pd.DataFrame(rows)

def run_binary_1layer(mask, ptg, Xtr, Xte, device, cfg: Config, n_students: int, seeds):
    teacher = PathwayNet1L(mask, use_relu=False).to(device)
    teacher.eval()
    with torch.no_grad():
        tlog_tr, _ = teacher(Xtr.to(device))
        tlog_te, _ = teacher(Xte.to(device))
        yte_hard = (tlog_te > 0).long()

        # --- Teacher performance reference ---
        teacher_acc = 1.0
        teacher_kd_mse = 0.0

    rows = []
    for s in seeds[:n_students]:
        set_all_seeds(s)
        student = PathwayNet1L(mask, use_relu=False).to(device)
        opt = torch.optim.Adam(student.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
        ds = DataLoader(TensorDataset(Xtr.to(device), tlog_tr), batch_size=cfg.batch, shuffle=True)
        student.train()
        for _ in range(cfg.epochs):
            for xb, tlog in ds:
                slog, _ = student(xb)
                loss = F.mse_loss(slog, tlog)
                opt.zero_grad(); loss.backward(); opt.step()
        student.eval()
        with torch.no_grad():
            slog_te, zS = student(Xte.to(device))
            _, zT = teacher(Xte.to(device))

            kd_mse = F.mse_loss(slog_te, tlog_te).item()
            acc = ((slog_te > 0).long() == yte_hard).float().mean().item()

            act_metrics = pathway_activation_alignment(zT, zS)

        
        class Dummy: pass
        T, S = Dummy(), Dummy()
        T.pathway = teacher.pathway; S.pathway = student.pathway
        rel, cs, cp = pathway_weight_scores_both(T, S, ptg, lambda_b=1.0)

        rows.append(dict(
            student_seed=s,

            # --- Teacher reference metrics ---
            teacher_kd_mse_logit=float(teacher_kd_mse),
            teacher_acc_vs_teacher=float(teacher_acc),

            # --- Student metrics ---
            kd_mse_logit=float(kd_mse),
            acc_vs_teacher=float(acc),

            # --- Weight recovery (student vs teacher) ---
            relL2_mean=float(rel.mean()),
            cos_std_mean=float(cs.mean()),
            cos_pos_mean=float(cp.mean()),
            
            act_corr_mean=act_metrics["act_corr_mean"],
            act_cosine_mean=act_metrics["act_cosine_mean"],
            act_r2=act_metrics["act_r2"],
        ))
    return pd.DataFrame(rows)

def run_regression(mask, ptg, Xtr, Xte, device, cfg: Config, n_students: int, seeds):
    teacher = RegressionTeacher(mask, hidden=cfg.hidden, p_drop=cfg.drop_teacher).to(device)
    teacher.eval()
    with torch.no_grad():
        y_tr_raw, _ = teacher(Xtr.to(device))
        y_te_raw, _ = teacher(Xte.to(device))

    mu, sd = y_tr_raw.mean(), y_tr_raw.std()
    y_tr_amp = cfg.gamma_y * (y_tr_raw - mu) / (sd + 1e-8)
    y_te_amp = cfg.gamma_y * (y_te_raw - mu) / (sd + 1e-8)

    # --- Teacher performance reference (teacher vs teacher amplified target) ---
    
    teacher_mse_amp = 0.0
    teacher_r2_amp = 1.0

    rows = []
    for s in seeds[:n_students]:
        set_all_seeds(s)
        student = RegressionStudent(mask, hidden=cfg.hidden, p_drop=cfg.drop_student).to(device)
        opt = torch.optim.Adam(student.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
        ds = DataLoader(TensorDataset(Xtr.to(device), y_tr_amp.detach()), batch_size=cfg.batch, shuffle=True)
        student.train()
        for _ in range(cfg.epochs):
            for xb, yb in ds:
                yp, _ = student(xb)
                loss = F.mse_loss(yp, yb)
                opt.zero_grad(); loss.backward(); opt.step()
        student.eval()
        with torch.no_grad():
            y_hat, zS = student(Xte.to(device))
            _, zT = teacher(Xte.to(device))

            mse = F.mse_loss(y_hat, y_te_amp).item()
            ss_res = torch.sum((y_te_amp - y_hat)**2)
            ss_tot = torch.sum((y_te_amp - y_te_amp.mean())**2)
            r2 = 1 - ss_res / ss_tot

            act_metrics = pathway_activation_alignment(zT, zS)

        rel, cs, cp = pathway_weight_scores_both(teacher.backbone, student.backbone, ptg, lambda_b=1.0)
        rows.append(dict(
            student_seed=s,

            # --- Teacher reference metrics ---
            teacher_mse_amp=float(teacher_mse_amp),
            teacher_r2_amp=float(teacher_r2_amp),

            # --- Student metrics ---
            mse_amp=float(mse),
            r2_amp=float(r2.item()),

            # --- Weight recovery (student vs teacher) ---
            relL2_mean=float(rel.mean()),
            cos_std_mean=float(cs.mean()),
            cos_pos_mean=float(cp.mean()),
            
            # --- Activation Nodes Recovery
            act_corr_mean=act_metrics["act_corr_mean"],
            act_cosine_mean=act_metrics["act_cosine_mean"],
            act_r2=act_metrics["act_r2"],
        ))
    return pd.DataFrame(rows)

def run_survival(mask, ptg, Xtr, Xte, device, cfg: Config, n_students: int, seeds):
    teacher = SurvivalTeacher(mask, hidden=cfg.hidden, p_drop=cfg.drop_teacher).to(device)
    teacher.eval()
    with torch.no_grad():
        r_tr_raw, _ = teacher(Xtr.to(device))
        r_te_raw, _ = teacher(Xte.to(device))

    mu, sd = r_tr_raw.mean(), r_tr_raw.std()
    r_tr_amp = cfg.gamma_risk * (r_tr_raw - mu) / (sd + 1e-8)
    r_te_amp = cfg.gamma_risk * (r_te_raw - mu) / (sd + 1e-8)

    t_te, e_te = simulate_times_from_risk(r_te_amp, censor_rate=cfg.censor_rate)

    # --- Teacher performance reference ---
    # KD MSE (teacher vs teacher amplified risk) = 0
    teacher_kd_mse = 0.0
    # C-index del teacher calcolato sulle stesse (time,event) simulate dal suo rischio amplificato
    teacher_cidx = concordance_index(r_te_amp, t_te.to(device), e_te.to(device))

    rows = []
    for s in seeds[:n_students]:
        set_all_seeds(s)
        student = SurvivalStudent(mask, hidden=cfg.hidden, p_drop=cfg.drop_student).to(device)
        opt = torch.optim.Adam(student.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
        ds = DataLoader(TensorDataset(Xtr.to(device), r_tr_amp.detach()), batch_size=cfg.batch, shuffle=True)
        student.train()
        for _ in range(cfg.epochs):
            for xb, rb in ds:
                rs, _ = student(xb)
                loss = F.mse_loss(rs, rb)
                opt.zero_grad(); loss.backward(); opt.step()
        student.eval()
        with torch.no_grad():
            rs_te, zS = student(Xte.to(device))
            _, zT = teacher(Xte.to(device))

            kd_mse = F.mse_loss(rs_te, r_te_amp).item()
            cidx = concordance_index(rs_te, t_te.to(device), e_te.to(device))

            act_metrics = pathway_activation_alignment(zT, zS)

        rel, cs, cp = pathway_weight_scores_both(teacher.backbone, student.backbone, ptg, lambda_b=1.0)
        rows.append(dict(
            student_seed=s,

            # --- Teacher reference metrics ---
            teacher_kd_mse_risk_amp=float(teacher_kd_mse),
            teacher_cindex=float(teacher_cidx),

            # --- Student metrics ---
            kd_mse_risk_amp=float(kd_mse),
            cindex=float(cidx),

            # --- Weight recovery (student vs teacher) ---
            relL2_mean=float(rel.mean()),
            cos_std_mean=float(cs.mean()),
            cos_pos_mean=float(cp.mean()),
            
            # --- Activation recovery ---
            act_corr_mean=act_metrics["act_corr_mean"],
            act_cosine_mean=act_metrics["act_cosine_mean"],
            act_r2=act_metrics["act_r2"],
        ))
    return pd.DataFrame(rows)


# ---------------------- main ----------------------
def main(args):
    device = get_device(args.device)
    os.makedirs(args.outdir, exist_ok=True)

    cfg = Config(
        G=args.G, P=args.P, size_min=args.size_min, size_max=args.size_max, overlap=args.overlap,
        n_train=args.n_train, n_test=args.n_test, hidden=args.hidden,
        drop_teacher=args.drop_teacher, drop_student=args.drop_student,
        batch=args.batch, epochs=args.epochs, lr=args.lr, wd=args.wd,
        temp=args.temp, gamma_risk=args.gamma_risk, gamma_y=args.gamma_y,
        censor_rate=args.censor_rate, teacher_seed=args.teacher_seed, data_seed=args.data_seed
    )

   
    set_all_seeds(cfg.teacher_seed)
    mask, ptg = build_sparse_overlapping_mask(
        G=cfg.G, P=cfg.P, size_range=(cfg.size_min, cfg.size_max), overlap_frac=cfg.overlap, seed=cfg.teacher_seed
    )
    X_all = synth_X_with_latents(
        n=cfg.n_train + cfg.n_test, G=cfg.G, ptg=ptg, beta=0.7, noise=0.6, seed=cfg.data_seed
    )
    Xtr = X_all[:cfg.n_train].to(device)
    Xte = X_all[cfg.n_train:].to(device)

    seeds = [args.student_seed_base + i for i in range(100)]

    if args.task in ("multiclass", "all"):
        print(">> MULTICLASS")
        df = run_multiclass(mask, ptg, Xtr, Xte, device, cfg, args.students, seeds, K=args.K)
        out = os.path.join(args.outdir, "metrics_multiclass.csv")
        df.to_csv(out, index=False); print("saved:", out)

    if args.task in ("binary", "all"):
        print(">> BINARY (nonlinear)")
        df = run_binary(mask, ptg, Xtr, Xte, device, cfg, args.students, seeds)
        out = os.path.join(args.outdir, "metrics_binary.csv")
        df.to_csv(out, index=False); print("saved:", out)

    if args.task in ("binary1l", "all"):
        print(">> BINARY 1-LAYER (linear)")
        df = run_binary_1layer(mask, ptg, Xtr, Xte, device, cfg, args.students, seeds)
        out = os.path.join(args.outdir, "metrics_binary1layer.csv")
        df.to_csv(out, index=False); print("saved:", out)

    if args.task in ("regression", "all"):
        print(">> REGRESSION")
        df = run_regression(mask, ptg, Xtr, Xte, device, cfg, args.students, seeds)
        out = os.path.join(args.outdir, "metrics_regression.csv")
        df.to_csv(out, index=False); print("saved:", out)

    if args.task in ("survival", "all"):
        print(">> SURVIVAL")
        df = run_survival(mask, ptg, Xtr, Xte, device, cfg, args.students, seeds)
        out = os.path.join(args.outdir, "metrics_survival.csv")
        df.to_csv(out, index=False); print("saved:", out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pathway-informed Teacher–Student tasks")
    parser.add_argument("--task", type=str, default="all",
                        choices=["all", "multiclass", "binary", "binary1l", "regression", "survival"])
    parser.add_argument("--students", type=int, default=20)
    parser.add_argument("--outdir", type=str, default="results_new")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    # data/model
    parser.add_argument("--G", type=int, default=400)
    parser.add_argument("--P", type=int, default=60)
    parser.add_argument("--size_min", type=int, default=14)
    parser.add_argument("--size_max", type=int, default=22)
    parser.add_argument("--overlap", type=float, default=0.55)
    parser.add_argument("--n_train", type=int, default=12000)
    parser.add_argument("--n_test", type=int, default=3000)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--drop_teacher", type=float, default=0.15)
    parser.add_argument("--drop_student", type=float, default=0.0)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--wd", type=float, default=0.0)
    # task specifics
    parser.add_argument("--temp", type=float, default=2.0)         # multiclass KD temperature
    parser.add_argument("--K", type=int, default=5)                # classes for multiclass
    parser.add_argument("--gamma_risk", type=float, default=6.0)   # survival amplify
    parser.add_argument("--gamma_y", type=float, default=6.0)      # regression amplify
    parser.add_argument("--censor_rate", type=float, default=0.05) # survival censoring
    # seeds
    parser.add_argument("--teacher_seed", type=int, default=42)
    parser.add_argument("--data_seed", type=int, default=123)
    parser.add_argument("--student_seed_base", type=int, default=9000)
    args = parser.parse_args()
    main(args)

