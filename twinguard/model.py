"""GNN digital twin (graph attention + GRU) and the per-link GRU baseline, as in the experiments notebook.

The notebook defines the models with module-level globals (L, W, A_T, NH, NM); here the same layers take the
line-graph adjacency and the output sizes as arguments, so that the models can be used on any topology.
State dicts are identical, so checkpoints of the notebook load unchanged.
"""
import copy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fnn
from scipy.stats import rankdata

QS = np.array([0.05, 0.5, 0.95], np.float32)


class GATLayer(nn.Module):
    # Dense multi-head graph attention on a fixed line graph, with residual connection and LayerNorm.
    def __init__(self, d_in, d, heads, dropout, adj):
        super().__init__()
        self.h, self.d = heads, d
        self.register_buffer("adj", adj, persistent=False)
        self.lin = nn.Linear(d_in, d * heads, bias=False)
        self.a_src = nn.Parameter(torch.randn(heads, d) * 0.1)
        self.a_dst = nn.Parameter(torch.randn(heads, d) * 0.1)
        self.res = nn.Identity() if d_in == d * heads else nn.Linear(d_in, d * heads, bias=False)
        self.norm = nn.LayerNorm(d * heads)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):                                   # x [B, L, d_in]
        B, L = x.shape[:2]
        h = self.lin(x).view(B, L, self.h, self.d)
        e = (h * self.a_dst).sum(-1).unsqueeze(2) + (h * self.a_src).sum(-1).unsqueeze(1)   # [B, i, j, heads]
        e = Fnn.leaky_relu(e, 0.2).masked_fill(self.adj.view(1, L, L, 1) == 0, float("-inf"))
        out = torch.einsum("bijh,bjhd->bihd", self.drop(e.softmax(2)), h).reshape(B, L, -1)
        return self.norm(Fnn.elu(out) + self.res(x))


class Heads(nn.Module):
    # Non-crossing quantiles (median + softplus offsets) and a violation logit, per link and horizon.
    def __init__(self, d, dropout, n_h, n_m):
        super().__init__()
        self.n_h, self.n_m = n_h, n_m
        self.drop = nn.Dropout(dropout)
        self.q = nn.Linear(d, n_h * n_m * 3)
        self.v = nn.Linear(d, n_h)

    def forward(self, z):                                   # z [B, L, d]
        B, L = z.shape[:2]
        z = self.drop(z)
        r = self.q(z).view(B, L, self.n_h, self.n_m, 3)
        med = r[..., 1]
        q = torch.stack([med - Fnn.softplus(r[..., 0]), med, med + Fnn.softplus(r[..., 2])], -1)
        return q, self.v(z)                                 # [B, L, H, M, Q], [B, L, H]


class _Base(nn.Module):
    # Per-step spatial encoder -> GRU over the window -> heads. Subclasses define the encoder.
    def __init__(self, f_in, hp, adj, n_h=3, n_m=3):
        super().__init__()
        adj = torch.as_tensor(np.asarray(adj, np.float32))
        self.L = adj.shape[0]
        self.emb = nn.Embedding(self.L, 8)
        d = hp["hidden"] * hp["heads"]
        self.encoder = self.make_encoder(f_in + 8, hp, adj)
        self.gru = nn.GRU(d, hp["gru"], num_layers=hp["gru_layers"], batch_first=True,
                          dropout=hp["dropout"] if hp["gru_layers"] > 1 else 0.0)
        self.heads = Heads(hp["gru"] + 8, hp["dropout"], n_h, n_m)

    def forward(self, x):                                   # x [B, W, L, F]
        B, W, L = x.shape[:3]
        e = self.emb.weight
        z = torch.cat([x, e.expand(B, W, L, -1)], -1).reshape(B * W, L, -1)
        z = self.encoder(z).view(B, W, L, -1).permute(0, 2, 1, 3).reshape(B * L, W, -1)
        _, hN = self.gru(z)
        return self.heads(torch.cat([hN[-1].view(B, L, -1), e.expand(B, L, -1)], -1))


class GNNTwin(_Base):
    # `layers` graph-attention layers per time step (message passing between links that share a router).
    def make_encoder(self, d_in, hp, adj):
        d = hp["hidden"] * hp["heads"]
        return nn.Sequential(*[GATLayer(d_in if i == 0 else d, hp["hidden"], hp["heads"], hp["dropout"], adj)
                               for i in range(hp["layers"])])


class _MLPBlock(nn.Module):
    def __init__(self, d_in, d, dropout):
        super().__init__()
        self.lin = nn.Linear(d_in, d); self.norm = nn.LayerNorm(d); self.drop = nn.Dropout(dropout)
        self.res = nn.Identity() if d_in == d else nn.Linear(d_in, d, bias=False)

    def forward(self, x):
        return self.norm(Fnn.elu(self.drop(self.lin(x))) + self.res(x))


class LinkGRU(_Base):
    # Baseline: identical skeleton, but the per-step encoder never mixes information between links.
    def make_encoder(self, d_in, hp, adj):
        d = hp["hidden"] * hp["heads"]
        return nn.Sequential(*[_MLPBlock(d_in if i == 0 else d, d, hp["dropout"]) for i in range(hp["layers"])])


MODELS = {"GNNTwin": GNNTwin, "LinkGRU": LinkGRU}


def loss_fn(q, logit, y, ym, v):
    # Pinball loss on log1p targets + BCE on the violation flag; only observed targets count.
    qs = torch.as_tensor(QS, device=q.device)
    yt = torch.log1p(torch.nan_to_num(y).clamp(min=0))
    u = yt.unsqueeze(-1) - q
    pin = torch.maximum(qs * u, (qs - 1) * u)                         # [B, L, H, M, Q]
    m = ym.float()
    pin = (pin * m[..., None, None]).sum() / (m.sum() * q.shape[-2] * len(QS) + 1e-6)
    vm = ~torch.isnan(v)
    bce = Fnn.binary_cross_entropy_with_logits(logit[vm], v[vm]) if vm.any() else logit.sum() * 0
    return pin + bce, pin.detach(), bce.detach()


def auroc(y, s):
    y = np.asarray(y); s = np.asarray(s)
    pos = y == 1; n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return (r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def _batches(d, bs, gen=None):
    n = len(d["X"])
    order = torch.randperm(n, generator=gen) if gen is not None else torch.arange(n)
    for i in range(0, n, bs):
        j = order[i:i + bs]
        yield d["X"][j], d["Y"][j], d["Ym"][j], d["V"][j]


def _tensors(d):
    return {k: torch.as_tensor(d[k]) for k in ("X", "Y", "Ym", "V")}


@torch.no_grad()
def val_loss(model, d, bs=1024):
    model.eval(); tot, n = 0.0, 0
    for x, y, ym, v in _batches(d, bs):
        q, lg = model(x); l, _, _ = loss_fn(q, lg, y, ym, v)
        tot += l.item() * len(x); n += len(x)
    return tot / n


def train(cls, hp, fixed, fit, val, adj, seed, init_state=None, log=None):
    """Train (or fine-tune, from init_state) with AdamW, LR halving on plateau, and early stopping on val loss.
    fit/val: dicts of numpy arrays X [N,W,L,F], Y [N,L,H,M], Ym [N,L,H], V [N,L,H]."""
    torch.manual_seed(seed); np.random.seed(seed); gen = torch.Generator().manual_seed(seed)
    model = cls(fit["X"].shape[-1], hp, adj)
    if init_state is not None:
        model.load_state_dict(init_state)
    fit_t, val_t = _tensors(fit), _tensors(val)
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=fixed["lr_factor"], patience=fixed["lr_patience"])
    best, best_state, best_ep, wait, hist, t0 = np.inf, copy.deepcopy(model.state_dict()), 0, 0, [], time.time()
    if init_state is not None:
        best = val_loss(model, val_t)                    # fine-tuning keeps the initial model unless it improves
    for ep in range(fixed["max_epochs"]):
        model.train(); tl, nb = 0.0, 0
        for x, y, ym, v in _batches(fit_t, hp["batch"], gen):
            q, lg = model(x); l, _, _ = loss_fn(q, lg, y, ym, v)
            opt.zero_grad(set_to_none=True); l.backward()
            nn.utils.clip_grad_norm_(model.parameters(), fixed["clip"]); opt.step()
            tl += l.item(); nb += 1
        vl = val_loss(model, val_t); hist.append((tl / nb, vl)); sched.step(vl)
        if vl < best - 1e-4:
            best, best_ep, wait = vl, ep + 1, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= fixed["patience"]:
                break
        if log:
            log(f"    epoch {ep + 1}: train {tl / nb:.4f}  val {vl:.4f}")
    model.load_state_dict(best_state)
    return model, {"best_val_loss": float(best), "best_epoch": best_ep, "epochs_run": len(hist), "hist": hist,
                   "seconds": time.time() - t0}


@torch.no_grad()
def predict(models, X, bs=1024):
    """Ensemble prediction: returns Q [S,N,L,H,M,3] (log1p scale) and P [S,N,L,H] violation probabilities."""
    Xt = torch.as_tensor(X); Qs, Ps = [], []
    for m in models:
        m.eval(); Q, P = [], []
        for i in range(0, len(Xt), bs):
            q, lg = m(Xt[i:i + bs]); Q.append(q.numpy()); P.append(torch.sigmoid(lg).numpy())
        Qs.append(np.concatenate(Q)); Ps.append(np.concatenate(P))
    return np.stack(Qs), np.stack(Ps)
