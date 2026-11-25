"""NOGI-style graph-conditioned vector fields with switches.

W is [source, target]: W[j, i] weights the message j -> i.

Variants
  orig          non-additive update MLP  upd([x_i, m_i]),        W >= 0   (NOGI as published)
  orig_z        upd([x_i, z_i, m_i]),                            W >= 0   (ablation: z gates messages)
  add           F(x_i) + sum_j W_ji G(x_i, x_j),                 W >= 0
  add_z         F(x_i, z_i) + sum_j W_ji G(x_i, x_j),            W >= 0
  add_signed    F(x_i) + sum_j W_ji G(x_i, x_j),                 W real
  add_z_signed  F(x_i, z_i) + sum_j W_ji G(x_i, x_j),            W real
  add_(z_)pm    as add_(z_)signed with W = W+ - W-, W+, W- >= 0 (clamped, L1 on both)
"""
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint

VARIANTS: Dict[str, dict] = {
    "orig": {"additive": False, "use_z": False, "mode": "nonneg"},
    "orig_z": {"additive": False, "use_z": True, "mode": "nonneg"},
    "add": {"additive": True, "use_z": False, "mode": "nonneg"},
    "add_z": {"additive": True, "use_z": True, "mode": "nonneg"},
    "add_signed": {"additive": True, "use_z": False, "mode": "signed"},
    "add_z_signed": {"additive": True, "use_z": True, "mode": "signed"},
    "add_pm": {"additive": True, "use_z": False, "mode": "pm"},
    "add_z_pm": {"additive": True, "use_z": True, "mode": "pm"},
    # the shared-function (NOGI-style) model with the non-negativity of W removed: baseline "signed NOGI"
    "orig_signed": {"additive": False, "use_z": False, "mode": "signed"},
}


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )


class GraphParam(nn.Module):
    def __init__(self, N: int, mode: str, init_scale: float = 1.0, mask_diag: bool = True):
        super().__init__()
        if mode not in ("nonneg", "signed", "pm"):
            raise ValueError(mode)
        self.N, self.mode, self.mask_diag = N, mode, mask_diag
        # Same U(0,1) start for both modes: the only
        # difference between nonneg and signed is the projection.
        self.raw = nn.Parameter(torch.rand(N, N) * init_scale)
        # 'pm': signed W as a difference of two clamped non-negative parts, so that the
        # clamp produces exact zeros as in 'nonneg' (an unclamped signed W never reaches 0).
        self.raw_neg = nn.Parameter(torch.rand(N, N) * init_scale) if mode == "pm" else None
        self.register_buffer("mask", 1.0 - torch.eye(N))
        # sign warm-up: while True, a 'signed' W is projected like 'nonneg' (set by train.py --sign_warmup)
        self.clamp_signed = False

    def _m(self, t: torch.Tensor) -> torch.Tensor:
        return t * self.mask if self.mask_diag else t

    def W(self) -> torch.Tensor:
        if self.mode == "pm":
            return self._m(self.raw - self.raw_neg)
        return self._m(self.raw)

    def set_support(self, B) -> None:
        """Fix the edge support (prune-and-refit): W is exactly 0 outside B (diagonal always excluded) and
        those entries get no gradient. Stored in the `mask` buffer, so saved states carry the support."""
        with torch.no_grad():
            B = torch.as_tensor(B, dtype=self.mask.dtype, device=self.mask.device)
            self.mask.copy_(B * (1.0 - torch.eye(self.N, device=self.mask.device)))
        self.mask_diag = True

    def l1(self) -> torch.Tensor:
        if self.mode == "pm":
            return self._m(self.raw).abs().sum() + self._m(self.raw_neg).abs().sum()
        return self.W().abs().sum()

    def prox(self, tau: float) -> None:
        """Soft-threshold step of the L1 proximal operator with an absolute threshold tau.
        Unlike a tiny L1 gradient under Adam, it sets small weights exactly to zero."""
        if tau <= 0:
            return
        with torch.no_grad():
            if self.mode == "signed":
                self.raw.copy_(torch.sign(self.raw) * torch.clamp(self.raw.abs() - tau, min=0.0))
            else:
                self.raw.copy_(torch.clamp(self.raw - tau, min=0.0))
                if self.raw_neg is not None:
                    self.raw_neg.copy_(torch.clamp(self.raw_neg - tau, min=0.0))

    def project(self) -> None:
        if self.mode in ("nonneg", "pm") or (self.mode == "signed" and self.clamp_signed):
            with torch.no_grad():
                self.raw.clamp_(min=0.0)
                if self.raw_neg is not None:
                    self.raw_neg.clamp_(min=0.0)


class VectorField(nn.Module):
    """feat='raw' feeds states directly to the MLPs (NOGI); feat='trig' feeds (cos x, sin x),
    which is required for phase variables that are not wrapped (unbounded theta)."""

    def __init__(self, node_dim: int, hidden: int, additive: bool, use_z: bool, d_z: int, N: int,
                 feat: str = "raw", coupling: str = "free", z_norm: str = "none", edge_z: bool = False,
                 edge_emb: int = 0, edge_scale: str = "batch"):
        super().__init__()
        if edge_scale not in ("batch", "epoch"):
            raise ValueError("unknown edge_scale: %s" % edge_scale)
        # edge_emb only. 'batch': per-edge RMS of the current batch in training, a running mean in evaluation (the
        # derivative of one trajectory then depends on the other trajectories of its batch). 'epoch': the RMS stored
        # by refresh_edge_scale() on fixed reference states, used unchanged in training and evaluation until the
        # next refresh, so the vector field of a trajectory depends on that trajectory only.
        self.edge_scale_mode = edge_scale
        if edge_z and not (additive and use_z):
            raise ValueError("edge_z needs an additive variant with z")
        if edge_emb and not additive:
            raise ValueError("edge_emb needs an additive variant")
        if edge_emb and edge_z:
            raise ValueError("edge_emb and edge_z are alternatives")
        # 'edge_emb' (design option B): G(x_i, x_j, b_i, a_j) with an outgoing embedding a_j and an incoming
        # embedding b_i that are separate from the self-dynamics latent z; every edge's coupling function is scaled to
        # unit RMS over the data (batch statistics in training, a running estimate in evaluation, as in batch norm),
        # which fixes the (W, G) scale gauge so that |W_ji| alone carries the strength and stays comparable.
        self.edge_emb = int(edge_emb)
        # 'edge_z' feeds G the latents of both endpoints, G(x_i, x_j, z_i, z_j): the coupling
        # form may depend on the types of source and target (edge heterogeneity by endpoint type).
        self.edge_z = edge_z
        if z_norm not in ("none", "standardize"):
            raise ValueError("unknown z_norm: %s" % z_norm)
        if feat not in ("raw", "trig"):
            raise ValueError("unknown feat: %s" % feat)
        if coupling not in ("free", "diffusive"):
            raise ValueError("unknown coupling: %s" % coupling)
        if coupling == "diffusive" and not additive:
            raise ValueError("coupling='diffusive' needs an additive variant")
        # 'diffusive' uses G(x_i, x_j) - G(x_i, x_i): the coupling cannot carry a per-node
        # constant, so node offsets (e.g. omega_i) cannot be absorbed into column sums of W.
        self.coupling = coupling
        # 'standardize' feeds F the per-dimension standardised z (zero mean, unit variance over nodes):
        # F can absorb any affine map of z, so this fixes the scale/shift gauge without a penalty on Z.
        self.z_norm = z_norm
        self.node_dim, self.additive, self.use_z, self.d_z, self.N, self.feat = node_dim, additive, use_z, d_z, N, feat
        self.fdim = node_dim if feat == "raw" else 2 * node_dim
        zin = d_z if use_z else 0
        if use_z:
            self.Z = nn.Parameter(torch.randn(N, d_z) * 0.1)
        else:
            self.Z = None
        if additive:
            self.F = _mlp(self.fdim + zin, hidden, node_dim)
            self.G = _mlp(2 * self.fdim + (2 * d_z if edge_z else 0) + 2 * self.edge_emb, hidden, node_dim)
            if self.edge_emb:
                self.Ea = nn.Parameter(torch.randn(N, self.edge_emb) * 0.1)     # outgoing (source) embedding
                self.Eb = nn.Parameter(torch.randn(N, self.edge_emb) * 0.1)     # incoming (target) embedding
                self.register_buffer("edge_scale", torch.ones(N, N))           # running RMS per edge [i, j]
        else:
            self.msg = _mlp(2 * self.fdim, hidden, node_dim)
            self.upd = _mlp(self.fdim + zin + node_dim, hidden, node_dim)

    def phi(self, x: torch.Tensor) -> torch.Tensor:
        if self.feat == "trig":
            return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
        return x

    def z_eff(self) -> torch.Tensor:
        """The z that F (or upd) actually sees."""
        if self.z_norm == "standardize":
            return (self.Z - self.Z.mean(0)) / (self.Z.std(0, unbiased=False) + 1e-6)
        return self.Z

    def _z(self, B: int) -> torch.Tensor:
        return self.z_eff().unsqueeze(0).expand(B, self.N, self.d_z)

    def edge_coupling(self, xi: torch.Tensor, xj: torch.Tensor, i: int, j: int) -> torch.Tensor:
        """Coupling of the single edge j -> i per unit W_ji, at receiver states xi and sender states xj (P, node_dim):
        the same computation as forward(). Read-outs call this instead of re-implementing the coupling."""
        hi, hj = self.phi(xi), self.phi(xj); P = len(hi)
        ext = []
        if self.edge_z:
            Z = self.z_eff(); ext = [Z[i].expand(P, -1), Z[j].expand(P, -1)]
        elif self.edge_emb:
            ext = [self.Eb[i].expand(P, -1), self.Ea[j].expand(P, -1)]
        g = self.G(torch.cat([hi, hj] + ext, -1))
        if self.coupling == "diffusive":
            g = g - self.G(torch.cat([hi, hi] + ext, -1))
        if self.edge_emb:
            sc = self.edge_scale
            g = g / (sc[i, j] + 1e-3 * sc.mean() + 1e-12)
        return g

    def pair_coupling(self, x: torch.Tensor) -> torch.Tensor:
        """Coupling per unit weight of every ordered pair at the states x (B, N, d), before any normalisation:
        (B, i, j, d) with receiver i and sender j, computed as in forward(). Shared and edge-embedding couplings."""
        if not self.additive or self.edge_z:
            raise ValueError("pair_coupling needs an additive model without edge_z")
        B, N, _ = x.shape
        h = self.phi(x)
        hi = h.unsqueeze(2).expand(B, N, N, self.fdim); hj = h.unsqueeze(1).expand(B, N, N, self.fdim)
        ext = []
        if self.edge_emb:
            ext = [self.Eb.view(1, N, 1, -1).expand(B, N, N, self.edge_emb), self.Ea.view(1, 1, N, -1).expand(B, N, N, self.edge_emb)]
        g = self.G(torch.cat([hi, hj] + ext, dim=-1))
        if self.coupling == "diffusive":
            g = g - self.G(torch.cat([hi, hi] + ext, dim=-1))
        return g

    @torch.no_grad()
    def coupling_rms(self, x: torch.Tensor) -> torch.Tensor:
        """Root mean square of the shared coupling over the pairs i != j, the states x and the state coordinates.
        With rescale_coupling() it fixes the scale that (W, G) -> (a W, G / a) leaves free."""
        if self.edge_emb:
            raise ValueError("the edge-embedding coupling is normalised per edge (edge_scale), not by coupling_rms")
        g = self.pair_coupling(x)
        off = 1.0 - torch.eye(self.N, device=g.device, dtype=g.dtype)
        return ((g.pow(2) * off.view(1, self.N, self.N, 1)).sum() / (g.shape[0] * self.N * (self.N - 1) * g.shape[3])).sqrt()

    @torch.no_grad()
    def rescale_coupling(self, r: float) -> None:
        """Divide the output of G by r (its last layer is linear); multiply W by r to keep the model unchanged."""
        last = self.G[-1]
        last.weight.div_(r); last.bias.div_(r)

    @torch.no_grad()
    def refresh_edge_scale(self, x: torch.Tensor) -> None:
        """edge_scale 'epoch': store the RMS of every edge's coupling over the reference states x."""
        self.edge_scale.copy_(self.pair_coupling(x).pow(2).mean(dim=(0, 3)).add(1e-12).sqrt())

    def forward(self, x: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        B, N, d = x.shape
        h = self.phi(x)                                   # (B, N, fdim)
        hi = h.unsqueeze(2).expand(B, N, N, self.fdim)    # receiver i
        hj = h.unsqueeze(1).expand(B, N, N, self.fdim)    # sender j
        pair = torch.cat([hi, hj], dim=-1)                # (B, i, j, 2 fdim)
        Wt = W.transpose(0, 1).unsqueeze(0).unsqueeze(-1)  # (1, i, j, 1): W[j, i]
        if self.additive:
            if self.edge_z:
                Z = self._z(B)                                        # (B, N, d_z)
                zz = torch.cat([Z.unsqueeze(2).expand(B, N, N, self.d_z),
                                Z.unsqueeze(1).expand(B, N, N, self.d_z)], dim=-1)  # (z_i, z_j)
                pair = torch.cat([pair, zz], dim=-1)
            if self.edge_emb:
                ee = torch.cat([self.Eb.view(1, N, 1, -1).expand(B, N, N, self.edge_emb),
                                self.Ea.view(1, 1, N, -1).expand(B, N, N, self.edge_emb)], dim=-1)   # (b_i, a_j)
                g = self.G(torch.cat([pair, ee], dim=-1))
                if self.coupling == "diffusive":
                    g = g - self.G(torch.cat([hi, hi, ee], dim=-1))
                if self.training and self.edge_scale_mode == "batch":
                    rms = g.pow(2).mean(dim=(0, 3)).add(1e-12).sqrt()                          # (i, j)
                    with torch.no_grad():
                        self.edge_scale.mul_(0.9).add_(0.1 * rms.detach())
                else:
                    rms = self.edge_scale
                rms = rms + 1e-3 * rms.mean().detach() + 1e-12
                return self.F(torch.cat([h, self._z(B)], dim=-1) if self.use_z else h) + (g / rms.unsqueeze(0).unsqueeze(-1) * Wt).sum(dim=2)
            inter = (self.G(pair) * Wt).sum(dim=2)
            if self.coupling == "diffusive" and self.edge_z:
                # G(x_i, x_i, z_i, z_j) depends on the pair, so subtract per edge
                pair_self = torch.cat([hi, hi, zz], dim=-1)
                inter = inter - (self.G(pair_self) * Wt).sum(dim=2)
            elif self.coupling == "diffusive":
                g_self = self.G(torch.cat([h, h], dim=-1))          # (B, N, d): G(x_i, x_i)
                inter = inter - g_self * W.sum(dim=0).view(1, N, 1)  # sum_j W[j, i]
            fin = torch.cat([h, self._z(B)], dim=-1) if self.use_z else h
            return self.F(fin) + inter
        m = (self.msg(pair) * Wt).sum(dim=2)
        parts = [h] + ([self._z(B)] if self.use_z else []) + [m]
        return self.upd(torch.cat(parts, dim=-1))


ADAPTIVE_METHODS = {"dopri5", "dopri8", "bosh3", "fehlberg2", "adaptive_heun"}
MAX_NUM_STEPS = 2000


class GraphNeuralODE(nn.Module):
    def __init__(self, vf: VectorField, t_end: float, rtol: float, atol: float, method: str = "dopri5",
                 step_size=None):
        super().__init__()
        self.vf, self.rtol, self.atol, self.method = vf, rtol, atol, method
        # A stiff learned field must fail fast (caught by the trainer) instead of hanging a GPU slot.
        self.options = {"max_num_steps": MAX_NUM_STEPS} if method in ADAPTIVE_METHODS else {}
        if step_size is not None and method not in ADAPTIVE_METHODS:
            self.options["step_size"] = float(step_size)       # fixed-step solvers: sub-steps over [0, t_end]
        self.register_buffer("t", torch.tensor([0.0, float(t_end)]))

    def forward(self, x0: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        params = tuple(self.vf.parameters()) + ((W,) if W.requires_grad else ())
        func = lambda t, y: self.vf(y, W)  # noqa: E731
        sol = odeint(func, x0, self.t.to(x0.device), method=self.method,
                     rtol=self.rtol, atol=self.atol, adjoint_params=params,
                     options=(self.options or None))
        return sol[-1]


def _edge_gauge_signs(vf: VectorField, xs: torch.Tensor, max_states: int = 256, chunk: int = 16) -> np.ndarray:
    """Per-edge orientation for design option B. With G(x_i, x_j; b_i, a_j) the (W, G) -> (-W, -G) symmetry
    holds for every edge separately, so the shared-G functional is applied per edge: s_ji = sign of the mean over
    observed states x of dG/dx_j at x_j = x_i = x, with that edge's (b_i, a_j). Returned in W's [j, i] indexing, so
    W * s is the oriented signed graph. With a shared G every entry equals the global gauge. The positive per-edge
    RMS scale of option B does not change a sign and is not applied."""
    N, e = vf.N, vf.edge_emb
    if xs.shape[0] > max_states:
        idx = torch.randperm(xs.shape[0], generator=torch.Generator().manual_seed(0))[:max_states]
        xs = xs[idx.to(xs.device)]
    ee = torch.cat([vf.Eb.detach().view(N, 1, e).expand(N, N, e),
                    vf.Ea.detach().view(1, N, e).expand(N, N, e)], dim=-1)          # [i, j] -> (b_i, a_j)
    acc = torch.zeros(N, N, device=xs.device)
    for k in range(0, xs.shape[0], chunk):
        xc = xs[k:k + chunk]; P = xc.shape[0]
        hi = vf.phi(xc).view(P, 1, 1, -1).expand(P, N, N, vf.fdim)
        xj = xc.view(P, 1, 1, -1).expand(P, N, N, xc.shape[-1]).clone().requires_grad_(True)
        with torch.enable_grad():
            out = vf.G(torch.cat([hi, vf.phi(xj), ee.unsqueeze(0).expand(P, N, N, 2 * e)], dim=-1)).sum()
            grad, = torch.autograd.grad(out, xj)
        acc += grad.sum(dim=(0, 3))                                                    # [i, j]
    s = torch.where(acc >= 0, 1.0, -1.0).T                                             # -> [j, i]
    return s.cpu().numpy()


def gauge_sign(vf: VectorField, x: torch.Tensor, max_nodes: int = 4096):
    """Orientation of the (W, G) -> (-W, -G) gauge for additive fields.

    Prescribed functional (no ground truth): the mean Jacobian dG/dx_j evaluated at
    coincident states x_j = x_i = x over observed node states. Positive for attractive
    or diffusive couplings such as sin(x_j - x_i) or tanh(x_j) - tanh(x_i).
    Returns +1 for non-additive variants (no gauge freedom), and for design option B (edge_emb) a +-1 matrix
    in W's [j, i] indexing, one orientation per edge (_edge_gauge_signs).
    """
    if not vf.additive:
        return 1
    xs = x.detach().reshape(-1, x.shape[-1])
    if vf.edge_emb:
        return _edge_gauge_signs(vf, xs)
    if xs.shape[0] > max_nodes:
        idx = torch.randperm(xs.shape[0], generator=torch.Generator().manual_seed(0))[:max_nodes]
        xs = xs[idx.to(xs.device)]
    xj = xs.clone().requires_grad_(True)
    with torch.enable_grad():
        inp = [vf.phi(xs), vf.phi(xj)]
        if vf.edge_z:  # evaluate at the mean latent for both endpoints
            zm = vf.z_eff().mean(0, keepdim=True).expand(xs.shape[0], -1).detach()
            inp += [zm, zm]
        out = vf.G(torch.cat(inp, dim=-1)).sum()
        grad, = torch.autograd.grad(out, xj)
    m = float(grad.mean())
    return 1 if m >= 0 else -1


def build(variant: str, N: int, node_dim: int, hidden: int, d_z: int, t_end: float,
          rtol: float, atol: float, method: str, mask_diag: bool, init_scale: float = 1.0,
          feat: str = "raw", coupling: str = "free", z_norm: str = "none", edge_z: bool = False, step_size=None,
          edge_emb: int = 0, edge_scale: str = "batch"):
    cfg = VARIANTS[variant]
    gp = GraphParam(N, cfg["mode"], init_scale=init_scale, mask_diag=mask_diag)
    vf = VectorField(node_dim, hidden, cfg["additive"], cfg["use_z"], d_z, N, feat=feat,
                     coupling=coupling, z_norm=z_norm, edge_z=edge_z, edge_emb=edge_emb, edge_scale=edge_scale)
    ode = GraphNeuralODE(vf, t_end, rtol, atol, method, step_size=step_size)
    return gp, ode
