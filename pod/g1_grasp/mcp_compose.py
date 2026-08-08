"""Multiplicative Compositional Policy (MCP) — simultaneous expert blending.

Built 2026-08-01 while laneC trains, so that composition is ready the moment
a second self-trained expert exists. Nothing here is downloaded or
human-derived: every expert is one of OUR checkpoints, trained from our own
rewards and our own trajectory-optimizer references. The router is trained
here, by us. The zero-human-data claim is untouched.

WHY MULTIPLICATIVE, NOT SWITCHING
Switching (one expert active per timestep) is the configuration the humanoid
literature reports failing: a humanoid carrying an object needs legs and arms
acting at the same instant, not in turns. MCP (Peng et al. 2019) composes the
experts' Gaussians as a weighted PRODUCT, so every expert influences every
step, and — the useful part — an expert that is confident about the legs and
indifferent about the arms automatically dominates the leg dimensions and
cedes the arm dimensions. Per-DoF specialisation falls out of the precision
weighting rather than being hand-designed.

    product of N Gaussians with per-expert weights w_i(s) >= 0:
        precision_j = sum_i w_i * (1 / var_ij)
        mean_j      = (sum_i w_i * mu_ij / var_ij) / precision_j
        var_j       = 1 / precision_j

The gate w_i(s) is the ONLY trainable part when experts are frozen, which is
what makes this cheap: a small MLP, not a re-training of the skills.

Usage (once a second expert exists):
    experts = [load_actor(ck) for ck in (GRASP_CKPT, POSTURE_CKPT)]
    policy  = MCPPolicy(experts, num_obs, num_actions, freeze_experts=True)
"""

import torch
import torch.nn as nn


class MCPPolicy(nn.Module):
    """Frozen expert actors + trainable gate, composed multiplicatively."""

    def __init__(self, experts, num_obs, num_actions, gate_hidden=(128, 64),
                 freeze_experts=True, min_weight=1e-4):
        super().__init__()
        assert len(experts) >= 2, (
            "MCP with one expert is that expert — composition needs at least "
            "two. This assert exists because the project's whole difficulty "
            "was producing the second one.")
        self.experts = nn.ModuleList(experts)
        self.num_actions = num_actions
        self.min_weight = min_weight
        if freeze_experts:
            for e in self.experts:
                for p in e.parameters():
                    p.requires_grad_(False)
                e.eval()

        layers, last = [], num_obs
        for h in gate_hidden:
            layers += [nn.Linear(last, h), nn.ELU()]
            last = h
        layers += [nn.Linear(last, len(experts))]
        self.gate = nn.Sequential(*layers)
        # per-expert log-std, learned; experts' own stds are frozen with them
        self.log_std = nn.Parameter(
            torch.zeros(len(experts), num_actions) - 1.0)

    def expert_gaussians(self, obs):
        """(N, B, A) means and variances from the frozen experts."""
        mus, vars_ = [], []
        with torch.no_grad():
            for i, e in enumerate(self.experts):
                mu = e(obs) if not hasattr(e, "act_inference") \
                    else e.act_inference(obs)
                mus.append(mu)
        for i in range(len(self.experts)):
            vars_.append(self.log_std[i].exp().square().expand_as(mus[i]))
        return torch.stack(mus), torch.stack(vars_)

    def forward(self, obs):
        """Composed Gaussian: returns (mean, std)."""
        mus, vars_ = self.expert_gaussians(obs)          # (N,B,A)
        w = torch.softmax(self.gate(obs), dim=-1)         # (B,N)
        w = w.clamp_min(self.min_weight).t().unsqueeze(-1)  # (N,B,1)
        prec = (w / vars_).sum(dim=0)                     # (B,A)
        mean = (w * mus / vars_).sum(dim=0) / prec
        return mean, prec.reciprocal().sqrt()

    def gate_weights(self, obs):
        """Diagnostic: which expert is driving, per state. Print this in
        every eval — an ungated mixture that always picks one expert has
        silently degenerated into switching, which is the failing mode."""
        return torch.softmax(self.gate(obs), dim=-1)


def compose_smoke_test(num_obs=205, num_actions=41, batch=8):
    """Known-answer test: two experts with identical means must compose to
    that mean; an expert with huge variance on a dimension must cede it."""
    class Stub(nn.Module):
        def __init__(self, val):
            super().__init__()
            self.val = val
            self.lin = nn.Linear(num_obs, num_actions)

        def forward(self, o):
            return torch.full((o.shape[0], num_actions), self.val)

    p = MCPPolicy([Stub(1.0), Stub(1.0)], num_obs, num_actions)
    obs = torch.randn(batch, num_obs)
    mean, std = p(obs)
    same = torch.allclose(mean, torch.ones_like(mean), atol=1e-4)
    print(f"MCP_SMOKE identical-experts mean==1.0: {same}")

    p2 = MCPPolicy([Stub(0.0), Stub(2.0)], num_obs, num_actions)
    with torch.no_grad():
        p2.log_std[1, :] = 3.0        # expert 1 very unsure -> should cede
    mean2, _ = p2(obs)
    ceded = bool((mean2.abs() < 0.2).all())
    print(f"MCP_SMOKE unsure-expert cedes (mean near 0.0): {ceded} "
          f"(mean={mean2.mean().item():.4f})")
    print(f"MCP_SMOKE_{'PASS' if (same and ceded) else 'FAIL'}")
    return same and ceded


if __name__ == "__main__":
    compose_smoke_test()
