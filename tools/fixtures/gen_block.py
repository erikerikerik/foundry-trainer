#!/usr/bin/env python3
"""Dev-only reference fixture generator for the Phase 0.5 spike (item 0.5.2).

Allowed under CLAUDE.md rule 2: this is the one-time reference-fixture exception.
Nothing here ships or runs in CI; only the generated data plus PROVENANCE.md are
committed. See docs/TESTING.md ("Reference fixtures") and docs/SPEC.md section 22.

Generates a scaled-down FLUX.2-klein-style block pair with the real op set:
one double-stream (MMDiT) block and one single-stream block, each with adaLN
modulation, RMSNorm QK-norm, 4-axis RoPE, explicit softmax attention, SwiGLU
MLP, gated residuals, and LoRA on selected linears.

Reference semantics the Rust implementation must match:
  - matmuls: compute-dtype inputs, fp32 accumulation (hardware behavior),
  - attention softmax: computed in fp32, probabilities cast back to compute dtype,
  - LayerNorm/RMSNorm statistics: computed in fp32, result cast back,
  - RoPE cos/sin: computed in fp32, cast to compute dtype,
  - adaLN modulation: (1 + scale) * norm(x) + shift; gates multiply branches.

Outputs (written next to this script):
  block_fp32.safetensors, block_bf16.safetensors, meta.json
Each file holds inputs, weights, intermediate activations (including the
attention inputs/output under attn_in/), output, loss, gradients (frozen-path
inputs, LoRA A/B, and per-attention dL/dq, dL/dk, dL/dv plus the upstream
gradient into each attention output under grad/attn/) and the AdamW updates
after 1 and 10 steps. meta.json holds the config, the fp32-vs-bf16 noise-floor
report and the determinism check result.
"""

import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

CONFIG = {
    "dim": 256,
    "heads": 2,               # head_dim 128 is fixed by TESTING.md; small width
    "head_dim": 128,
    "rope_axes": [32, 32, 32, 32],
    "rope_theta": 10000.0,
    "img_grid": 4,            # 4x4 = 16 image tokens
    "txt_len": 8,
    "mlp_inner": 512,
    "lora_rank": 4,
    "lora_alpha": 8.0,
    "lora_targets": ["double.img.to_q", "double.img.to_v", "single.to_qkv"],
    "seed": 20261028,
    "lr": 1e-3,
    "adamw_betas": [0.9, 0.999],
    "adamw_eps": 1e-8,
    "weight_decay": 0.0,
    "opt_steps": 10,
    "norm_eps": 1e-6,
}

OUT_DIR = Path(__file__).resolve().parent


# ----- ops -------------------------------------------------------------------

def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm over the last dim; statistics in fp32, result in x's dtype."""
    xf = x.float()
    scale = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (xf * scale).to(x.dtype) * weight


def layer_norm_mod(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor, eps: float) -> torch.Tensor:
    """LayerNorm(affine=False) then adaLN: (1 + scale) * norm + shift."""
    xf = F.layer_norm(x.float(), (x.shape[-1],), eps=eps)
    return (xf * (1.0 + scale.float()) + shift.float()).to(x.dtype)


def rope_tables(positions: torch.Tensor, axes: list[int], theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """positions: (seq, n_axes) float -> cos/sin of shape (seq, sum(axes))."""
    parts_c, parts_s = [], []
    for axis, dim_a in enumerate(axes):
        half = dim_a // 2
        inv = theta ** (-torch.arange(half, dtype=torch.float32) * 2.0 / dim_a)
        freqs = positions[:, axis].unsqueeze(-1) * inv  # (seq, half)
        emb = torch.cat([freqs, freqs], dim=-1)          # (seq, dim_a)
        parts_c.append(emb.cos())
        parts_s.append(emb.sin())
    return torch.cat(parts_c, dim=-1), torch.cat(parts_s, dim=-1)


def rope_rotate_half(x: torch.Tensor, axes: list[int]) -> torch.Tensor:
    out = []
    off = 0
    for dim_a in axes:
        half = dim_a // 2
        seg = x[..., off:off + dim_a]
        x1, x2 = seg[..., :half], seg[..., half:]
        out.append(torch.cat([-x2, x1], dim=-1))
        off += dim_a
    return torch.cat(out, dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, axes: list[int]) -> torch.Tensor:
    """x: (seq, heads, head_dim) in compute dtype; cos/sin fp32 tables (seq, head_dim)."""
    xf = x.float()
    rot = rope_rotate_half(xf, axes)
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (xf * cos + rot * sin).to(x.dtype)


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """q,k,v: (seq, heads, head_dim). Explicit softmax in fp32 (reference semantics)."""
    scale = 1.0 / math.sqrt(q.shape[-1])
    qt, kt, vt = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)  # (h, s, d)
    scores = (qt.float() @ kt.float().transpose(-1, -2)) * scale
    probs = torch.softmax(scores, dim=-1).to(q.dtype)
    out = probs @ vt
    return out.transpose(0, 1)  # (s, h, d)


class LoRALinear:
    """y = W x + b + (alpha/rank) * B (A x). A/B are the trainable LoRA tensors."""

    def __init__(self, name: str, out_f: int, in_f: int, rank: int, alpha: float, gen: torch.Generator):
        self.name = name
        self.weight = torch.randn(out_f, in_f, generator=gen) * 0.02
        self.bias = torch.zeros(out_f)
        self.a = torch.randn(rank, in_f, generator=gen) * 0.02
        self.b = torch.randn(out_f, rank, generator=gen) * 0.01  # nonzero so LoRA grads are nonzero
        self.scale = alpha / rank

    def params(self) -> dict[str, torch.Tensor]:
        return {f"weights/{self.name}.weight": self.weight,
                f"weights/{self.name}.bias": self.bias,
                f"weights/{self.name}.lora_a": self.a,
                f"weights/{self.name}.lora_b": self.b}

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias)
        return y + self.scale * F.linear(F.linear(x, self.a), self.b)


class PlainLinear:
    def __init__(self, name: str, out_f: int, in_f: int, gen: torch.Generator):
        self.name = name
        self.weight = torch.randn(out_f, in_f, generator=gen) * 0.02
        self.bias = torch.zeros(out_f)

    def params(self) -> dict[str, torch.Tensor]:
        return {f"weights/{self.name}.weight": self.weight, f"weights/{self.name}.bias": self.bias}

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class SwiGLU:
    def __init__(self, name: str, dim: int, inner: int, gen: torch.Generator):
        self.gate = PlainLinear(f"{name}.gate", inner, dim, gen)
        self.up = PlainLinear(f"{name}.up", inner, dim, gen)
        self.down = PlainLinear(f"{name}.down", dim, inner, gen)

    def params(self) -> dict[str, torch.Tensor]:
        return {**self.gate.params(), **self.up.params(), **self.down.params()}

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


# ----- blocks ----------------------------------------------------------------

class DoubleBlock:
    """One FLUX-style double-stream (MMDiT) block."""

    def __init__(self, gen: torch.Generator):
        d, hd, r, al = CONFIG["dim"], CONFIG["head_dim"], CONFIG["lora_rank"], CONFIG["lora_alpha"]
        self.mod = PlainLinear("double.mod", 6 * d, d, gen)
        self.img_q = LoRALinear("double.img.to_q", d, d, r, al, gen) if "double.img.to_q" in CONFIG["lora_targets"] else PlainLinear("double.img.to_q", d, d, gen)
        self.img_k = PlainLinear("double.img.to_k", d, d, gen)
        self.img_v = LoRALinear("double.img.to_v", d, d, r, al, gen) if "double.img.to_v" in CONFIG["lora_targets"] else PlainLinear("double.img.to_v", d, d, gen)
        self.img_out = PlainLinear("double.img.to_out", d, d, gen)
        self.txt_q = PlainLinear("double.txt.to_q", d, d, gen)
        self.txt_k = PlainLinear("double.txt.to_k", d, d, gen)
        self.txt_v = PlainLinear("double.txt.to_v", d, d, gen)
        self.txt_out = PlainLinear("double.txt.to_out", d, d, gen)
        self.qk_norm = {n: torch.ones(hd) for n in ("img_q", "img_k", "txt_q", "txt_k")}
        self.img_mlp = SwiGLU("double.img.mlp", d, CONFIG["mlp_inner"], gen)
        self.txt_mlp = SwiGLU("double.txt.mlp", d, CONFIG["mlp_inner"], gen)

    def params(self) -> dict[str, torch.Tensor]:
        out = {}
        for m in (self.mod, self.img_q, self.img_k, self.img_v, self.img_out,
                  self.txt_q, self.txt_k, self.txt_v, self.txt_out, self.img_mlp, self.txt_mlp):
            out.update(m.params())
        for n, w in self.qk_norm.items():
            out[f"weights/double.{n}_norm.weight"] = w
        return out

    def forward(self, img, txt, temb, cos_i, sin_i, cos_t, sin_t, acts: dict):
        eps = CONFIG["norm_eps"]
        axes = CONFIG["rope_axes"]
        h, hd = CONFIG["heads"], CONFIG["head_dim"]
        sh1, sc1, g1, sh2, sc2, g2 = self.mod(F.silu(temb)).chunk(6)

        ih = layer_norm_mod(img, sh1, sc1, eps)
        th = layer_norm_mod(txt, sh1, sc1, eps)
        acts["double.img_norm1"] = ih
        acts["double.txt_norm1"] = th

        q_i = self.img_q(ih).view(-1, h, hd)
        k_i = self.img_k(ih).view(-1, h, hd)
        v_i = self.img_v(ih).view(-1, h, hd)
        q_t = self.txt_q(th).view(-1, h, hd)
        k_t = self.txt_k(th).view(-1, h, hd)
        v_t = self.txt_v(th).view(-1, h, hd)
        q_i = rms_norm(q_i, self.qk_norm["img_q"], eps)
        k_i = rms_norm(k_i, self.qk_norm["img_k"], eps)
        q_t = rms_norm(q_t, self.qk_norm["txt_q"], eps)
        k_t = rms_norm(k_t, self.qk_norm["txt_k"], eps)
        q_i, k_i = apply_rope(q_i, cos_i, sin_i, axes), apply_rope(k_i, cos_i, sin_i, axes)
        q_t, k_t = apply_rope(q_t, cos_t, sin_t, axes), apply_rope(k_t, cos_t, sin_t, axes)
        acts["double.q"], acts["double.k"] = q_i, k_i  # image-stream q/k after norm+rope

        q = torch.cat([q_t, q_i], dim=0)
        k = torch.cat([k_t, k_i], dim=0)
        v = torch.cat([v_t, v_i], dim=0)
        attn = attention(q, k, v)
        # attention inputs/output for the 0.5.4 isolated backward test
        acts["attn_in/double.q"], acts["attn_in/double.k"], acts["attn_in/double.v"] = q, k, v
        acts["attn_in/double.attn"] = attn
        n_t = txt.shape[0]
        a_t, a_i = attn[:n_t], attn[n_t:]
        acts["double.attn_img"] = a_i

        img = img + g1 * self.img_out(a_i.reshape(a_i.shape[0], -1))
        txt = txt + g1 * self.txt_out(a_t.reshape(a_t.shape[0], -1))
        acts["double.img_post_attn"] = img

        ih2 = layer_norm_mod(img, sh2, sc2, eps)
        th2 = layer_norm_mod(txt, sh2, sc2, eps)
        img = img + g2 * self.img_mlp(ih2)
        txt = txt + g2 * self.txt_mlp(th2)
        acts["double.img_out"], acts["double.txt_out"] = img, txt
        return img, txt


class SingleBlock:
    """One FLUX-style single-stream block: fused QKV + fused SwiGLU from one normed input."""

    def __init__(self, gen: torch.Generator):
        d, hd, r, al = CONFIG["dim"], CONFIG["head_dim"], CONFIG["lora_rank"], CONFIG["lora_alpha"]
        self.mod = PlainLinear("single.mod", 6 * d, d, gen)
        self.qkv = LoRALinear("single.to_qkv", 3 * d, d, r, al, gen) if "single.to_qkv" in CONFIG["lora_targets"] else PlainLinear("single.to_qkv", 3 * d, d, gen)
        self.q_norm = torch.ones(hd)
        self.k_norm = torch.ones(hd)
        self.mlp_gate = PlainLinear("single.mlp.gate", CONFIG["mlp_inner"], d, gen)
        self.mlp_up = PlainLinear("single.mlp.up", CONFIG["mlp_inner"], d, gen)
        self.mlp_down = PlainLinear("single.mlp.down", d, CONFIG["mlp_inner"], gen)

    def params(self) -> dict[str, torch.Tensor]:
        return {**self.mod.params(), **self.qkv.params(),
                "weights/single.q_norm.weight": self.q_norm,
                "weights/single.k_norm.weight": self.k_norm,
                **self.mlp_gate.params(), **self.mlp_up.params(), **self.mlp_down.params()}

    def forward(self, x, temb, cos, sin, acts: dict):
        eps = CONFIG["norm_eps"]
        axes, h, hd = CONFIG["rope_axes"], CONFIG["heads"], CONFIG["head_dim"]
        sh1, sc1, g_at, sh2, sc2, g_mlp = self.mod(F.silu(temb)).chunk(6)

        nh = layer_norm_mod(x, sh1, sc1, eps)
        acts["single.norm1"] = nh

        q, k, v = self.qkv(nh).view(-1, 3, h, hd).unbind(1)
        q = apply_rope(rms_norm(q, self.q_norm, eps), cos, sin, axes)
        k = apply_rope(rms_norm(k, self.k_norm, eps), cos, sin, axes)
        acts["single.q"], acts["single.k"] = q, k
        attn3 = attention(q, k, v)
        acts["attn_in/single.q"], acts["attn_in/single.k"], acts["attn_in/single.v"] = q, k, v
        acts["attn_in/single.attn"] = attn3
        attn = attn3.reshape(x.shape[0], -1)
        acts["single.attn"] = attn

        mlp = self.mlp_down(F.silu(self.mlp_gate(nh)) * self.mlp_up(nh))
        acts["single.mlp"] = mlp
        out = x + g_at * attn + g_mlp * mlp
        acts["single.out"] = out
        return out


# ----- pipeline ---------------------------------------------------------------

def build_model(gen: torch.Generator):
    return DoubleBlock(gen), SingleBlock(gen)


def run(dtype: torch.Tensor.dtype, record_opt: bool):
    """Full forward + backward + AdamW on LoRA params. Returns (tensors, loss)."""
    torch.manual_seed(CONFIG["seed"])
    gen = torch.Generator().manual_seed(CONFIG["seed"])
    double, single = build_model(gen)

    d, hd = CONFIG["dim"], CONFIG["head_dim"]
    g, tl = CONFIG["img_grid"], CONFIG["txt_len"]
    img = torch.randn(g * g, d, generator=gen)
    txt = torch.randn(tl, d, generator=gen)
    temb = torch.randn(d, generator=gen)
    target_img = torch.randn(g * g, d, generator=gen)
    target_txt = torch.randn(tl, d, generator=gen)

    # 4-axis positions: img axis0=row, axis1=col; txt axis0=index; axes 2-3 stay 0.
    rows = torch.arange(g * g) // g
    cols = torch.arange(g * g) % g
    pos_img = torch.stack([rows.float(), cols.float(), torch.zeros(g * g), torch.zeros(g * g)], dim=-1)
    pos_txt = torch.stack([torch.arange(tl).float(), torch.zeros(tl), torch.zeros(tl), torch.zeros(tl)], dim=-1)
    cos_i, sin_i = rope_tables(pos_img, CONFIG["rope_axes"], CONFIG["rope_theta"])
    cos_t, sin_t = rope_tables(pos_txt, CONFIG["rope_axes"], CONFIG["rope_theta"])
    # single block sees the concatenated sequence, img positions then txt positions
    pos_all = torch.cat([pos_img, pos_txt], dim=0)
    cos_a, sin_a = rope_tables(pos_all, CONFIG["rope_axes"], CONFIG["rope_theta"])

    def cast(t: torch.Tensor) -> torch.Tensor:
        return t.to(dtype)

    params = {}
    for p in (double.params(), single.params()):
        params.update(p)
    params = {k: v.to(dtype) for k, v in params.items()}
    # rebind module tensors to the cast copies
    def rebind(mod):
        for name in list(vars(mod)):
            val = vars(mod)[name]
            if torch.is_tensor(val):
                vars(mod)[name] = val.to(dtype)
            elif isinstance(val, dict):
                vars(mod)[name] = {kk: vv.to(dtype) for kk, vv in val.items()}
            elif isinstance(val, (PlainLinear, LoRALinear, SwiGLU)):
                rebind(val)
    rebind(double)
    rebind(single)
    for m in _loras(double) + _loras(single):
        m.a.requires_grad_(True)
        m.b.requires_grad_(True)

    img, txt = img.to(dtype), txt.to(dtype)
    temb = temb.to(dtype)
    target_img, target_txt = target_img.to(dtype), target_txt.to(dtype)
    cos_i, sin_i, cos_t, sin_t, cos_a, sin_a = (t.to(dtype) for t in (cos_i, sin_i, cos_t, sin_t, cos_a, sin_a))

    img_in = img.clone().requires_grad_(True)
    txt_in = txt.clone().requires_grad_(True)

    acts: dict[str, torch.Tensor] = {
        "inputs/img": img_in, "inputs/txt": txt_in, "inputs/temb": temb,
        "inputs/target_img": target_img, "inputs/target_txt": target_txt,
    }
    img2, txt2 = double.forward(img_in, txt_in, temb, cos_i, sin_i, cos_t, sin_t, acts)
    joint = torch.cat([txt2, img2], dim=0)
    joint_out = single.forward(joint, temb, cos_a, sin_a, acts)
    final_txt, final_img = joint_out[:tl], joint_out[tl:]
    acts["output/img"], acts["output/txt"] = final_img, final_txt

    loss = F.mse_loss(final_img, target_img) + F.mse_loss(final_txt, target_txt)
    acts["loss"] = loss
    loss.backward(retain_graph=True)

    out: dict[str, torch.Tensor] = {}
    for name, t in acts.items():
        out[name] = t.detach().clone()
    for name, t in params.items():
        out[name] = t.detach().clone()
    out["grad/inputs/img"] = img_in.grad.detach().clone()
    out["grad/inputs/txt"] = txt_in.grad.detach().clone()
    # 0.5.4 isolated attention-backward test: dL/dq, dL/dk, dL/dv plus the
    # upstream gradient flowing into each attention output (the seed for a
    # standalone custom-op backward check).
    for tag in ("double", "single"):
        qn, kn, vn, an = (acts[f"attn_in/{tag}.{n}"] for n in ("q", "k", "v", "attn"))
        gq, gk, gv, ga = torch.autograd.grad(
            loss, [qn, kn, vn, an], retain_graph=True
        )
        out[f"grad/attn/{tag}.q"] = gq.detach().clone()
        out[f"grad/attn/{tag}.k"] = gk.detach().clone()
        out[f"grad/attn/{tag}.v"] = gv.detach().clone()
        out[f"grad/attn/{tag}.out"] = ga.detach().clone()
    lora_params = []
    for mod in (double, single):
        for m in _loras(mod):
            out[f"grad/{m.name}.lora_a"] = m.a.grad.detach().clone()
            out[f"grad/{m.name}.lora_b"] = m.b.grad.detach().clone()
            lora_params.extend([m.a, m.b])

    if record_opt:
        opt = torch.optim.AdamW(lora_params, lr=CONFIG["lr"], betas=CONFIG["adamw_betas"],
                                eps=CONFIG["adamw_eps"], weight_decay=CONFIG["weight_decay"])
        for step in range(1, CONFIG["opt_steps"] + 1):
            for p in lora_params:
                p.grad = None
            acts2: dict[str, torch.Tensor] = {}
            img2b, txt2b = double.forward(img_in.detach(), txt_in.detach(), temb, cos_i, sin_i, cos_t, sin_t, acts2)
            joint_b = torch.cat([txt2b, img2b], dim=0)
            joint_out_b = single.forward(joint_b, temb, cos_a, sin_a, acts2)
            l = F.mse_loss(joint_out_b[tl:], target_img) + F.mse_loss(joint_out_b[:tl], target_txt)
            l.backward()
            opt.step()
            if step in (1, CONFIG["opt_steps"]):
                for m in _loras(double) + _loras(single):
                    out[f"opt{step}/{m.name}.lora_a"] = m.a.detach().clone()
                    out[f"opt{step}/{m.name}.lora_b"] = m.b.detach().clone()
    return out


def _loras(mod) -> list[LoRALinear]:
    return [v for v in vars(mod).values() if isinstance(v, LoRALinear)]


def compare(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> dict:
    """fp32 (a) vs bf16 (b) noise floor for activations, grads, opt updates, loss."""
    report = {}
    skip = ("inputs/", "weights/")
    for name, ta in a.items():
        if name.startswith(skip) or name not in b:
            continue
        tb = b[name].float()
        taf = ta.float()
        rel = (taf - tb).norm() / (taf.norm() + 1e-12)
        cos = F.cosine_similarity(taf.flatten().unsqueeze(0), tb.flatten().unsqueeze(0)).item()
        report[name] = {"rel_l2": rel.item(), "cosine": cos}
    return report


def main() -> int:
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)

    fp32_a = run(torch.float32, record_opt=True)
    fp32_b = run(torch.float32, record_opt=True)
    det = all(torch.equal(fp32_a[k], fp32_b[k]) for k in fp32_a)
    if not det:
        print("DETERMINISM: FAIL (fp32 repeat differs)", file=sys.stderr)
        return 1
    print("DETERMINISM: PASS (fp32 repeat bit-identical)")

    bf16 = run(torch.bfloat16, record_opt=True)
    floor = compare(fp32_a, bf16)

    save_file({k: v.contiguous() for k, v in fp32_a.items()}, OUT_DIR / "block_fp32.safetensors")
    save_file({k: v.contiguous() for k, v in bf16.items()}, OUT_DIR / "block_bf16.safetensors")

    meta = {
        "config": CONFIG,
        "determinism_fp32_repeat": "pass",
        "noise_floor_fp32_vs_bf16": floor,
        "thresholds_initial": {"rel_l2_max": 2e-2, "cosine_min": 0.999},
        "over_threshold": {k: v for k, v in floor.items()
                           if v["rel_l2"] > 2e-2 or v["cosine"] < 0.999},
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"tensors: {len(fp32_a)} fp32 / {len(bf16)} bf16")
    worst = max(floor.items(), key=lambda kv: kv[1]["rel_l2"])
    print(f"noise floor worst rel_l2: {worst[0]} = {worst[1]['rel_l2']:.3e} (cosine {worst[1]['cosine']:.6f})")
    if meta["over_threshold"]:
        print(f"NOTE: {len(meta['over_threshold'])} tensors exceed the initial thresholds "
              "(these become the measured noise floor; see TESTING.md)")
    print("WROTE: block_fp32.safetensors, block_bf16.safetensors, meta.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
