# Diamond Fusion Crystal Language Model — Architecture Spec v1.0

**Authors:** Warren Reed Baum (@delistish), Degen Overlord (AI Agent)  
**Date:** March 25, 2026  
**Target:** OpenAI Parameter Golf Competition

## 1. Abstract

Diamond Fusion Crystal Language Model is a parameter-efficient transformer architecture designed for the OpenAI Parameter Golf Competition. The model uses **one shared transformer block repeated N times** with learned **crystal modulation gates** and periodic **diamond fusion junctions** to approximate the representational depth of a much larger model while keeping the parameter footprint compact.

The core idea is simple: keep the network shallow in parameters, but deep in computation. Rather than allocating distinct weights for every layer, the model reuses the same block across all iterations, while per-iteration gates modulate attention and MLP contributions. A U-Net-like recurrence pattern stores and later re-injects hidden-state skips, giving the model a structured memory of earlier depths.

With this design, the target footprint is approximately **2.9M parameters**, compared with roughly **17M parameters** for a conventional baseline of similar width and depth.

## 2. Competition Constraints

This design is explicitly shaped around the competition envelope:

- **Artifact limit:** 16 MB
- **Training budget:** 10 minutes on 8x H100
- **Evaluation metric:** BPB on FineWeb
- **Tokenizer:** 1024 BPE vocabulary
- **Current SOTA:** 1.1194 BPB

The architecture prioritizes parameter reuse, small state, and fast iteration. The intent is to spend compute on recurrence and structured composition rather than on weight duplication.

## 3. Core Innovations

### 3.1 Depth Recurrence

The model adopts a **depth recurrence** pattern inspired by the Scale Inversion idea from the Ternary Crystal paper. Instead of stacking many unique blocks, a single shared block is replayed multiple times. This turns depth into a controllable process rather than a fixed structural cost.

### 3.2 Crystal Modulation Gates

For each iteration `i`, the model learns two gates:

- `attn_gate[i]`
- `mlp_gate[i]`

These gates are vectors, not scalars, and they control how strongly the shared block's attention and MLP outputs contribute at that iteration. A practical initialization is `1 / sqrt(N)` so early training remains stable when the recurrence count `N` is large.

### 3.3 U-Net Diamond Skip Connections

The architecture uses a U-Net-like depth layout:

- **Encoder half:** stores skip tensors at each iteration
- **Decoder half:** consumes those skips in reverse order

This creates a diamond-shaped information flow where the middle of the network acts as a hinge between compression and reconstruction. The result is better preservation of low-level structure while still allowing deep iterative refinement.

### 3.4 Diamond Fusion Multi-Expert

The multi-expert variant introduces **K experts** with different gate patterns. Every few iterations, the experts are fused through a diamond consensus step:

\[
 h_{fused} = \sum_i \alpha_i h_i + \beta \cdot agreement(h_1, \ldots, h_K)
\]

Where:
- `alpha_i` are softmax fusion weights
- `agreement(...)` is a bonus term based on expert agreement, implemented via variance reduction / consensus shaping
- `beta` controls the strength of the agreement signal

This encourages diversity without letting the experts drift into incoherence.

## 4. Component Details

### 4.1 Shared Crystal Block

The shared block contains:

- residual mixing
- grouped query attention (GQA): **8 query heads / 4 KV heads**, head dim 64
- partial RoPE on the first **16 / 64** dimensions
- QK normalization with learned gain
- MLP using **LeakyReLU(0.5)^2** nonlinearity
- **3x MLP expansion**
- zero-initialized output projections for stable recurrence

### 4.2 CrystalGPT

CrystalGPT is the single-expert recurrent model:

`embed -> RMSNorm -> encoder(6 iters with skips) -> decoder(6 iters consuming skips) -> RMSNorm -> tied logit head -> softcap(30)`

This arrangement makes the first half of depth responsible for building latent structure and the second half responsible for reconstructing and sharpening it.

### 4.3 CrystalMoE

CrystalMoE extends the same scaffold to multiple experts:

- same shared block topology
- per-expert gates
- periodic diamond fusion every 3 iterations
- consensus-based mixing across experts

The intended effect is specialization with controlled recombination.

## 5. Parameter Budget

Estimated parameter allocation:

| Component | Params |
|---|---:|
| Embedding | 524K |
| Attention | 786K |
| MLP | 1573K |
| Scales | 2K |
| Gates | 12K |
| Skips | 3K |
| **Total** | **~2.9M** |

This budget is deliberately compact and competition-friendly. The design goal is to remain well under the artifact cap while preserving enough expressivity to compete on BPB.

## 6. Hardware Strategy

Development and validation are staged across available hardware:

- **M4 Mini:** MLX smoke tests and quick correctness checks
- **Mac Studio:** integration work, commits, and local iteration
- **4090:** PyTorch sweeps and broader ablation experiments
- **H100 final run:** competition-grade training and final submission selection

This keeps the fast feedback loop on Apple silicon while reserving large-scale experimentation for the stronger GPU boxes.

## 7. Research Program

The research path is staged intentionally:

1. baseline transformer
2. crystal recurrence
3. diamond fusion
4. ablations
5. paper / writeup

Each stage isolates one novelty so the effect of recurrence, gating, and fusion can be measured cleanly.

## 8. References

- Ternary Crystal paper
- Parameter Golf repo
