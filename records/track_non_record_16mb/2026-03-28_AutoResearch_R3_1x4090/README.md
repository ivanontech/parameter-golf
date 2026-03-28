# Non-Record Submission: 1.1957 BPB — AutoResearch R3 (1×RTX 4090)

**Score:** 1.1957 BPB  
**Parameters:** 100.9M  
**Hardware:** 1× NVIDIA RTX 4090 (24GB)  
**Training time:** 300.3s (5 min budget)  
**Total time (incl. eval):** 394.4s  
**Steps:** 408  
**MFU:** 4.42%  
**Peak VRAM:** 18.6 GB  

## Architecture

- 12-layer transformer, 1024 embedding dim, 8 attention heads, 8 KV heads
- Value Embeddings (31.4M params) with gated fusion on alternating layers  
- Window pattern: L (all long-range attention)
- Vocab: 32768 BPE
- Sequence length: 2048
- MLP ratio: 4x (standard)
- AdamW + Muon optimizer with cosine warmdown schedule
- TOTAL_BATCH_SIZE: 2^16 (64K tokens) — key insight: smaller batches = more steps = better convergence
- Gradient accumulation: 2 microbatches
- No warmup (WARMUP_RATIO=0.0)
- MATRIX_LR=0.10, WEIGHT_DECAY=0.2

## Key Insight

Reducing batch size from 2^19 (default) to 2^16 increased training steps from 63 to 408 within the same 5-minute budget. The 6.5x increase in gradient updates dramatically improved convergence, dropping val_bpb from 1.476 to 1.196. This was the single biggest improvement found across 3 rounds of automated hyperparameter search.

## Method

Automated hyperparameter search (autoresearch) with systematic sweeps across:
- R1: Device batch size, model width/depth, architecture variants
- R2: Learning rate sweep, warmup, weight decay
- R3: **Batch size reduction** (breakthrough), window patterns, vocab size, LR re-sweep

31 total experiments, 11 in R3. Based on [karpathy/autoresearch](https://github.com/karpathy/autoresearch).
