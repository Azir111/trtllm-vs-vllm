# trtllm-vs-vllm

**Cross-framework × cross-precision LLM inference benchmark on a consumer Blackwell GPU.**

TensorRT-LLM vs. vLLM, NVFP4 vs. FP8, swept across concurrency on a single **RTX 5070 Ti (Blackwell, sm_120, 16 GB)** under WSL2 — including a full root-cause investigation of a non-monotonic throughput curve.

> `TensorRT-LLM 1.3.0rc15` · `vLLM 0.13.0` · `NVFP4 / FP8` · `Llama-3.1-8B-Instruct` · `WSL2 + Docker`

---

## TL;DR

- **NVFP4 beats FP8 on high-concurrency throughput** in both frameworks (decode is memory-bound; 4-bit weights move less). TRT-LLM @c32: **3572 vs 2010 tok/s (+78%)**; vLLM @c64: **5271 vs 4269 tok/s (+23%)**.
- **No framework wins outright — they trade off.** vLLM has consistently lower **TTFT** (c1: ~14 ms vs TRT-LLM ~42 ms → better for interactive). TRT-LLM has lower steady-state **TPOT** at graph-captured batch sizes → better for batch throughput.
- **FP16 is not deployable on this card.** 8B × 2 bytes = 16 GB = the entire 16303 MiB board, before any KV cache → guaranteed OOM. Quantization here is a prerequisite, not an option.
- **Headline finding — a debugged non-monotonic curve.** TRT-LLM NVFP4 throughput *drops 2.5×* at concurrency 40/48/56, then recovers at 64. After **falsifying three hypotheses**, the cause was located to **CUDA-graph capture holes** (sizes 33–63 are never captured → fall back to eager). Fixed with `enable_padding` for a **2.5× recovery**.

---

## Hardware & software

| | |
|---|---|
| GPU | NVIDIA RTX 5070 Ti — Blackwell, **sm_120**, 16303 MiB |
| Driver / CUDA | 610.47 / 13.1 |
| TensorRT-LLM | 1.3.0rc15 (PyTorch backend, FlashInfer attention) |
| vLLM | v0.13.0+faa43dbf.nv26.01 (FLASH_ATTN, flashinfer-cutlass NVFP4 GEMM) |
| Model | `nvidia/Llama-3.1-8B-Instruct-NVFP4` / `-FP8` (ModelOpt pre-quantized) |
| Host | Windows + WSL2 (Ubuntu, systemd) + Docker + nvidia-container-toolkit |

Both NVFP4 endpoints were verified to run **native sm_120 FP4 GEMM, not Marlin weight-only fallback** — a silent trap that would have made the cross-framework comparison apples-to-oranges. See [`SETUP.md`](SETUP.md) §9.

---

## What's measured

**Matrix:** framework `{TRT-LLM, vLLM}` × precision `{NVFP4, FP8}` × concurrency `{1, 2, 4, 8, 16, 32, 48, 64}` (FP16 = analytic deploy-boundary).

**Metrics** (OpenAI-compatible streaming, self-built harness):

- **TTFT** — time to first token (prefill latency proxy)
- **TPOT** — steady-state time per output token
- **ITL** — inter-token latency distribution
- **QPS** / output tok/s — throughput

**Protocol** (controlled comparison): fixed prompt (ISL constant — same model ⇒ same tokenizer), `max_tokens=256` + `ignore_eos` (OSL pinned), greedy decoding, warmup round discarded, per-framework default attention backend (declared, not forced equal). Both frameworks expose `/v1/completions` streaming, so the harness is framework-agnostic — only `base_url` changes.

---

## Key results

Full table in [`SETUP.md`](SETUP.md) §10.4 and [`results/summary.csv`](results/summary.csv). Throughput (output tok/s) at a glance:

| concurrency | trtllm-nvfp4 | trtllm-fp8 | vllm-nvfp4 | vllm-fp8 |
|---|---|---|---|---|
| 8  |  978 |  712 |  840 |  641 |
| 16 | 1891 | 1367 | 1634 | 1298 |
| 32 | **3572** | 2010 | 3035 | 2154 |
| 48 | 1949 † | 2621 | 4234 | 3421 |
| 64 | 6453 | 3415 | **5271** | 4269 |

† TRT-LLM NVFP4 @c40/48/56 are *default-config* values sitting in a CUDA-graph capture hole (see below). With `enable_padding`, c48 → **4681 tok/s**.

---

## Highlight: debugging a non-monotonic throughput curve

TRT-LLM NVFP4 throughput was **not monotonic** in concurrency:

```
c32: 3572 tok/s, TPOT  8.8 ms   (normal)
c40: 1766 tok/s, TPOT 22.4 ms   ← collapse
c48: 1949 tok/s, TPOT 24.4 ms   ← collapse
c56: 2485 tok/s, TPOT 22.3 ms   ← collapse
c64: 6453 tok/s, TPOT  9.6 ms   (full speed again)
```

vLLM under identical conditions was smooth and monotonic. Investigation:

1. ❌ **KV-cache preemption** — KV pool = 4102 blocks × 32 = 131k tokens; ~450 tokens/request ⇒ concurrency ceiling ~290. c40 nowhere near it.
2. ❌ **max_batch_size** — startup log shows `max_batch_size=2048`. Not the limit.
3. ❌ **max_num_tokens (prefill budget)** — *controlled experiment:* doubled 8192 → 16384; collapse band unchanged point-for-point. **Falsified.**
4. ✅ **CUDA-graph capture holes** — startup log capture list is `128, 64, 32, 31, …, 16`; **sizes 33–63 are never captured**. Concurrency 40/48/56 fall into the hole → fall back to eager execution → TPOT doubles. c64 hits a captured size → full speed.

**Fix & verify** — pass `cuda_graph_config: {enable_padding: true}` (pads non-captured sizes up to the nearest graph):

| c | default (hole) | + enable_padding | speedup |
|---|---|---|---|
| 40 | 1766 tok/s | **4359** | 2.5× |
| 48 | 1949 tok/s | **4681** | 2.4× |
| 56 | 2485 tok/s | **5851** | 2.4× |

Collapse band gone, curve monotonic — root cause confirmed. vLLM avoids this out-of-the-box via a denser capture list + default padding.

---

## Reproduce

**Prereqs:** WSL2 + Docker + nvidia-container-toolkit, a Blackwell GPU, and the pre-quantized checkpoints downloaded host-side (decoupled download pattern for constrained networks — see [`SETUP.md`](SETUP.md) §2, §5).

```bash
# download checkpoints (host side)
hf download nvidia/Llama-3.1-8B-Instruct-NVFP4 --local-dir ./models/llama31-8b-nvfp4
hf download nvidia/Llama-3.1-8B-Instruct-FP8   --local-dir ./models/llama31-8b-fp8
```

**Serve one endpoint at a time** (16 GB holds one 8B endpoint):

```bash
# vLLM
vllm serve ./models/llama31-8b-nvfp4 --served-model-name m \
  --max-model-len 4096 --gpu-memory-utilization 0.9 --port 8000

# TRT-LLM
trtllm-serve serve ./models/llama31-8b-nvfp4 --backend pytorch --tp_size 1 \
  --host 0.0.0.0 --port 8000
# + CUDA-graph padding fix:
#   --extra_llm_api_options configs/pad.yaml
```

**Benchmark & aggregate:**

```bash
python3 scripts/bench.py --tag vllm_nvfp4 --concurrencies 1,2,4,8,16,32,48,64 --rounds 10
# ... repeat per (framework, precision) cell, swapping the running server ...
python3 scripts/summarize.py        # collates results_*.json -> table + summary.csv
```

`bench.py` auto-discovers the model id via `/v1/models`, so the same command works against both frameworks.

---

## Repo structure

```
.
├── README.md
├── SETUP.md                 # full engineering log (env baseline → benchmark → debug)
├── scripts/
│   ├── bench.py             # framework-agnostic OpenAI-streaming benchmark runner
│   ├── summarize.py         # aggregate results_*.json → table + CSV
│   └── smoke/               # per-framework / per-precision smoke tests
│       ├── smoke.py
│       ├── nvfp4_smoke.py
│       └── vllm_nvfp4_smoke.py
├── configs/
│   └── pad.yaml             # cuda_graph_config: enable_padding (the fix)
└── results/
    ├── summary.csv
    └── results_*.json
```

---

## Full engineering log

[`SETUP.md`](SETUP.md) is the complete, blow-by-blow record (in Chinese): environment baseline, the China-network decoupled-download pattern, the FP8 FlashInfer JIT-compile OOM and its fix, native-vs-Marlin verification, the full benchmark matrix, and the CUDA-graph investigation with every falsified hypothesis.

## Notes on running in mainland China

Persistent network constraints drove a **download/inference decoupling** pattern (host-side VPN pull + container read-only mount) and a dockerd-proxy + CDN-domain workaround for `nvcr.io`. The FP8 path additionally requires a host with **≥26 GB RAM** (or limited nvcc parallelism) because its sm_120 kernels are JIT-compiled on first launch and the parallel `nvcc` jobs OOM a default 15 GB WSL. Details in `SETUP.md` §5 and §10.3.
