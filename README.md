# trtllm-vs-vllm

**消费级 Blackwell 显卡上的跨框架 × 跨精度大模型推理 benchmark。**

在单张 **RTX 5070 Ti（Blackwell, sm_120, 16 GB）** + WSL2 上，把 TensorRT-LLM 与 vLLM、NVFP4 与 FP8 放在并发维度上系统对照 —— 并包含一次对「吞吐随并发非单调」反直觉现象的完整根因定位。

> `TensorRT-LLM 1.3.0rc15` · `vLLM 0.13.0` · `NVFP4 / FP8` · `Llama-3.1-8B-Instruct` · `WSL2 + Docker`

---

## 速览（TL;DR）

- **高并发吞吐 NVFP4 > FP8，两个框架内都成立**（decode 是 memory-bound，4bit 权重搬运更少）。TRT-LLM @c32：**3572 vs 2010 tok/s（+78%）**；vLLM @c64：**5271 vs 4269 tok/s（+23%）**。
- **两个框架各有所长。** vLLM 的 **TTFT** 全程更低（c1：~14ms vs TRT-LLM ~42ms，利于交互）；TRT-LLM 在 graph 捕获的并发点上 **TPOT** 更低，批吞吐更猛。
- **FP16 在这张卡上不可部署。** 8B × 2 字节 = 16GB ＝ 整卡 16303 MiB，连 KV cache 都放不下 → 必 OOM。量化在这里不是可选项，是前提。
- **重点发现 —— 一条被 debug 出来的非单调曲线。** TRT-LLM NVFP4 吞吐在并发 40/48/56 处 *塌陷 2.5×*，到 64 又满血。在**连续证伪三个假说**后，根因定位到 **CUDA graph 捕获空洞**（33–63 这段 batch size 从未被捕获 → 回退 eager），用 `enable_padding` 实测 **2.5× 修复**。

---

## 硬件 / 软件

| | |
|---|---|
| GPU | NVIDIA RTX 5070 Ti — Blackwell, **sm_120**, 16303 MiB |
| 驱动 / CUDA | 610.47 / 13.1 |
| TensorRT-LLM | 1.3.0rc15（PyTorch backend，FlashInfer 注意力） |
| vLLM | v0.13.0+faa43dbf.nv26.01（FLASH_ATTN，flashinfer-cutlass NVFP4 GEMM） |
| 模型 | `nvidia/Llama-3.1-8B-Instruct-NVFP4` / `-FP8`（ModelOpt 预量化） |
| 宿主 | Windows + WSL2（Ubuntu, systemd）+ Docker + nvidia-container-toolkit |

两个 NVFP4 端点都验证过走的是 **sm_120 原生 FP4 GEMM，而非 Marlin weight-only 回退** —— 后者是个静默陷阱，一旦回退跨框架对照就变成苹果比橘子。详见 [`SETUP.md`](SETUP.md) 第9节。

---

## 测量

**矩阵：** 框架 `{TRT-LLM, vLLM}` × 精度 `{NVFP4, FP8}` × 并发 `{1, 2, 4, 8, 16, 32, 48, 64}`（FP16 = 解析判定的部署边界）。

**指标**（OpenAI 兼容流式，自建 harness）：

- **TTFT** —— 首 token 延迟（prefill 代理）
- **TPOT** —— 稳态每 token 时间
- **ITL** —— token 间隔分布
- **QPS** / 输出 tok/s —— 吞吐

**协议**（控变量）：固定 prompt（ISL 恒定 —— 同模型即同 tokenizer）、`max_tokens=256` + `ignore_eos`（钉死 OSL）、greedy 解码、warmup 轮丢弃、各框架默认注意力后端（声明而不强行统一）。两个框架都暴露 `/v1/completions` 流式，因此 harness 框架无关，只换 `base_url`。

---

## 核心结果

完整表见 [`SETUP.md`](SETUP.md) 10.4 与 [`results/summary.csv`](results/summary.csv)。吞吐（输出 tok/s）速览：

| 并发 | trtllm-nvfp4 | trtllm-fp8 | vllm-nvfp4 | vllm-fp8 |
|---|---|---|---|---|
| 8  |  978 |  712 |  840 |  641 |
| 16 | 1891 | 1367 | 1634 | 1298 |
| 32 | **3572** | 2010 | 3035 | 2154 |
| 48 | 1949 † | 2621 | 4234 | 3421 |
| 64 | 6453 | 3415 | **5271** | 4269 |

† TRT-LLM NVFP4 @c40/48/56 是**默认配置**下落在 CUDA graph 捕获空洞里的值（见下）。开 `enable_padding` 后 c48 → **4681 tok/s**。

---

## 重点：debug 一条非单调吞吐曲线

TRT-LLM NVFP4 吞吐随并发**非单调**：

```
c32: 3572 tok/s, TPOT  8.8 ms   （正常）
c40: 1766 tok/s, TPOT 22.4 ms   ← 塌陷
c48: 1949 tok/s, TPOT 24.4 ms   ← 塌陷
c56: 2485 tok/s, TPOT 22.3 ms   ← 塌陷
c64: 6453 tok/s, TPOT  9.6 ms   （满血反弹）
```

vLLM 同条件下平滑单调。排查过程：

1. ❌ **KV cache 抢占** —— KV 池 = 4102 blocks × 32 = 131k tokens；每请求 ~450 tokens ⇒ 并发上限 ~290。c40 远未触及。
2. ❌ **max_batch_size** —— 启动日志 `max_batch_size=2048`，不卡。
3. ❌ **max_num_tokens（prefill 预算）** —— *控制变量实验：* 8192 → 16384 翻倍，凹陷带逐点不变。**证伪。**
4. ✅ **CUDA graph 捕获空洞** —— 启动日志捕获列表为 `128, 64, 32, 31, …, 16`；**33–63 这段从未捕获**。并发 40/48/56 落入空洞 → 回退 eager 执行 → TPOT 翻倍。c64 命中捕获点 → 满血。

**验证 + 解决** —— 传 `cuda_graph_config: {enable_padding: true}`（把非捕获 size 向上 pad 到最近 graph）：

| c | 默认（空洞） | + enable_padding | 提升 |
|---|---|---|---|
| 40 | 1766 tok/s | **4359** | 2.5× |
| 48 | 1949 tok/s | **4681** | 2.4× |
| 56 | 2485 tok/s | **5851** | 2.4× |

凹陷带消失、曲线恢复单调 —— 根因被实证。vLLM 因捕获列表更密 + 默认 padding，开箱即平滑。

---

## 复现

**前置：** WSL2 + Docker + nvidia-container-toolkit、一张 Blackwell 卡，以及宿主机预先下好的预量化 checkpoint（受限网络下的解耦下载法见 [`SETUP.md`](SETUP.md) 2、5）。

```bash
# 宿主机下载 checkpoint
hf download nvidia/Llama-3.1-8B-Instruct-NVFP4 --local-dir ./models/llama31-8b-nvfp4
hf download nvidia/Llama-3.1-8B-Instruct-FP8   --local-dir ./models/llama31-8b-fp8
```

**一次起一个端点**（16GB 只放得下一个 8B 端点）：

```bash
# vLLM
vllm serve ./models/llama31-8b-nvfp4 --served-model-name m \
  --max-model-len 4096 --gpu-memory-utilization 0.9 --port 8000

# TRT-LLM
trtllm-serve serve ./models/llama31-8b-nvfp4 --backend pytorch --tp_size 1 \
  --host 0.0.0.0 --port 8000
# + CUDA graph padding 修复：
#   --extra_llm_api_options configs/pad.yaml
```

**测量 + 汇总：**

```bash
python3 scripts/bench.py --tag vllm_nvfp4 --concurrencies 1,2,4,8,16,32,48,64 --rounds 10
# ... 每个 (框架, 精度) cell 重复，切换正在运行的 server ...
python3 scripts/summarize.py        # 汇总 results_*.json → 表格 + summary.csv
```

`bench.py` 通过 `/v1/models` 自动发现 model id，所以同一条命令打两个框架都通。

---

## 仓库结构

```
.
├── README.md          
├── SETUP.md                 # 完整工程日志（环境基线 → benchmark → debug）
├── scripts/
│   ├── bench.py             # 框架无关的 OpenAI 流式 benchmark runner
│   ├── summarize.py         # 汇总 results_*.json → 表格 + CSV
│   └── smoke/               # 各框架 / 各精度的冒烟脚本
│       ├── smoke.py
│       ├── nvfp4_smoke.py
│       └── vllm_nvfp4_smoke.py
├── configs/
│   └── pad.yaml             # cuda_graph_config: enable_padding（修复）
└── results/
    ├── summary.csv
    └── results_*.json
```

---

## 完整工程日志

[`SETUP.md`](SETUP.md) 是逐步的完整记录：环境基线、国内网络的解耦下载法、FP8 的 FlashInfer JIT 编译 OOM 及其解法、native-vs-Marlin 验证、完整 benchmark 矩阵，以及含每一个被证伪假说的 CUDA graph 排查全过程。

## 在国内运行的说明

持续的网络约束催生了**下载与推理解耦**的模式（宿主机走 VPN 拉权重 + 容器只读挂载），以及针对 `nvcr.io` 的 dockerd 代理 + CDN 域名放行的绕法。FP8 路径还额外要求宿主机 **≥26 GB 内存**（或限制 nvcc 并行度）—— 因为它的 sm_120 kernel 在首次启动时现场 JIT 编译，多路并行 `nvcc` 会把默认 15 GB 的 WSL 打爆 OOM。详见 `SETUP.md` §5 与 §10.3。
