# TRT-LLM on RTX 5070Ti — 环境基线 & 冒烟存档

> 存档日期：2026-06-18
> 目的：固化「TRT-LLM 能在 Blackwell 消费卡 (sm_120) 上加载模型并推理」的可复现环境，作为后续 trtllm-vs-vllm benchmark 项目的基线。

---

## 1. 硬件 / 软件版本

| 项 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 Ti (Blackwell, **sm_120**, 16303 MiB) |
| Driver | 610.47 |
| TRT-LLM | **1.3.0rc15** |
| PyTorch | 2.11.0a0+eb65b36914.nv26.02 |
| CUDA | 13.1 |
| 容器 | `nvcr.io/nvidia/tensorrt-llm/release:latest` （拉取于 2026-06-18，压缩 ~20.5GB） |
| 宿主 | Windows + WSL2 (Ubuntu, systemd) + Docker (`docker.io`) + nvidia-container-toolkit |

> **复现注意**：`latest` 标签当时实际指向 **1.3.0rc15**，并非 NGC 页面上最新的 rc18。要严格复现请直接钉版本：
> `docker pull nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc15`
> 该仓库当时**无非-rc 正式版**，最新稳定线即 rc 系列（NV 周更项目常态，rc 经过 CI，可用）。

---

## 2. 一键复现：进容器

下载与推理**解耦**——模型在 WSL 宿主机下好，容器只读本地挂载、全程不联网（详见踩坑 第 5 节）。

```bash
docker run --rm -it --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  --gpus all \
  -v ~/trtllm-smoke:/workspace \
  -w /workspace \
  nvcr.io/nvidia/tensorrt-llm/release:latest
```

模型预先下到宿主机 `~/trtllm-smoke/models/tinyllama`（→ 容器内 `/workspace/models/tinyllama`）：

```bash
# 在 WSL，容器外
cd ~/trtllm-smoke
unset HF_ENDPOINT            # 走官网，不被 hf-mirror 踢回源站
export HF_HUB_DISABLE_XET=1  # 避开 Xet 那跳
pip install -q "huggingface_hub[cli]"
hf download TinyLlama/TinyLlama-1.1B-Chat-v1.0 --local-dir ./models/tinyllama
```

---

## 3. 冒烟脚本 `smoke.py`

```python
from tensorrt_llm import LLM, SamplingParams


def main():
    llm = LLM(model="/workspace/models/tinyllama")
    prompts = ["Hello, my name is", "The capital of France is", "The future of AI is"]
    params = SamplingParams(temperature=0.8, top_p=0.95)

    for out in llm.generate(prompts, params):
        print(f"{out.prompt!r} -> {out.outputs[0].text!r}")


if __name__ == "__main__":   # ← 必须，TRT-LLM 用 MPI 多进程，缺这行会 MPI_ABORT
    main()
```

运行：`python smoke.py`

---

## 4. 验证结果（PASS 判据）

PASS 标准：三个 prompt 均续写出连贯文本，**无 `sm_120` / `ptxas` 报错**。

实际输出（2026-06-18）：

```
'Hello, my name is' -> 'John Smith. I am a student of computer science at the University of Minnesota...'
'The capital of France is' -> 'Paris.'
'The future of AI is' -> 'still uncertain, but it holds immense potential for improving human activities...'
```

→ 验证通过的链路：容器 + TRT-LLM 库 + **sm_120 kernel** + GPU 透传 + 模型加载 + 推理。

---

## 5. 踩坑记录（现象 → 根因 → 解法）


1. **WSL 默认用户丢失，退回 root**
   - 根因：`sudo tee /etc/wsl.conf`（无 `-a`）整体覆写，冲掉了 `[user] default=qwer` 段。迁移到 F 盘的发行版靠这行保存默认用户。
   - 解法：补回 `[user]\ndefault=qwer` + `wsl --shutdown` 重启。

2. **systemd 未运行**（`Cannot start unit with --now when systemd is not running`）
   - 根因：wsl.conf 缺 `[boot] systemd=true`，或改了没重启 WSL。
   - 解法：补 `systemd=true` + `wsl --shutdown`；临时兜底用 `sudo service docker start`。

3. **`nvidia-container-toolkit` apt 源拉不下来**（SSL / connection reset，IP 为 `2606:...`）
   - 根因：IPv6 链路被重置。
   - 解法：强制 IPv4（`curl -4` / `Acquire::ForceIPv4`）或走代理。

4. **`docker pull nvcr.io` 失败：EOF**
   - 根因：dockerd 是独立后台进程，**不继承 shell 代理**，直连 nvcr.io 被掐。
   - 解法：给 daemon 配代理（systemd drop-in `/etc/systemd/system/docker.service.d/proxy.conf` 写 `HTTP(S)_PROXY`）。

5. **blob 下载 EOF**（manifest 拿到了，下数据失败）
   - 根因：认证/manifest 在 `nvcr.io`，**镜像层在 `layers.nvcr.io`** 这个 CDN 域名，规则模式没放行它。
   - 解法：VPN 切**全局 / TUN 模式**，所有域名走代理。

6. **容器内下模型失败**（`LocalEntryNotFoundError`，但 `curl hf-mirror.com` 首页能 200）
   - 根因：hf-mirror 对具体权重文件返回 **HTTP 308 重定向回 `huggingface.co`**（连不上）。首页在镜像、权重文件被踢回源站。
   - 解法：**下载与推理解耦** —— 宿主机走 VPN 下权重、容器只读本地挂载，绕开 hf-mirror 的源站重定向。

7. **容器走代理全部失败**（`curl -x ... CONNECT aborted`，三个候选地址都不通）
   - 根因：VPN TUN 模式拦截容器虚拟流量；`198.18.0.2` 是 TUN 段、`host.docker.internal` 被拦、`192.168.1.1` 未监听 7890。
   - 解法：放弃容器代理，改用宿主机下载 + 挂载（同坑 6）。

8. **`MPI_ABORT` / 脚本"不动了"**
   - 根因：TRT-LLM 用 MPI 多进程 spawn worker，脚本缺 `if __name__ == '__main__'`，子进程重新导入时无限递归。
   - 解法：执行逻辑包进 `main()` + `if __name__ == "__main__":`。

---

## 6. 概括

> 在「国内网络 + WSL2 + Blackwell 消费卡」三重约束下，把**下载与推理解耦**：宿主机走 VPN 拉权重、容器只读本地挂载，绕开 hf-mirror 对大文件的源站重定向；并定位了 dockerd 代理不继承、`layers.nvcr.io` CDN 分流、MPI 多进程 idiom 等一系列真实部署坑。

---

## 7. 下一个 gate（待办）

- [x] **NVFP4 路径验证**（项目命门）：拉一个 `nvidia/` ModelOpt 预量化 NVFP4 checkpoint，确认 sm_120 上 NVFP4 推理可跑。 → **PASS，详见第 8 节**
- [x] vLLM 侧对照环境搭建（Blackwell + NVFP4）。 → **PASS，详见第 9 节**
- [x] benchmark 矩阵设计：精度(FP16/FP8/NVFP4) × 并发 × 框架，复用自研 TTFT/TPOT/ITL/QPS harness。 → **完成，详见第 10 节**

---

## 8. NVFP4 路径验证

> 验证日期：2026-06-18
> 目的：固化「sm_120 消费卡上 NVFP4 推理可跑」。FP16 冒烟（第 4 节）已证明 sm_120 普通 kernel 链路通，本节唯一新增的未知量是 **FP4 GEMM / FlashInfer 的 sm_120 kernel 在 rc15 容器里是否编进**。

### 8.1 checkpoint 选择

| 项 | 值 |
|---|---|
| 模型 | `nvidia/Llama-3.1-8B-Instruct-NVFP4`（别名 `…-FP4`，HF 上互通） |
| 选型理由 | 官方 `nvidia/` dense NVFP4 里最小的；8B@NVFP4 上盘 ~5GB，16GB 卡富余大 |
| 前置条件 | NVFP4 推理要求 Blackwell GPU + TRT-LLM v0.17+ → rc15 满足 |

> 其余官方 NVFP4（DeepSeek-R1 / Llama-3.3-70B / 405B / Llama-4-Scout MoE）均过大或为 MoE，不在本卡范围。
> **下载务必确认 `hf_quant_config.json` 在目录里** —— LLM API 靠它自动识别 NVFP4，解耦下载最易漏这种小配置文件，漏了会报「找不到量化配置」而非 kernel 错。

### 8.2 下载（复用第 2 节解耦链路，宿主机执行）

```bash
cd ~/trtllm-smoke
unset HF_ENDPOINT
export HF_HUB_DISABLE_XET=1
hf download nvidia/Llama-3.1-8B-Instruct-NVFP4 --local-dir ./models/llama31-8b-nvfp4
```

### 8.3 冒烟脚本 `nvfp4_smoke.py`（直接贴终端写入）

```bash
cat > ~/trtllm-smoke/nvfp4_smoke.py << 'EOF'
from tensorrt_llm import LLM, SamplingParams


def main():
    llm = LLM(
        model="/workspace/models/llama31-8b-nvfp4",
        backend="pytorch",
        attn_backend="FLASHINFER",
        tensor_parallel_size=1,
    )
    prompts = ["Hello, my name is", "The capital of France is", "The future of AI is"]
    params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=64)

    for out in llm.generate(prompts, params):
        print(f"{out.prompt!r} -> {out.outputs[0].text!r}")


if __name__ == "__main__":   # 同第 5 节坑 8，MPI spawn 必须
    main()
EOF
```

进容器（第 2 节那条 `docker run` 原样用）后：`python nvfp4_smoke.py`

### 8.4 PASS 判据

在 FP16（第 4 节）基础上加一条精度面硬约束：

1. 三个 prompt 续写连贯；
2. **无 `no kernel image is available` / cutlass FP4 / sm_120 报错**（本步真正在验的就是这个）；
3. **量化确认生效**（确认未偷偷退回 FP16）。判据不看 `nvidia-smi` 总量 —— pytorch backend 默认按 `free_gpu_memory_fraction≈0.9` 把剩余显存几乎全预占成 KV 池，总量会冲到 13–15GB，不能据此判断。改用两条硬证据：
   - **显存账反证**：8B 权重若为 FP16 = 16GB，已等于整卡显存，加上 KV 池必 OOM；能成功加载并留得下数 GB 给 KV，反推权重必为量化态。启动日志的 `Allocated X GiB for ... paged KV cache` 一行即可读出 KV 池占用。
   - **参数实证**：`cat models/llama31-8b-nvfp4/hf_quant_config.json` 看到 `"quant_algo": "NVFP4"`（权重文件自带声明）。或 `TLLM_LOG_LEVEL=INFO python nvfp4_smoke.py` 让 TRT-LLM echo 量化配置。

### 8.5 实际输出（2026-06-18）

```
'Hello, my name is' -> ' Bethany and I am a 24 year old freelance writer based in Los Angeles. I love to travel, hike, and try out new foods and drinks...'
'The capital of France is' -> ' Paris, a city known for its fashion, art, and romance. The Eiffel Tower is an iconic landmark...'
'The future of AI is' -> ' already here. Artificial intelligence (AI) is no longer a concept of the future. It’s already a reality that’s transforming industries...'
```

→ 三条全连贯、零 kernel 报错直接出文本。**结论：rc15 容器内 FP4 GEMM + FlashInfer 注意力两套 kernel 均含 sm_120**，5070Ti 上「容器 + sm_120 NVFP4 kernel + 模型加载 + 推理」整条链路验证通过。

**量化生效实锤（启动日志）**：日志报 `[MemUsageChange] Allocated 8.01 GiB for max tokens in paged KV cache`（后按上下文重算收缩到 6.35 GiB）。光 KV 池就 8GB，而 8B@FP16 权重本身就要 16GB —— 二者并存绝无可能塞进 16GB 卡。能成功加载并分出 8GB 给 KV，反证权重为量化态，PASS 判据第 3 条坐实。需明文「NVFP4」字样时 `cat hf_quant_config.json` 即可（rc15 默认日志不 echo 量化算法）。

**无害噪音（已知，不影响 PASS）**：日志中 `torchao / transformers 5.5.3 / nvidia-modelopt 版本错配`、`Error querying confidential compute state: Not Supported`、`storeContextBlocks: Can not find sequence` 均为 NV 周更 rc 容器的典型版本错配 / 环境探测告警，不影响加载与输出，无需追查。

### 8.6 sm_120 NVFP4 triage（备查，本次未触发）

后续换模型 / 换版本若崩，按概率排查：

- **FP4 GEMM kernel 没编进 sm_120** → `no kernel image…` 或 cutlass FP4 断言。处置：换更新 rc（rc18）或源码编译带 `TORCH_CUDA_ARCH_LIST=12.0`。这是唯一可能卡死路线的硬伤。
- **FlashInfer 注意力缺 sm_120** → 报错点在 attention。先去掉 `attn_backend="FLASHINFER"` 用默认后端，或试 `attn_backend="TRTLLM"`，隔离是 GEMM 还是 attention 问题。
- **KV cache OOM** → `kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.7)` 压预分配。
- **Llama 3.x 在 FP8/NVFP4 下 pipeline parallelism 异常**（官方已知坑）→ `export TRTLLM_LLAMA_EAGER_FUSION_DISABLED=1`。单卡 TP=1 不碰 PP，正常用不到，仅作 fusion 报错兜底。

### 8.7 概述

> 在 sm_120 Blackwell 消费卡上验证了 NVFP4 端到端推理可跑：确认 rc15 容器内 FP4 GEMM 与 FlashInfer 注意力 kernel 均含 sm_120 编译目标，并复用「宿主机下载 + 容器只读挂载」的解耦链路，绕开国内网络对预量化权重的拉取障碍 —— 为后续 trtllm-vs-vllm × 多精度 benchmark 钉死了 TRT-LLM 侧的 NVFP4 端点。

---

## 9. vLLM 侧 NVFP4 对照

> 验证日期：2026-06-18
> 目的：搭起 benchmark 的第二个端点 —— 同一份 NVFP4 权重在 vLLM 上跑通。
> 关键前提：**模型零重下**，直接复用第 8 节已下好的 `models/llama31-8b-nvfp4`（vLLM 直接吃 ModelOpt NVFP4 HF checkpoint）。本节唯一新增未知量：vLLM 在 sm_120 上是否走**原生** NVFP4 kernel。

### 9.1 核心陷阱：Marlin 静默回退

vLLM 与 TRT-LLM 不同 —— 若不认 sm_120 的 FP4，**不报错**，而是静默回退到 Marlin weight-only kernel，照样出正确文本，只打一行 warning。一旦回退，benchmark 就成了「TRT-LLM 原生 FP4 GEMM」对「vLLM Marlin 仿 FP4」，对照作废。**本节 PASS 的核心不是「能出文本」，是「确认走原生、未回退 Marlin」。** dense NVFP4 的 sm_120 原生 kernel 自 vLLM 0.13.0 起编入。

### 9.2 容器选择

| 项 | 值 |
|---|---|
| 首选镜像 | `nvcr.io/nvidia/vllm:26.01-py3`（同 nvcr.io，复用第 5 节 dockerd 代理 + TUN 全局链路；论坛有人就用它在 5070 Ti 跑通 NVFP4） |
| 备选镜像 | `vllm/vllm-openai:cu130-nightly`（走 docker.io，PyTorch kernel 按 SM 12.0 编，匹配 5070 Ti；注意 GB10 是 sm_121 不匹配） |
| 实测引擎版本 | vLLM `v0.13.0+faa43dbf.nv26.01`（≥0.13.0，含 dense sm_120 NVFP4 kernel） |

### 9.3 进容器（mirror 第 2 节，挂载同一目录）

```bash
docker run --rm -it --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  --gpus all \
  -v ~/trtllm-smoke:/workspace \
  -w /workspace \
  nvcr.io/nvidia/vllm:26.01-py3
```

### 9.4 冒烟脚本 `vllm_nvfp4_smoke.py`（直接贴终端写入）

同款 3 prompt / 同款采样参数，与 TRT-LLM 侧（第 8 节）严格对齐：

```bash
cat > ~/trtllm-smoke/vllm_nvfp4_smoke.py << 'EOF'
from vllm import LLM, SamplingParams


def main():
    llm = LLM(
        model="/workspace/models/llama31-8b-nvfp4",
        max_model_len=4096,
        gpu_memory_utilization=0.9,
    )
    prompts = ["Hello, my name is", "The capital of France is", "The future of AI is"]
    params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=64)

    for out in llm.generate(prompts, params):
        print(f"{out.prompt!r} -> {out.outputs[0].text!r}")


if __name__ == "__main__":
    main()
EOF
```

进容器后：`python vllm_nvfp4_smoke.py`。量化靠 `hf_quant_config.json` 自动识别（日志确认 `quantization=modelopt_fp4`），无需手动指定。

### 9.5 PASS 判据（核心是 native，不是「能出文本」）

1. 三个 prompt 续写连贯；
2. **走原生 FP4 GEMM**，日志主动打印：`[modelopt.py:982] Using flashinfer-cutlass for NVFP4 GEMM`；
3. **零回退**：全日志无 `does not have native support for FP4` / Marlin warning；
4. （可选主动核验）容器内 `from vllm._custom_ops import cutlass_scaled_mm_supports_fp4; print(cutlass_scaled_mm_supports_fp4(120))` → `True`。

### 9.6 实际输出（2026-06-18）

启动日志关键行：

```
[modelopt.py:858] Detected ModelOpt NVFP4 checkpoint.
EngineCore ... (v0.13.0+faa43dbf.nv26.01) ... quantization=modelopt_fp4
[modelopt.py:982] Using flashinfer-cutlass for NVFP4 GEMM          ← 原生 FP4，非 Marlin
[cuda.py:351] Using FLASH_ATTN attention backend
[gpu_model_runner.py:3659] Model loading took 5.6473 GiB memory     ← 权重实占，量化坐实
[gpu_worker.py] Available KV cache memory: 7.33 GiB / 60,064 tokens / 14.66x 并发
```

推理输出（temp=0.8，与 TRT-LLM 侧文本不同属采样随机性，非异常）：

```
'Hello, my name is' -> ' Helen and I am a devoted animal lover and owner of Angels of Mercy Pet Rescue...'
'The capital of France is' -> ' known for its fashion, food, art and romance. Paris is a city that has been loved by millions...'
'The future of AI is' -> ' in search and optimization, not in playing games like Go...'
```

→ **结论**：vLLM 0.13.0 在 sm_120 上走 **flashinfer-cutlass 原生 NVFP4 GEMM**（非 Marlin 回退），权重实占 5.65GB 量化坐实。benchmark 第二端点就位。

### 9.7 benchmark 方法学要点

- **两框架注意力后端不同属预期**：TRT-LLM 侧显式 `FLASHINFER`，vLLM 侧自动选 `FLASH_ATTN`（仅 FP4 GEMM 走 flashinfer-cutlass）。A/B 对照本就是「各框架最优默认路径」对打，不强行统一 kernel。
- **warmup 必须排除在测量外**：vLLM 启动含一次性 `torch.compile 44s + CUDA graph capture 7s`，TRT-LLM 侧亦有自身 graph/warmup。harness 测稳态前先打一轮 warmup 请求甩掉编译/捕获开销，否则首请求 TTFT 被污染成数十秒。
- **逐字比质量需固定随机性**：测吞吐/延迟无所谓采样随机；若要逐字比输出质量，两边统一 greedy / 固定 seed。

### 9.8 概述
> 搭起 trtllm-vs-vllm 的对照端点时，识别并规避了 vLLM 在 sm_120 上对 NVFP4 的 **Marlin 静默回退**陷阱：以日志中 `Using flashinfer-cutlass for NVFP4 GEMM` 主动确认走原生 FP4 tensor core 而非 weight-only 仿真，保证了跨框架 benchmark 的精度路径可比；并沉淀出「注意力后端差异需声明、warmup 须排除测量」的测量协议，确保对照公平性。

---

## 10. benchmark 矩阵：精度 × 并发 × 框架

> 验证日期：2026-06-21
> 模型：Llama-3.1-8B-Instruct（全 cell 同一模型，tokenizer 一致→ISL 自动受控）
> 端点：trtllm-serve / vllm serve，均 OpenAI 兼容 `/v1/completions` 流式 → harness 框架无关，仅换 base_url。

### 10.1 矩阵设计

| 框架 | 精度 | 并发 sweep | 状态 |
|---|---|---|---|
| TRT-LLM | NVFP4 | 1,2,4,8,16,32,40,48,56,64 | ✅ |
| TRT-LLM | FP8 | 1,2,4,8,16,32,48,64 | ✅ |
| vLLM | NVFP4 | 1,2,4,8,16,32,48,64 | ✅ |
| vLLM | FP8 | 1,2,4,8,16,32,48,64 | ✅ |
| 两框架 | FP16 | — | 解析判定**不可部署**：8B×2=16GB ＝ 整卡 16303 MiB，加 KV/激活必 OOM |

可分离两个干净对照：**固定精度看框架差**、**固定框架看精度差**。FP16 行作为「部署边界」—— 量化在这张卡上不是可选项而是前提，是整个研究的动机。

### 10.2 测量协议（harness 内建）

- **固定 ISL/OSL**：全 cell 同一 prompt 字符串（同模型→token 数恒定）+ `max_tokens=256` + `ignore_eos` 钉死 OSL。
- **greedy（temperature=0）**：确定性，质量公平 + 计时稳定。
- **warmup 排除**：每并发级先打一轮丢弃，甩掉 torch.compile / CUDA graph capture 一次性开销。
- **注意力后端差异声明**：TRT-LLM=FLASHINFER，vLLM=FLASH_ATTN（各框架最优默认，不强行统一）。
- **指标**：TTFT（首 token，prefill 代理）、TPOT（稳态每 token）、ITL（token 间隔）、QPS + 输出 tok/s。
- 每并发级 rounds=10（低并发压抖动，高并发同样适用）。

### 10.3 环境踩坑：FP8 的 FlashInfer JIT 编译 OOM（真实工程税）

NVFP4 的 sm_120 FP4 kernel 是容器**预编**（秒过），但 **FP8 的 sm_120 groupwise GEMM kernel（e4m3/e5m2）容器内未预编、首次启动现场 JIT**——一批 nvcc 任务 `ninja -j 12` 并行编译，每个 `cicc` 吃 2–3GB，峰值 24–36GB，**直接被 OOM killer 杀**（`nvcc error: cicc died due to signal 9`）。

- 根因：WSL 默认仅 15GiB 内存，扛不住 12 路并行 nvcc。
- 解法链：`.wslconfig` 内存 15→26GB（`wsl --shutdown` 生效）+ swap 24GB 兜峰值 → 26GB 裸跑过 12 路。
- 辅助手段（内存上不去时）：`--cpus 4` 砍 ninja 并行度、`FLASHINFER_JIT_NUM_WORKERS=1`/`NVCC_THREADS=1` 限并发。

> 工程判断：定位出「sm_120 上 FP4 预编、FP8 JIT」的容器路径差异 —— 这是国内消费卡 + 新架构组合的真实部署成本，FP8 的启动代价远高于 NVFP4。

### 10.4 四 cell × 全并发对照表（默认配置）

> trtllm_nvfp4 的 c40/48/56 为**默认（带 CUDA graph 捕获空洞）**表现，反映开箱行为；优化见 10.6。

```
框架_精度       c     QPS    tok/s   TTFTp50(ms)  TPOT(ms)  ITLp95(ms)
trtllm_nvfp4    1    0.193    49.5     41.5       20.13     21.08
trtllm_nvfp4    2    0.261    66.8     48.0       29.82     30.77
trtllm_nvfp4    4    1.932   494.5     48.1        7.92      8.32
trtllm_nvfp4    8    3.820   977.8     50.4        8.01      8.46
trtllm_nvfp4   16    7.386  1890.7     53.5        8.28      8.78
trtllm_nvfp4   32   13.954  3572.2     58.1        8.76      9.32
trtllm_nvfp4   40    6.898  1765.9     80.9       22.41     24.53   ← 捕获空洞
trtllm_nvfp4   48    7.614  1949.1     86.8       24.36     34.27   ← 捕获空洞
trtllm_nvfp4   56    9.707  2485.0     84.6       22.26     24.07   ← 捕获空洞
trtllm_nvfp4   64   25.208  6453.3     84.8        9.61     10.39
trtllm_fp8      1    0.204    52.2     43.5       19.06     19.75
trtllm_fp8      2    0.311    79.6     44.7       25.02     25.80
trtllm_fp8      4    1.454   372.2     44.3       10.61     11.09
trtllm_fp8      8    2.782   712.1     44.4       11.10     11.60
trtllm_fp8     16    5.341  1367.3     50.5       11.54     12.05
trtllm_fp8     32    7.851  2010.0     50.2       15.77     16.44
trtllm_fp8     48   10.239  2621.1     68.7       18.08     19.82
trtllm_fp8     64   13.338  3414.6     64.8       18.53     19.51
vllm_nvfp4      1    0.410   104.9     14.3        9.52     10.01
vllm_nvfp4      2    0.835   213.7     15.2        9.33      9.66
vllm_nvfp4      4    1.672   428.0     25.3        9.29      9.71
vllm_nvfp4      8    3.281   840.0     27.6        9.45      9.95
vllm_nvfp4     16    6.382  1633.8     33.5        9.70     10.24
vllm_nvfp4     32   11.857  3035.3     43.0       10.41     11.22
vllm_nvfp4     48   16.537  4233.5     55.2       11.16     12.19
vllm_nvfp4     64   20.590  5271.0     63.6       11.93     13.21
vllm_fp8        1    0.331    84.7     17.4       11.78     12.56
vllm_fp8        2    0.670   170.0     17.8       11.72     12.21
vllm_fp8        4    1.275   261.2     20.6       16.05     12.49
vllm_fp8        8    2.561   641.2     35.1       12.25     12.60
vllm_fp8       16    5.099  1298.3     42.6       12.19     12.83
vllm_fp8       32    8.419  2153.7     67.7       12.93     13.75
vllm_fp8       48   13.362  3420.6     78.6       13.78     14.91
vllm_fp8       64   16.675  4268.7     85.7       14.71     16.07
```

### 10.5 三条核心结论

1. **NVFP4 > FP8，两框架内均成立**：高并发吞吐 trtllm 3572 vs 2010 tok/s（c32，+78%）、vllm 5271 vs 4269（c64，+23%）；稳态 TPOT NVFP4 亦更低。decode 为 memory-bound，4bit 权重搬运更少，并发越高优势越显——项目主论点由数据支撑。
2. **框架取向差异（各有所长）**：vLLM **TTFT 全程更低**（c1：14ms vs trtllm 42ms），首 token 响应快、利于交互；TRT-LLM 命中 graph 捕获点时**稳态 TPOT 更低**、批吞吐更猛。选型取决于「延迟敏感」还是「吞吐优先」。
3. **可复现性**：v1/v2 两轮独立测量逐行吻合（QPS 偏差 <1%），测量协议稳定可信。

### 10.6 重点发现：CUDA graph 捕获空洞导致吞吐非单调（定位 + 解决闭环）

**现象**：trtllm_nvfp4 吞吐随并发**非单调**——c32 正常（3572 tok/s）→ c40/48/56 塌陷（~1700–2500，TPOT 翻 2.5 倍到 ~22ms）→ c64 满血反弹（6453，TPOT 9.6）。vLLM 同条件曲线平滑单调。

**逐一证伪三个假说**：
- ❌ KV 抢占：KV 池 4102 blocks × 32 = 131k tokens，每请求 ~450 tokens → 并发上限 ~290，c40 远未触及。
- ❌ max_batch_size：启动日志 `max_batch_size=2048`，远不卡。
- ❌ max_num_tokens：控制变量实验，8192→16384 翻倍，凹陷带逐点不变（c40 1766→1739 等）→ **实验证伪**。

**定位**：启动日志 CUDA graph 捕获列表为 `128, 64, 32, 31, 30, …, 16`——**32 与 64 之间（33–63）是捕获空洞**。c40/48/56 落入空洞 → 回退 eager 逐 op 执行 → TPOT 翻倍；c64 命中捕获点 → 满血。

**验证 + 解决**：`--extra_llm_api_options` 传 `cuda_graph_config: {enable_padding: true}`，把非捕获 size 向上 pad 到最近 graph：

| c | 默认（空洞） | +enable_padding | 提升 |
|---|---|---|---|
| 40 | 1766 tok/s, TPOT 22.4 | **4359, TPOT 8.95** | 2.5× |
| 48 | 1949, TPOT 24.4 | **4681, TPOT 10.0** | 2.4× |
| 56 | 2485, TPOT 22.3 | **5851, TPOT 9.3** | 2.4× |

凹陷带消失、曲线恢复单调，**根因 100% 实锤**。vLLM 因捕获列表更密 + 默认 padding 故开箱即平滑。

> 配置写法：`cat > pad.yaml << 'EOF'` 内容 `cuda_graph_config:\n  enable_padding: true`，起服务加 `--extra_llm_api_options /workspace/pad.yaml`（rc15 无 `--cuda_graph_config` 命令行 flag）。

### 10.7 项目概述

> 在 RTX 5070Ti（sm_120, 16GB）上完成 TRT-LLM vs vLLM 的跨框架 × 跨精度（NVFP4/FP8）推理 benchmark，自建 OpenAI 兼容流式 harness 测 TTFT/TPOT/ITL/QPS。定量确立「NVFP4 高并发吞吐优于 FP8、量化使 8B 在 16GB 卡上从不可部署变为可部署」；并定位一个 TRT-LLM 吞吐随并发**非单调**的反直觉现象——通过算 KV 容量、控制变量翻倍 max_num_tokens、查启动日志三步**连续证伪三个假说**，最终锁定 **CUDA graph 捕获空洞（c33–63 未捕获→回退 eager）**，以 `enable_padding` 实测 2.5× 修复。展示了「假说—证伪—控制变量验证」的完整性能 debug 方法论。

