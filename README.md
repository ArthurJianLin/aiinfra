# AI Infra 实验

个人 AI Infra 学习主线（L5 推理优化 + L2/L3 系统底座）的实验资产目录：vLLM 部署运维 + GPU 推理 profiling。所有实验在真实 vGPU 环境上完成并验证。

## 目录内容

| 路径 | 说明 |
|---|---|
| `vLLM部署环境与运维手册.html` | vLLM 0.7.3 部署全过程：版本匹配矩阵（驱动/vllm/torch/transformers）、7 个真实故障的根因与解法、启动/验证命令、故障速查表 |
| `vLLM推理Profile报告.html` | prefill/decode 两阶段时间轴分析：5 张图、三层采集手段交叉验证、6 个 profiling 问题攻坚、300 字观察 |
| `profile-2026-10-17/` | profile 实验全部产物（脚本/图/原始数据），结构见下 |

## profile-2026-10-17 目录结构

```
profile-2026-10-17/
├── scripts/    # 复现脚本 ×7
│   ├── vllm_profile_bench.py     # nsys 采集的基准（NVTX 相位标记 + 壁钟计时）
│   ├── vllm_torchprof_bench.py   # torch profiler 内核级采集（chrome trace 导出）
│   ├── vllm_step_timeline.py     # 引擎步级时间轴（hook Scheduler/Engine + CUDA 事件）
│   ├── analyze_chrome.py         # chrome trace 解析出图（支持 .json / .json.gz 输入）
│   ├── analyze_trace.py          # nsys sqlite 解析出图（供非 vGPU 机器内核级采集成功时使用）
│   ├── mini_cuda.py              # 最小 CUDA 复现（验证 nsys 是否采到 kernel）
│   └── mini_torch_prof.py        # torch profiler 可用性验证
├── figures/    # 时间轴图 ×5（与报告图 1–5 一致）
│   ├── figA_engine_step_timeline.png   # 引擎步全景：prefill 大脉冲 vs decode 小节拍
│   ├── figB_decode_anatomy.png         # decode 步解剖：忙时 vs 间隔
│   ├── figC_kernel_timeline.png        # 内核级全景（torch profiler）
│   ├── figD_prefill_zoom.png           # prefill 放大：529 大内核
│   └── figE_decode_zoom.png            # decode 放大：每步 278 小内核
└── results/    # 原始数据
    ├── profile_result.json             # 基准计时结果
    ├── step_records.json               # 137 个引擎步记录
    ├── analyze_chrome_summary.json     # 内核分相统计（Top kernel 等）
    ├── vllm_chrome.json.gz             # chrome trace（46MB 压至 2.1MB，解压后可拖 perfetto）
    ├── vllm_prefill_decode.nsys-rep    # nsys 报告（NVTX 有效 / 无 CUDA 数据，兼容性实证）
    ├── vllm_prefill_decode.sqlite      # 上者的 sqlite 导出
    └── mini_cuda.*                     # vGPU CUPTI 不兼容的最小复现证据
```

## 实验环境

| 组件 | 版本 | 约束 |
|---|---|---|
| GPU | NVIDIA A10-24Q 24GB | **vGPU 规格**（客户机视角） |
| GPU 驱动 | 535.183.06 | **锁定不可升级**，CUDA 12.2 上限 |
| vllm | 0.7.3 | 0.8+ 需 torch 2.6 / 新 CUPTI，均被驱动挡住 |
| torch | 2.5.1+cu121（自带 CUPTI 12.1） | 与 r535 驱动匹配 |
| nsys | 2026.5.1（自带 CUPTI 12.9） | 与 r535 不兼容，仅 NVTX 可用 |

## 关键兼容性结论（vGPU / 老驱动环境必读）

1. **nsys 的 CUPTI 版本必须 ≤ 驱动 CUDA 版本**：nsys 2026.5 捆绑 CUPTI 12.9，初始化时启用 CUDA 12.4+ 的 `cuGreenCtxDestroy` 驱动回调，在 r535（CUDA 12.2）上返回 `INVALID_PARAMETER` → CUDA 注入静默失败，采集到 0 个 kernel 事件（诊断仅 warning 级）。需要 nsys 内核级采集须用捆绑 CUPTI ≤ 12.2 的 2024.x 之前版本，或升级驱动至 r550+（vGPU 环境需平台管理员操作宿主机）。
2. **torch profiler 是老驱动上的最佳路径**：torch 2.5.1 自带 CUPTI 12.1 匹配 r535，内核级数据完整可采。
3. **perf_event_paranoid=4**：CPU 采样类剖析（nsys CPU backtrace、perf）全部不可用，无 sudo 无法调整。
4. vGPU 卡（如 A10-24Q）驱动升级由宿主机平台决定，客户机无法自行更换。

## 复现步骤

```bash
# 0) 环境：~/vllm-env（vllm 0.7.3 + torch 2.5.1+cu121），模型 Qwen2.5-0.5B-Instruct 本地缓存
source ~/vllm-env/bin/activate

# 1) 安装 nsys（无 sudo，装到家目录；桌面机必须加 --nox11）
curl -sL -o nsight-systems.run "https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2026_5/nsightsystems-linux-public-2026.5.1.161-3889610.run"
bash nsight-systems.run --noexec --noprogress --nox11 --target ~/nsight-systems-2026.5.1

# 2) nsys 采集（本 vGPU 机输出 NVTX 层，兼容性现象复现用）
~/nsight-systems-2026.5.1/pkg/bin/nsys profile -t cuda,nvtx --force-overwrite=true \
  -o vllm_prefill_decode python scripts/vllm_profile_bench.py

# 3) 内核级采集（torch profiler，本机可行路径）+ 引擎步级时间轴
python scripts/vllm_torchprof_bench.py    # → vllm_chrome.json
gzip -9 vllm_chrome.json                  # → vllm_chrome.json.gz
python scripts/vllm_step_timeline.py      # → step_records.json + figA/figB

# 4) 解析出图（默认读 ../results/vllm_chrome.json.gz）
cd scripts && python analyze_chrome.py    # → figC/figD/figE
```

数字参考（1184-token prompt，bf16，cudagraph on）：prefill 单步 19.1 ms / 529 kernel；decode 每步中位 3.61 ms / 278 kernel，单流 ~270 tok/s。

