#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""vllm_step_timeline.py — 引擎步骤级 prefill/decode 时间轴（绕开 CUPTI 限制）

背景：本机 A10-24Q 为 vGPU，CUPTI 内核级采集被禁（nsys 只出 NVTX 无 kernel）。
方案：hook vLLM V0 的 Scheduler.schedule / LLMEngine.step，用 torch.cuda.Event
      （纯驱动 API，vGPU 可用）测每步 GPU 耗时，用 perf_counter 测步间隔，
      用 SchedulerOutputs.num_prefill_groups 区分 prefill/decode 步。
"""
import os, time, json, statistics

os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
os.environ.setdefault("VLLM_USE_V1", "0")

import torch
from vllm import LLM, SamplingParams
from vllm.engine.llm_engine import LLMEngine
from vllm.core.scheduler import Scheduler

MODEL = "/home/ctyun/.cache/TeleAgent/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
TEMP = os.path.dirname(os.path.abspath(__file__))

# ---------- hooks ----------
_orig_schedule = Scheduler.schedule
def _schedule_patched(self, *a, **k):
    out = _orig_schedule(self, *a, **k)
    self._last_sched = out[1] if isinstance(out, tuple) else out
    return out
Scheduler.schedule = _schedule_patched

RECORDS = []  # 每个引擎步一条

_orig_step = LLMEngine.step
def _step_patched(self, *a, **k):
    t0 = time.perf_counter()
    ev0 = torch.cuda.Event(enable_timing=True)
    ev1 = torch.cuda.Event(enable_timing=True)
    ev0.record()
    out = _orig_step(self, *a, **k)
    ev1.record()
    t1 = time.perf_counter()
    sched = getattr(self.scheduler[0], "_last_sched", None)
    RECORDS.append({
        "t0": t0,
        "wall_ms": (t1 - t0) * 1e3,
        "phase": ("prefill" if sched.num_prefill_groups > 0 else "decode") if sched else "?",
        "batched_tokens": sched.num_batched_tokens if sched else -1,
        "_ev0": ev0, "_ev1": ev1,
    })
    return out
LLMEngine.step = _step_patched

# ---------- bench ----------
llm = LLM(model=MODEL, gpu_memory_utilization=0.25, max_model_len=4096,
          seed=0, disable_log_stats=True, dtype="bfloat16")
tok = llm.get_tokenizer()
base = ("高性能推理系统的核心矛盾在于算子执行效率与调度开销的平衡，"
        "工程师需要在内核启动延迟、内存搬运带宽与计算密度之间不断权衡取舍。")
prompt = "请基于以下技术论述续写并深入展开：\n" + base * 30

runs = []  # 每次 generate 的记录切片

def gen(text, max_tokens, tag):
    sp = SamplingParams(max_tokens=max_tokens, temperature=0.0, ignore_eos=True, seed=0)
    i0 = len(RECORDS)
    t_wall = time.perf_counter()
    llm.generate([text], sp)
    torch.cuda.synchronize()
    i1 = len(RECORDS)
    runs.append({"tag": tag, "i0": i0, "i1": i1, "wall_ms": (time.perf_counter() - t_wall) * 1e3,
                 "prompt_tokens": len(tok.encode(text)), "max_tokens": max_tokens})
    print(f"[bench] {tag}: steps={i1-i0} wall={runs[-1]['wall_ms']:.2f}ms", flush=True)

gen(prompt + "（热身）", 4, "warmup")
time.sleep(0.3)
gen(prompt + "（测量甲）", 1, "prefill_only")
time.sleep(0.3)
gen(prompt + "（测量乙）", 129, "prefill_decode")

# 回填 GPU 事件耗时
for r in RECORDS:
    r["gpu_ms"] = r["_ev0"].elapsed_time(r["_ev1"])
    del r["_ev0"], r["_ev1"]

t_base = RECORDS[0]["t0"]
for r in RECORDS:
    r["start_ms"] = (r["t0"] - t_base) * 1e3
    del r["t0"]

# ---------- 统计 ----------
def seg_stats(recs):
    pre = [r for r in recs if r["phase"] == "prefill"]
    dec = [r for r in recs if r["phase"] == "decode"]
    out = {"n_steps": len(recs), "n_prefill": len(pre), "n_decode": len(dec)}
    if pre:
        out["prefill"] = {k: round(pre[0][k], 3) for k in ("gpu_ms", "wall_ms", "batched_tokens")}
    if len(dec) >= 2:
        iv = [dec[i + 1]["start_ms"] - dec[i]["start_ms"] for i in range(len(dec) - 1)]
        out["decode"] = {
            "gpu_ms_median": round(statistics.median([r["gpu_ms"] for r in dec]), 3),
            "gpu_ms_max": round(max(r["gpu_ms"] for r in dec), 3),
            "interval_ms_median": round(statistics.median(iv), 3),
            "busy_pct_median": round(statistics.median(
                [a / b * 100 for a, b in zip([r["gpu_ms"] for r in dec], iv + [iv[-1]]) if b > 0]), 1),
        }
    return out

summary = {"runs": runs, "per_run": {}}
for run in runs:
    recs = RECORDS[run["i0"]:run["i1"]]
    summary["per_run"][run["tag"]] = seg_stats(recs)

with open(os.path.join(TEMP, "step_records.json"), "w", encoding="utf-8") as f:
    json.dump({"summary": summary, "records": RECORDS}, f, ensure_ascii=False)
print("[bench] SUMMARY " + json.dumps(summary["per_run"], ensure_ascii=False, indent=1), flush=True)

# ---------- 作图 ----------
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager, pyplot as plt
for fp in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]:
    if os.path.exists(fp):
        font_manager.fontManager.addfont(fp); break
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
C = {"warmup": "#9aa4b2", "prefill_only": "#2563eb", "prefill_decode": "#d97706"}

# 图A：步骤级时间轴全景
fig, ax = plt.subplots(figsize=(10, 4.4), dpi=110)
bounds = {}
for run in runs:
    recs = RECORDS[run["i0"]:run["i1"]]
    if not recs: continue
    bounds[run["tag"]] = (recs[0]["start_ms"], recs[-1]["start_ms"])
    for r in recs:
        col = "#2563eb" if r["phase"] == "prefill" else C[run["tag"] if run["tag"] != "prefill_only" else "prefill_only"]
        ax.vlines(r["start_ms"], 0, r["gpu_ms"], color=col, lw=1.4, alpha=0.9,
                  label="_")
    # 窗口底色
    ax.axvspan(bounds[run["tag"]][0] - 5, bounds[run["tag"]][1] + 15,
               color=C[run["tag"]], alpha=0.06)
import matplotlib.lines as mlines
ax.legend(handles=[
    mlines.Line2D([], [], color="#2563eb", lw=2, label="prefill 步（GPU 大步骤）"),
    mlines.Line2D([], [], color="#d97706", lw=2, label="decode 步（每步一个小 CUDA Graph 回放）"),
    mlines.Line2D([], [], color="#9aa4b2", lw=2, label="warmup"),
], loc="upper right", fontsize=9)
ax.set_xlabel("时间 (ms，相对首个引擎步)")
ax.set_ylabel("单步 GPU 耗时 (ms)")
ax.set_title("vLLM 引擎步骤时间轴：prefill 一次性大步 vs decode 周期性小步（Qwen2.5-0.5B @ A10 vGPU, bf16, cudagraph on）")
fig.tight_layout(); fig.savefig(os.path.join(TEMP, "figA_engine_step_timeline.png")); plt.close(fig)

# 图B：decode 步解剖（测量乙）
run3 = [r for r in runs if r["tag"] == "prefill_decode"][0]
recs = RECORDS[run3["i0"]:run3["i1"]]
dec = [r for r in recs if r["phase"] == "decode"]
if len(dec) > 4:
    fig, ax = plt.subplots(figsize=(10, 4.0), dpi=110)
    xs = list(range(len(dec)))
    ax.bar(xs, [r["gpu_ms"] for r in dec], color="#d97706", alpha=0.85, label="decode 步 GPU 忙时")
    intervals = [dec[i + 1]["start_ms"] - dec[i]["start_ms"] for i in range(len(dec) - 1)] + [float("nan")]
    ax.plot(xs, intervals, "-o", color="#1f2d3d", ms=2.2, lw=1, label="步间隔（含调度/拷贝间隙）")
    gm = statistics.median([r["gpu_ms"] for r in dec])
    im = statistics.median([x for x in intervals[:-1]])
    ax.set_xlabel("decode 步序号 (0..{})".format(len(dec) - 1))
    ax.set_ylabel("耗时 (ms)")
    ax.set_title(f"Decode 阶段解剖：每步 GPU 忙时中位数 {gm:.2f} ms / 步间隔中位数 {im:.2f} ms → 单流 GPU 忙占比 ~{gm/im*100:.0f}%")
    ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(TEMP, "figB_decode_anatomy.png")); plt.close(fig)

print("[done] figures + step_records.json", flush=True)
