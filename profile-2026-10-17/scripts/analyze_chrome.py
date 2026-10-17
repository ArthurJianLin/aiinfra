#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""analyze_chrome.py — 解析 torch profiler chrome trace，切分 prefill/decode 内核事件，出图+统计

用法:
  python analyze_chrome.py [trace 文件 (.json 或 .json.gz)]
  默认读 ../results/vllm_chrome.json.gz（相对本脚本），图输出到本目录。
"""
import os, sys, json, statistics, glob, gzip

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager, pyplot as plt

TEMP = os.path.dirname(os.path.abspath(__file__))
TRACE = sys.argv[1] if len(sys.argv) > 1 else os.path.join(TEMP, "..", "results", "vllm_chrome.json.gz")

for fp in glob.glob("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"):
    if os.path.exists(fp):
        font_manager.fontManager.addfont(fp); break
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ev = json.load(gzip.open(TRACE, "rt") if TRACE.endswith(".gz") else open(TRACE))["traceEvents"]
print(f"[load] {len(ev)} events")

# 相位窗口（record_function → user_annotation）
windows = {}
for e in ev:
    if e.get("ph") == "X" and str(e.get("name", "")).startswith("===phase:") and "dur" in e:
        tag = e["name"].strip("=<>").replace("phase:", "")
        windows.setdefault(tag, []).append((e["ts"], e["ts"] + e["dur"]))
print("[windows]", {k: [f"{b-a:.0f}us" for a, b in v] for k, v in windows.items()})
assert windows.get("prefill_only") and windows.get("prefill_decode"), "缺相位窗口"

kern = [(e["ts"], e["ts"] + e["dur"], e["name"], e["dur"]) for e in ev
        if e.get("ph") == "X" and e.get("cat") == "kernel"]
memcpy = [e for e in ev if e.get("ph") == "X" and e.get("cat") in ("gpu_memcpy", "gpu_memset")]
print(f"[kernels] {len(kern)} kernel / {len(memcpy)} memcpy events")

def in_win(w):
    a, b = w[0]
    return [k for k in kern if a - 50 <= k[0] < b + 2000]

kw = in_win(windows["warmup"])
k1 = in_win(windows["prefill_only"])
k2 = in_win(windows["prefill_decode"])
K = len(k1)
pre2, decode = k2[:K], k2[K:]
print(f"[phase] warmup={len(kw)} prefill={len(k1)} prefill(repeat)={len(pre2)} decode={len(decode)}")

def stats(ks):
    if not ks:
        return None
    span = (ks[-1][1] - ks[0][0]) / 1e3
    busy = sum(k[3] for k in ks) / 1e3
    agg = {}
    for k in ks:
        d, c = agg.get(k[2], (0, 0))
        agg[k[2]] = (d + k[3], c + 1)
    top = sorted(agg.items(), key=lambda kv: -kv[1][0])[:8]
    return {"n_kernels": len(ks), "span_ms": round(span, 3), "busy_ms": round(busy, 3),
            "busy_pct": round(busy / span * 100, 1) if span else 0,
            "top_kernels": [{"name": n[:80], "total_us": round(d, 1), "count": c} for n, (d, c) in top]}

def split_steps(ks, gap_us=400):
    steps, cur = [], [ks[0]]
    for k in ks[1:]:
        if k[0] - cur[-1][1] > gap_us:
            steps.append(cur); cur = [k]
        else:
            cur.append(k)
    if cur:
        steps.append(cur)
    return steps

steps = split_steps(decode) if decode else []
iv = [ (steps[i][0][0] - steps[i-1][0][0])/1e3 for i in range(1, len(steps)) ]

summary = {
    "prefill": stats(k1),
    "decode": stats(decode),
    "decode_steps": {
        "n_detected": len(steps),
        "kernels_per_step_median": int(statistics.median([len(s) for s in steps])) if steps else 0,
        "median_step_interval_ms": round(statistics.median(iv), 3) if iv else None,
        "per_step_kernel_busy_us": round(statistics.median([sum(k[3] for k in s) for s in steps]), 1) if steps else 0,
    },
    "n_memcpy_events": len(memcpy),
}
print("[summary]", json.dumps(summary, ensure_ascii=False, indent=1)[:1200])

t0 = (kw or k1)[0][0]
C = {"warmup": "#9aa4b2", "prefill": "#2563eb", "decode": "#d97706"}

# 图1 全景
fig, ax = plt.subplots(figsize=(10, 4.4), dpi=110)
for ks_, tag in [(kw, "warmup"), (k1, "prefill_only"), (decode, "decode")]:
    if not ks_:
        continue
    col = C["prefill"] if tag == "prefill_only" else C[tag]
    ax.vlines([(k[0] - t0)/1e3 for k in ks_], 0, [k[3] for k in ks_],
              color=col, lw=0.8, alpha=0.85,
              label={"warmup": "warmup 窗口内核", "prefill_only": "prefill 阶段内核（大而密）",
                     "decode": "decode 阶段内核（小步节拍）"}[tag])
for tag, w in windows.items():
    if tag in ("prefill_only", "prefill_decode"):
        ax.axvspan((w[0][0]-t0)/1e3, (w[0][1]-t0)/1e3, color=C["prefill" if tag=="prefill_only" else "decode"], alpha=0.06)
ax.set_xlabel("时间 (ms，相对 warmup 起点)")
ax.set_ylabel("单个内核耗时 (μs)")
ax.set_title("vLLM 推理内核级时间轴全景：prefill 密集大内核 vs decode 周期性小步（torch profiler / CUPTI 12.1）")
ax.legend(loc="upper right", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(TEMP, "figC_kernel_timeline.png")); plt.close(fig)

# 图2 prefill 放大
if k1:
    s0 = k1[0][0]
    fig, ax = plt.subplots(figsize=(10, 3.8), dpi=110)
    ax.vlines([(k[0]-s0)/1e3 for k in k1], 0, [k[3] for k in k1], color=C["prefill"], lw=1.0)
    st = summary["prefill"]
    ax.set_xlabel("时间 (ms，相对 prefill 起点)")
    ax.set_ylabel("单个内核耗时 (μs)")
    ax.set_title(f"Prefill 放大（1181 tokens，一次前向）：{st['n_kernels']} 个内核，span {st['span_ms']} ms，GPU 忙 {st['busy_pct']}%")
    fig.tight_layout(); fig.savefig(os.path.join(TEMP, "figD_prefill_zoom.png")); plt.close(fig)

# 图3 decode 放大（10 步）
if steps:
    mid = steps[len(steps)//2: len(steps)//2+10]
    flat = [k for s in mid for k in s]
    s0 = flat[0][0]
    fig, ax = plt.subplots(figsize=(10, 3.8), dpi=110)
    ax.vlines([(k[0]-s0)/1e3 for k in flat], 0, [k[3] for k in flat], color=C["decode"], lw=1.2)
    ax.set_xlabel("时间 (ms，相对所选 decode 起点步)")
    ax.set_ylabel("单个内核耗时 (μs)")
    ds = summary["decode_steps"]
    ax.set_title(f"Decode 放大：连续 10 步，每步中位 {ds['kernels_per_step_median']} 个内核（CUDA Graph 回放），步中位间隔 {ds['median_step_interval_ms']} ms")
    fig.tight_layout(); fig.savefig(os.path.join(TEMP, "figE_decode_zoom.png")); plt.close(fig)

with open(os.path.join(TEMP, "analyze_chrome_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print("[done]")
