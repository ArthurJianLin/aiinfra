#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""analyze_trace.py — 解析 nsys sqlite 导出，切分 prefill/decode，出图 + 汇总 JSON"""
import os, sys, json, glob, sqlite3, statistics, subprocess

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib import font_manager

TEMP = os.path.dirname(os.path.abspath(__file__))
SQLITE = sys.argv[1] if len(sys.argv) > 1 else os.path.join(TEMP, "vllm_prefill_decode.sqlite")
SUMMARY_OUT = os.path.join(TEMP, "analyze_summary.json")

# ---------- CJK 字体 ----------
cjk_candidates = []
for pat in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK*.ttc",
            "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
            "/usr/share/fonts/**/*CJK*.tt[cf]"]:
    cjk_candidates += glob.glob(pat, recursive=True)
cjk_candidates = sorted(set(cjk_candidates),
                        key=lambda p: (0 if "SansCJK-Regular" in p else 1 if "SansCJK" in p else 2))
font_name = None
for fp in cjk_candidates:
    try:
        font_manager.fontManager.addfont(fp)
        font_name = font_manager.FontProperties(fname=fp).get_name()
        print(f"[font] using {fp} -> {font_name}")
        break
    except Exception as e:
        print(f"[font] skip {fp}: {e}")
plt.rcParams["font.sans-serif"] = ([font_name] if font_name else []) + ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

con = sqlite3.connect(SQLITE)
cur = con.cursor()

def sid(x):
    """StringIds 数字 → 文本"""
    if x is None:
        return "?"
    r = cur.execute("SELECT value FROM StringIds WHERE id=?", (x,)).fetchone()
    return r[0] if r else str(x)

# ---------- NVTX 窗口 ----------
rows = cur.execute("SELECT start, end, text FROM NVTX_EVENTS ORDER BY start").fetchall()
windows = {}
for s, e, t in rows:
    txt = sid(t) if isinstance(t, int) else (t or "?")
    if txt in ("warmup", "prefill_only", "prefill_decode") and e is not None:
        windows[txt] = (s, e)
print("[nvtx] windows:", {k: (v[1]-v[0])/1e6 for k, v in windows.items()}, "ms")
assert "prefill_only" in windows and "prefill_decode" in windows, "NVTX 窗口缺失"

# ---------- 内核事件 ----------
ks = cur.execute(
    "SELECT k.start, k.end, k.shortName FROM CUPTI_ACTIVITY_KIND_KERNEL k ORDER BY k.start"
).fetchall()
kernels = [(s, e, sid(n)) for s, e, n in ks]
print(f"[kernels] total {len(kernels)}")

def bucket(w0, w1, margin_ns=5_000_000):
    return [(s, e, n) for s, e, n in kernels if w0 - margin_ns <= s < w1 + margin_ns]

kw = bucket(*windows.get("warmup", (windows["prefill_only"][0]-1, windows["prefill_only"][0])))
k1 = bucket(*windows["prefill_only"])            # prefill + 1 decode step
k2 = bucket(*windows["prefill_decode"])          # prefill + 129 decode steps
K = len(k1)
pre2, decode = k2[:K], k2[K:]
print(f"[phase] warmup={len(kw)} prefill1={len(k1)} prefill2={len(pre2)} decode={len(decode)}")

def phase_stats(ks, name):
    if not ks:
        return None
    span = (ks[-1][1] - ks[0][0]) / 1e6
    busy = sum((e - s) for s, e, _ in ks) / 1e6
    agg = {}
    for s, e, n in ks:
        d, c = agg.get(n, (0, 0))
        agg[n] = (d + (e - s) / 1e3, c + 1)
    top = sorted(agg.items(), key=lambda kv: -kv[1][0])[:8]
    return {
        "n_kernels": len(ks),
        "span_ms": round(span, 3),
        "busy_ms": round(busy, 3),
        "busy_pct": round(busy / span * 100, 1) if span else 0,
        "top_kernels": [{"name": n[:72], "total_us": round(d, 1), "count": c} for n, (d, c) in top],
    }

# decode 步进切分：相邻内核间隙 > 800us 认为跨步
def split_steps(ks, gap_ns=800_000):
    steps, cur = [], [ks[0]]
    for k in ks[1:]:
        if k[0] - cur[-1][1] > gap_ns:
            steps.append(cur); cur = [k]
        else:
            cur.append(k)
    steps.append(cur)
    return steps

decode_steps = split_steps(decode) if decode else []
step_intervals = [(decode_steps[i][0][0] - decode_steps[i-1][0][0])/1e6 for i in range(1, len(decode_steps))]

stats = {
    "warmup": phase_stats(kw, "warmup"),
    "prefill": phase_stats(k1, "prefill"),
    "decode": phase_stats(decode, "decode"),
    "decode_steps": {
        "n_detected": len(decode_steps) if decode else 0,
        "median_step_interval_ms": round(statistics.median(step_intervals), 3) if step_intervals else None,
        "kernels_per_step_median": int(statistics.median([len(x) for x in decode_steps])) if decode_steps else 0,
        "per_step_busy_us_median": round(statistics.median([sum((e-s) for s,e,_ in st)/1e3 for st in decode_steps]), 1) if decode_steps else 0,
    },
}
print("[stats]", json.dumps(stats, ensure_ascii=False, indent=1)[:1500])

# ---------- 作图 ----------
C = {"warmup": "#9aa4b2", "prefill": "#2563eb", "decode": "#d97706"}
t0 = min(kw[0][0] if kw else windows["prefill_only"][0], windows["prefill_only"][0])

# 图1 全景
fig, ax = plt.subplots(figsize=(10, 4.2), dpi=110)
for ks_, tag in [(kw, "warmup"), (k1, "prefill_only"), (decode, "decode")]:
    if not ks_:
        continue
    label = "prefill（含1步decode）" if tag == "prefill_only" else ("warmup" if tag == "warmup" else "decode（128步）")
    col = C["prefill"] if tag == "prefill_only" else C[tag]
    ax.vlines([(s - t0)/1e6 for s, e, _ in ks_], 0, [(e - s)/1e3 for s, e, _ in ks_],
              color=col, lw=0.8, alpha=0.8, label=label)
for wname, (a, b) in windows.items():
    if wname in ("prefill_only", "prefill_decode"):
        ax.axvspan((a-t0)/1e6, (b-t0)/1e6, color="#3b82f6" if wname=="prefill_only" else "#d97706",
                   alpha=0.07)
ax.set_xlabel("时间 (ms，相对 prefill_only 窗口起点)")
ax.set_ylabel("内核耗时 (μs)")
ax.set_title("vLLM 推理 prefill / decode 两阶段全景时间轴（Qwen2.5-0.5B @ A10, bf16）")
ax.legend(loc="upper right", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(TEMP, "fig1_timeline.png")); plt.close(fig)

# 图2 prefill 放大（window1）
if k1:
    s0 = k1[0][0]
    fig, ax = plt.subplots(figsize=(10, 3.8), dpi=110)
    ax.vlines([(s - s0)/1e6 for s, e, _ in k1], 0, [(e - s)/1e3 for s, e, _ in k1],
              color=C["prefill"], lw=1.0)
    ax.set_xlabel("时间 (ms，相对窗口起点)")
    ax.set_ylabel("内核耗时 (μs)")
    st = stats["prefill"]
    ax.set_title(f"Prefill 放大：{st['n_kernels']} 个内核一次性密集执行（span {st['span_ms']} ms，GPU 忙占比 {st['busy_pct']}%）")
    fig.tight_layout(); fig.savefig(os.path.join(TEMP, "fig2_prefill.png")); plt.close(fig)

# 图3 decode 放大（取 10 个连续步）
if decode_steps:
    mid = decode_steps[len(decode_steps)//2 : len(decode_steps)//2 + 10]
    flat = [k for st_ in mid for k in st_]
    s0 = flat[0][0]
    fig, ax = plt.subplots(figsize=(10, 3.8), dpi=110)
    ax.vlines([(s - s0)/1e6 for s, e, _ in flat], 0, [(e - s)/1e3 for s, e, _ in flat],
              color=C["decode"], lw=1.2)
    for i, st_ in enumerate(mid):
        ax.axvline((st_[0][0]-s0)/1e6, color="#b45309", ls="--", lw=0.7, alpha=0.6)
        ax.text((st_[0][0]-s0)/1e6+0.02, ax.get_ylim()[1]*0.86, f"步{i}", fontsize=7, color="#b45309")
    ds = stats["decode_steps"]
    ax.set_xlabel("时间 (ms，相对首个所选步起点)")
    ax.set_ylabel("内核耗时 (μs)")
    ax.set_title(f"Decode 放大：连续 10 步，每步一撮小内核 + 步间调度间隙（步中位间隔 {ds['median_step_interval_ms']} ms）")
    fig.tight_layout(); fig.savefig(os.path.join(TEMP, "fig3_decode.png")); plt.close(fig)

with open(SUMMARY_OUT, "w", encoding="utf-8") as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)
print("[done] summary ->", SUMMARY_OUT)
