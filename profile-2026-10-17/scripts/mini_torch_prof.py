#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""mini_torch_prof.py — 验证 torch profiler (Kineto/CUPTI) 在本机 vGPU 上是否也被禁"""
import json, torch
from torch.profiler import profile, ProfilerActivity

a = torch.randn(2048, 2048, device="cuda")
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for _ in range(10):
        b = a @ a
    torch.cuda.synchronize()

evs = prof.key_averages()
gpu_rows = []
for e in evs:
    if getattr(e, "device_type", None) is not None and "CUDA" in str(e.device_type):
        gpu_rows.append((e.key, e.count, e.device_time_total))
print("[torch-profiler] device rows:")
for k, c, t in gpu_rows[:10]:
    print(f"  {k} count={c} device_us_total={t}")
print(f"[torch-profiler] CUDA device rows = {len(gpu_rows)} （0 表示 GPU 计时被禁）")
