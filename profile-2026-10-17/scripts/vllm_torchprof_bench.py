#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""vllm_torchprof_bench.py — torch profiler 内核级 prefill/decode 时间轴采集

背景：nsys 2026.5 的 CUPTI 12.9 与本机驱动 535（CUDA 12.2）不兼容（green ctx 回调不存在
→ CUDA 注入失败）；torch 自带 CUPTI 12.1 匹配驱动，可采到 GPU kernel 事件。
输出：chrome trace（.json），含 ===phase:xxx=== user_annotation 相位窗口。
"""
import os, time, json

os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
os.environ.setdefault("VLLM_USE_V1", "0")

import torch
from torch.profiler import profile, ProfilerActivity, record_function
from vllm import LLM, SamplingParams

MODEL = "/home/ctyun/.cache/TeleAgent/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
TEMP = os.path.dirname(os.path.abspath(__file__))
TRACE = os.path.join(TEMP, "vllm_chrome.json")

llm = LLM(model=MODEL, gpu_memory_utilization=0.25, max_model_len=4096,
          seed=0, disable_log_stats=True, dtype="bfloat16")
tok = llm.get_tokenizer()
base = ("高性能推理系统的核心矛盾在于算子执行效率与调度开销的平衡，"
        "工程师需要在内核启动延迟、内存搬运带宽与计算密度之间不断权衡取舍。")
prompt = "请基于以下技术论述续写并深入展开：\n" + base * 30
print(f"[bench] prompt_tokens={len(tok.encode(prompt))}", flush=True)

def gen(text, mt):
    llm.generate([text], SamplingParams(max_tokens=mt, temperature=0.0, ignore_eos=True, seed=0))

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             record_shapes=False, with_stack=False) as prof:
    with record_function("===phase:warmup==="):
        t0 = time.perf_counter(); gen(prompt + "（热身）", 4); torch.cuda.synchronize()
        print(f"[bench] warmup {(time.perf_counter()-t0)*1e3:.2f}ms", flush=True)
    time.sleep(0.3)
    with record_function("===phase:prefill_only==="):
        t0 = time.perf_counter(); gen(prompt + "（测量甲）", 1); torch.cuda.synchronize()
        print(f"[bench] prefill_only {(time.perf_counter()-t0)*1e3:.2f}ms", flush=True)
    time.sleep(0.3)
    with record_function("===phase:prefill_decode==="):
        t0 = time.perf_counter(); gen(prompt + "（测量乙）", 129); torch.cuda.synchronize()
        print(f"[bench] prefill_decode {(time.perf_counter()-t0)*1e3:.2f}ms", flush=True)

prof.export_chrome_trace(TRACE)
n = sum(1 for e in prof.events())
print(f"[done] events~{n} -> {TRACE}", flush=True)
