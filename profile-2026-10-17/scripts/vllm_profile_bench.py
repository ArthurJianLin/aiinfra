#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""vllm_profile_bench.py — prefill/decode 分离测量脚本（供 nsys 跟踪）

设计：
  - 同一进程内做三次 generate：warmup / prefill_only(max_tokens=1) / prefill_decode(max_tokens=129)
  - 每次用 NVTX range 标记，nsys 导出 sqlite 后可按窗口切分内核事件
  - 用 (t_prefill_decode - t_prefill_only) / 128 得到单流 decode 每 token 时延
"""
import os, time, json

os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
os.environ.setdefault("VLLM_USE_V1", "0")  # 0.7.3 默认 V0 引擎，进程内运行

import torch
from vllm import LLM, SamplingParams

MODEL = "/home/ctyun/.cache/TeleAgent/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "profile_result.json")

llm = LLM(model=MODEL, gpu_memory_utilization=0.25, max_model_len=4096,
          seed=0, disable_log_stats=True, dtype="bfloat16")

tok = llm.get_tokenizer()
base = ("高性能推理系统的核心矛盾在于算子执行效率与调度开销的平衡，"
        "工程师需要在内核启动延迟、内存搬运带宽与计算密度之间不断权衡取舍。")
prompt = "请基于以下技术论述续写并深入展开：\n" + base * 30

def run_once(text, max_tokens, tag):
    sp = SamplingParams(max_tokens=max_tokens, temperature=0.0, ignore_eos=True, seed=0)
    n_in = len(tok.encode(text))
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    torch.cuda.nvtx.range_push(tag)
    out = llm.generate([text], sp)[0]
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    dt = t1 - t0
    ngen = len(out.outputs[0].token_ids)
    print(f"[bench] {tag}: {dt*1000:.2f} ms | prompt_tokens={n_in} | gen_tokens={ngen}", flush=True)
    return dt, n_in, ngen

# 1) 热身（一次性开销：首次采样器执行、缓存分配等）
_, n_warm, _ = run_once(prompt + "（热身）", 4, "warmup")
time.sleep(0.4)

# 2) prefill-only：新 prompt + 仅 1 个生成 token
t_p, n_p1, _ = run_once(prompt + "（测量甲）", 1, "prefill_only")
time.sleep(0.4)

# 3) prefill + decode：新 prompt + 129 个生成 token
t_pd, n_p2, n_d = run_once(prompt + "（测量乙）", 129, "prefill_decode")

n_decode = n_d - 1
decode_per_tok = (t_pd - t_p) / n_decode
result = {
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "gpu": "NVIDIA A10-24Q (24GB)",
    "gpu_mem_util": 0.25,
    "max_model_len": 4096,
    "dtype": "bfloat16",
    "cudagraph": "on (default)",
    "prompt_tokens_warmup": n_warm,
    "prompt_tokens_prefill_only": n_p1,
    "prompt_tokens_prefill_decode": n_p2,
    "t_prefill_plus1_ms": round(t_p * 1000, 2),
    "t_prefill_decode_ms": round(t_pd * 1000, 2),
    "decode_tokens": n_decode,
    "decode_ms_per_token": round(decode_per_tok * 1000, 3),
    "decode_tok_per_s": round(1.0 / decode_per_tok, 1),
}
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
print("[bench] RESULT " + json.dumps(result, ensure_ascii=False), flush=True)
