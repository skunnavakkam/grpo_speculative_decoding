#!/usr/bin/env python3
"""
vLLM inference worker (GPU-0) with EAGLE speculative decoding.

Start first:

    CUDA_VISIBLE_DEVICES=0 python vllm_worker.py
"""

import os
import time
import ray
from vllm import LLM
from vllm.config import ParallelConfig
from vllm.engine.arg_utils import EngineArgs


BASE_MODEL = "Qwen/Qwen3-4B"
DRAFT_MODEL = "AngelSlim/Qwen3-4B_eagle3"

# ---------------------------------------------------------------------
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # pin to GPU-0
ray.init(address="auto")


@ray.remote(num_gpus=1, max_restarts=-1)
class VLLMEngine(LLM):
    """Detached actor so the trainer can find it by name."""

    pass

parallel_config = ParallelConfig(pipeline_parallel_size=1, tensor_parallel_size=1, worker_use_ray=True)
parallel_config.worker_extension_cls = "rlhf_utils.WorkerExtension"

engine_args = EngineArgs(
    model=BASE_MODEL,
    dtype="fp16",
    tensor_parallel_size=1,
    enforce_eager=True,  # faster swaps
)

engine_args.parallel_config = parallel_config
engine_args.speculative_config =  {
    "model": DRAFT_MODEL,
    "method": "eagle",
    "num_speculative_tokens": 5,
}

engine = VLLMEngine.options(name="vllm_engine", lifetime="detached").remote(
    model=BASE_MODEL,
    engine_args=engine_args
)

print("[vllm-worker] ready – waiting for generate / weight-update RPCs …")
while True:
    time.sleep(3600)
