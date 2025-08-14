#!/usr/bin/env python3
"""
vLLM inference worker (GPU-0) with EAGLE speculative decoding.

Start first:

    CUDA_VISIBLE_DEVICES=0 python vllm_worker.py
"""

import os

import ray
import torch
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from rlhf_utils import stateless_init_process_group
from transformers import AutoModelForCausalLM, AutoTokenizer

from vllm import LLM, SamplingParams
from vllm.utils import get_ip, get_open_port
from eagle.model.ea_model import EaModel
from huggingface_hub import hf_hub_download

from pathlib import Path
from eagle.ge_data import allocation

from datasets import load_dataset

NUM_TRAINING_STEPS = 100
NUM_ROLLOUTS = 4
SYSTEM_PROMPT = "Summarize the following text in 100 words or less."
eps = 0.2


class MyLLM(LLM):
    """Configure the vLLM worker for Ray placement group execution."""

    def __init__(self, *args, **kwargs):
        # Remove the top-level CUDA_VISIBLE_DEVICES variable set by Ray
        # so that vLLM can manage its own device placement within the worker.
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        super().__init__(*args, **kwargs)


BASE_MODEL = "Qwen/Qwen3-4B"
DRAFT_MODEL = "AngelSlim/Qwen3-4B_eagle3"

# Load base model and draft model on CUDA:0
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL).to("cuda:0")
base_model_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
configpath = os.path.join(DRAFT_MODEL, "config.json")

if not os.path.exists(configpath):
    configpath = hf_hub_download(DRAFT_MODEL, "config.json")


draft_model = EaModel(base_model, BASE_MODEL, configpath)
load_model_path = os.path.join(DRAFT_MODEL, "pytorch_model.bin")
if not os.path.exists(load_model_path):
    load_model_path = hf_hub_download(DRAFT_MODEL, "pytorch_model.bin")
ea_layer_state_dict = torch.load(load_model_path, map_location=base_model.device)
draft_model.ea_layer.load_state_dict(ea_layer_state_dict, strict=True)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
ray.init()

pg_inference = placement_group([{"GPU": 1, "CPU": 0}] * 2)
ray.get(pg_inference.ready())

scheduling_inference = PlacementGroupSchedulingStrategy(
    placement_group=pg_inference,
    placement_group_capture_child_tasks=True,
    placement_group_bundle_index=0,
)

llm = ray.remote(
    num_cpus=0,
    num_gpus=0,
    scheduling_strategy=scheduling_inference,
)(MyLLM).remote(
    model=BASE_MODEL,
    speculative_config={
        "model": DRAFT_MODEL,
        "method": "eagle",
        "num_speculative_tokens": 5,
    },
    enforce_eager=True,
    worker_extension_cls="rlhf_utils.WorkerExtension",
    tensor_parallel_size=2,
    distributed_executor_backend="ray",
)

sampling_params = SamplingParams(
    max_tokens=1024,
    temperature=0.6,
    top_p=0.95,
    top_k=40,
    repetition_penalty=1.18,
    n=NUM_ROLLOUTS,
)

master_address = get_ip()
master_port = get_open_port()

handle = llm.collective_rpc.remote(
    "init_weight_update_group", args=(master_address, master_port, 1, 3)
)

model_update_group = stateless_init_process_group(
    master_address, master_port, 0, 3, torch.device("cuda:0")
)
ray.get(handle)


dataset = load_dataset("trl-lib/tldr")
# Format dataset examples into conversational format
prompts = []
for example in dataset["train"]:
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that summarizes text.",
        },
        {
            "role": "user",
            "content": f"Please summarize this text:\n{example['content']}",
        },
        {"role": "assistant", "content": example["summary"]},
    ]
    prompts.append(messages)

prompts = base_model_tokenizer.apply_chat_template(
    prompts, tokenize=False, add_generation_prompt=True
)


def reward_len(text):
    return -abs(20 - len(text))


bm_optimizer = torch.optim.AdamW(base_model.parameters(), lr=1e-5)
ea_optimizer = torch.optim.AdamW(draft_model.ea_layer.parameters(), lr=1e-5)

response_cache = []

path = Path("ge_data").mkdir(exist_ok=True)

for i in range(NUM_TRAINING_STEPS):
    # generate NUM_ROLLOUTS rollouts
    prompt = prompts[i]
    outputs = ray.get(llm.generate.remote(prompt, sampling_params))

    generated_texts = [output.outputs[0].text for output in outputs]
    rewards = torch.tensor([reward_len(text) for text in generated_texts])
    r_mean = rewards.mean()
    r_std = rewards.std()

    advantage = (rewards - r_mean) / (r_std + 1e-8)

    responses = [p + g for p, g in zip(prompt, generated_texts)]

    tokens = base_model_tokenizer(responses, return_tensors="pt", padding=True).to(
        "cuda:0"
    )

    outputs = base_model(
        input_ids=tokens["input_ids"],
        attention_mask=tokens["attention_mask"],
        return_dict=True,
    )
    logprobs = torch.log_softmax(outputs.logits, dim=-1)

    with torch.no_grad():
        old_logprobs = logprobs.clone()
    new_logprobs = logprobs

    ratio = torch.exp(new_logprobs - old_logprobs)

    surr1 = ratio * advantage
    surr2 = torch.clamp(ratio, 1 - eps, 1 + eps) * advantage

    loss = -torch.min(surr1, surr2).mean()

    bm_optimizer.zero_grad()
    loss.backward()
    bm_optimizer.step()

    for name, p in base_model.named_parameters():
        handle = llm.collective_rpc.remote(
            "update_weight", args=(name, p.dtype, p.shape)
        )
        model_update_group.broadcast(p, src=0, stream=torch.cuda.current_stream())
        ray.get(handle)

    if i % 5 == 0:
        # update the draft model
        # Take the last two rollouts from the cache
        last_two_rollouts = generated_texts[-2 * NUM_ROLLOUTS :]

        pass
