#!/usr/bin/env python3
"""
GRPO with EAGLE speculative model – trainer process (GPU-1)
*Uses cached log-probs instead of a frozen copy of the policy.*

Implements the first three GRPO fixes:
1. **Group sampling** – sample `GROUP_SIZE` roll-outs per prompt and compute group-relative advantages.
2. **Std-guard** – clamp the standard deviation to avoid divide-by-zero.
3. **Advantage normalisation** – rewards are centred **and** z-scored within the group before the PPO surrogate.

Launch **after** `vllm_worker.py` in a second terminal:

    CUDA_VISIBLE_DEVICES=1 python grpo_train.py
"""

import os
import time
from typing import List, Tuple

import ray
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import SamplingParams
from vllm.utils import get_ip, get_open_port
from vllm.examples.offline_inference.rlhf_utils import (
    stateless_init_process_group,
)

# ────────────────────────────── HYPER-PARAMS ──────────────────────────────
NUM_STEPS = 100
BASE_MODEL = "Qwen/Qwen3-4B"
DRAFT_MODEL = "AngelSlim/Qwen3-4B_eagle3"
DATASET = "trl-lib/tldr"
GROUP_SIZE = 4  # roll-outs per prompt (≥2 for GRPO)
MAX_NEW = 100
TEMP, TOP_P = 0.7, 0.95
CLIP_EPS = 0.2
SYSTEM_PROMPT = "Summarize the following text in about 100 characters."
LR, DRAFT_LR = 1e-5, 1e-4
FINETUNE_STEPS, DRAFT_FRAC = 5, 0.3
EPS = 1e-8  # numerical stability constant

# ────────────────────────────── ENV / RAY ─────────────────────────────────
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")  # trainer lives on GPU-1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ray.init(address="auto", num_cpus=0, num_gpus=1)
llm_actor = ray.get_actor("vllm_engine")  # created by vllm_worker.py

# Shared NCCL process-group between trainer (rank-0) and worker (rank-1)
addr, port = get_ip(), get_open_port()
ray.get(
    llm_actor.collective_rpc.remote("init_weight_update_group", args=(addr, port, 1, 2))
)
pg = stateless_init_process_group(addr, port, 0, 2, device)

# ───────────────────────── BASE POLICY & DRAFT ────────────────────────────

tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

policy = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map={"": device},
    trust_remote_code=True,
)

from eagle.model.ea_model import EaModel

ea = EaModel(
    base_model=policy,
    base_model_name_or_path=BASE_MODEL,
    ea_model_path=DRAFT_MODEL,
)

ea.draft_model.to(device).train()

opt_policy = torch.optim.Adam(policy.parameters(), lr=LR)
opt_draft = torch.optim.Adam(ea.draft_model.parameters(), lr=DRAFT_LR)

# ──────────────────────────── HELPERS ─────────────────────────────────────


def reward_fn(completions) -> List[float]:
    """Reward = −|len(summary) − 100|."""
    return [-abs(len(c.outputs[0].text) - 100) for c in completions]


def build_messages(text: str):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]


def seq_log_prob(model, inp_ids: torch.Tensor, lab_ids: torch.Tensor):
    """Log-probability summed over sequence (no grad)."""
    with torch.no_grad():
        out = model(input_ids=inp_ids, labels=lab_ids)
    seq_lens = (lab_ids != -100).sum(dim=1)
    return -out.loss * seq_lens.float()


def broadcast_weights(model, prefix: str):
    """Broadcast all parameters of *model* from trainer → worker."""
    for name, p in model.named_parameters():
        handle = llm_actor.collective_rpc.remote(
            "update_weight", args=(f"{prefix}:{name}", p.dtype, p.shape)
        )
        pg.broadcast(p, src=0, stream=torch.cuda.current_stream())
        ray.get(handle)


# ──────────────────────────── DATA ────────────────────────────────────────

ds = load_dataset(DATASET, split="train").select(range(1000))
prompts = [ex["prompt"] for ex in ds]

# ────────────────────────── TIMING BUFFERS ───────────────────────────────
step_times: List[Tuple[float, float]] = []  # (generation_time, update_time)

print("🚀 training with cached log-probs & group sampling …")
for step in range(NUM_STEPS):
    # ── 1 · GENERATION (GROUP_SIZE roll-outs) ────────────────────────────
    t0 = time.perf_counter()

    messages = build_messages(prompts[step])
    prompt_txt = tok.apply_chat_template(messages, tokenize=False)
    prompt_ids = tok(prompt_txt).input_ids

    outputs = []
    for i in range(GROUP_SIZE):
        # Vary seed per rollout to ensure diversity
        out = ray.get(
            llm_actor.generate.remote(
                messages=[messages],
                prompt_token_ids=[prompt_ids],
                sampling_params=SamplingParams(
                    temperature=TEMP,
                    top_p=TOP_P,
                    max_tokens=MAX_NEW,
                    seed=42 + step * GROUP_SIZE + i,
                ),
                request_id=[f"{step}-{i}"],
            )
        )
        outputs.extend(out)  # each call returns a list of length 1

    gen_time = time.perf_counter() - t0

    # ── 1b · CACHE π_old LOG-PROB for each rollout ───────────────────────
    inp_tensors, lab_tensors = [], []
    for out in outputs:
        comp_ids = out.token_ids[len(out.prompt_tokens) :]
        inp = torch.tensor(out.prompt_tokens + comp_ids, device=device)
        lab = torch.tensor([-100] * len(out.prompt_tokens) + comp_ids, device=device)
        inp_tensors.append(inp)
        lab_tensors.append(lab)

    inp_tensor = pad_sequence(
        inp_tensors, batch_first=True, padding_value=tok.pad_token_id
    )
    lab_tensor = pad_sequence(lab_tensors, batch_first=True, padding_value=-100)

    old_logp = seq_log_prob(policy, inp_tensor, lab_tensor)  # shape [G]

    # ── 2 · ADVANTAGE COMPUTATION (with std-guard) ───────────────────────
    rewards = torch.tensor(reward_fn(outputs), device=device, dtype=torch.float32)
    std = rewards.std()
    std = std.clamp_min(1e-6)  # guard against σ=0
    advantages = (rewards - rewards.mean()) / std  # centred & z-scored

    # ── 3 · POLICY UPDATE (clipped surrogate) ────────────────────────────
    t1 = time.perf_counter()

    # Compute π_new log-prob per rollout *with gradients*
    new_logps = []
    for inp_i, lab_i in zip(inp_tensors, lab_tensors):
        out_i = policy(input_ids=inp_i.unsqueeze(0), labels=lab_i.unsqueeze(0))
        seq_len_i = (lab_i != -100).sum()
        new_logps.append(-out_i.loss * seq_len_i)
    new_logp = torch.stack(new_logps)  # shape [G]

    ratio = torch.exp(new_logp - old_logp)
    clipped_ratio = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS)
    pg_loss = -(torch.min(ratio, clipped_ratio) * advantages).mean()

    opt_policy.zero_grad(set_to_none=True)
    pg_loss.backward()
    opt_policy.step()

    # ── 4 · OPTIONAL: DRAFT FINE-TUNE (every FINETUNE_STEPS) ─────────────
    if (step + 1) % FINETUNE_STEPS == 0:
        ea.draft_model.train()
        with torch.no_grad():
            hidden = policy(
                input_ids=inp_tensor,
                attention_mask=(inp_tensor != tok.pad_token_id).long(),
                output_hidden_states=True,
                return_dict=True,
            ).hidden_states[-2]
        # Use only the first rollout for draft fine-tune (illustrative)
        feat_in, feat_tgt = hidden[0][:-1].unsqueeze(0), hidden[0][1:].unsqueeze(0)
        d_pred = ea.draft_model(feat_in)[0]
        d_loss = F.mse_loss(d_pred, feat_tgt)
        opt_draft.zero_grad(set_to_none=True)
        d_loss.backward()
        opt_draft.step()
        ea.draft_model.eval()

    # ── 5 · BROADCAST UPDATED WEIGHTS ────────────────────────────────────
    broadcast_weights(policy, "policy")
    broadcast_weights(ea.draft_model, "draft")

    upd_time = time.perf_counter() - t1
    step_times.append((gen_time, upd_time))

    if (step + 1) % 10 == 0:
        print(
            f"[{step + 1:03d}] reward {rewards.mean():+6.2f} | pg_loss {pg_loss.item():.4f} | "
            f"gen {gen_time:.3f}s | upd {upd_time:.3f}s"
        )

# ─────────────────────────── TIMING SUMMARY ──────────────────────────────
print("\n🕒  Timing summary (seconds):")
for idx, (g, u) in enumerate(step_times, 1):
    print(f"Step {idx:03d}: generation {g:.3f} | update {u:.3f}")

gen_avg = sum(g for g, _ in step_times) / len(step_times)
upd_avg = sum(u for _, u in step_times) / len(step_times)
print(
    f"\nAverage over {NUM_STEPS} steps → generation: {gen_avg:.3f}s | update: {upd_avg:.3f}s"
)
print("✅ training complete.")
