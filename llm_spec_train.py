#!/usr/bin/env python3
"""
GRPO with EAGLE speculative model – trainer process (GPU-1).

Launch *after* vllm_worker.py:

    CUDA_VISIBLE_DEVICES=1 python grpo_train.py
"""

import copy
import os
import random
import time
from typing import List, Tuple

import ray
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import SamplingParams
from vllm.utils import get_ip, get_open_port
from rlhf_utils import stateless_init_process_group

# --- Hyper-parameters --------------------------------------------------
NUM_STEPS = 100
BASE_MODEL = "Qwen/Qwen3-4B"
DRAFT_MODEL = "AngelSlim/Qwen3-4B_eagle3"
DATASET = "trl-lib/tldr"
GROUP_SIZE = 4
MAX_NEW = 100
TEMP, TOP_P = 0.7, 0.95
CLIP_EPS = 0.2
SYSTEM_PROMPT = "Summarize the following text in about 100 characters."
LR, DRAFT_LR = 1e-5, 1e-4
FINETUNE_STEPS, DRAFT_FRAC = 5, 0.3
EPS = 1e-8
# ----------------------------------------------------------------------

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")  # trainer → GPU-1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ray.init(address="auto", num_cpus=0, num_gpus=1)
llm_actor = ray.get_actor("vllm_engine")  # from worker

# ----- NCCL group shared with the worker (rank 0 = trainer) ----------
addr, port = get_ip(), get_open_port()
ray.get(
    llm_actor.collective_rpc.remote(
        "init_weight_update_group",
        args=(addr, port, 1, 2),  # rank 1 worker
    )
)
pg = stateless_init_process_group(addr, port, 0, 2, device)

# ----- Base policy & EaModel draft (GPU-1) ----------------------------
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

old_policy = copy.deepcopy(policy).eval()
for p in old_policy.parameters():
    p.requires_grad = False

opt_policy = torch.optim.Adam(policy.parameters(), lr=LR)
opt_draft = torch.optim.Adam(ea.draft_model.parameters(), lr=DRAFT_LR)


# ----- Small helpers --------------------------------------------------
def reward_fn(completions) -> List[float]:
    return [-abs(len(c.outputs[0].text) - 100) for c in completions]


def build_messages(txt: str):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": txt},
    ]


def seq_log_prob(model, inp_ids, lab_ids):
    with torch.no_grad():
        out = model(input_ids=inp_ids, labels=lab_ids)
    L = (lab_ids != -100).sum(dim=1)
    return -out.loss * L.float()


def broadcast_weights(model, rank_name: str):
    """Broadcast *all* parameters in `model` from trainer→worker."""
    for n, p in model.named_parameters():
        handle = llm_actor.collective_rpc.remote(
            "update_weight", args=(f"{rank_name}:{n}", p.dtype, p.shape)
        )
        pg.broadcast(p, src=0, stream=torch.cuda.current_stream())
        ray.get(handle)


# ----- Data -----------------------------------------------------------
ds = load_dataset(DATASET, split="train").select(range(1000))
prompts = [ex["prompt"] for ex in ds]
cached_rollouts: List[Tuple[List[int], List[int]]] = []

print("🚀 training with speculative decoding …")
for step in range(NUM_STEPS):
    # -- Generate (speculative, on GPU-0) ------------------------------
    msgs = build_messages(prompts[step])
    prompt_txt = tok.apply_chat_template(msgs, tokenize=False)
    prompt_ids = tok(prompt_txt).input_ids

    outputs = ray.get(
        llm_actor.generate.remote(
            messages=[msgs],
            prompt_token_ids=[prompt_ids],
            sampling_params=SamplingParams(
                temperature=TEMP, top_p=TOP_P, max_tokens=MAX_NEW, seed=42 + step
            ),
            request_id=[f"{step}-0"],
        )
    )

    # -- Compute GRPO loss --------------------------------------------
    rewards = torch.tensor(reward_fn(outputs), device=device)
    adv = (rewards - rewards.mean()) / (rewards.std() + EPS)

    comp = outputs[0].token_ids[len(outputs[0].prompt_tokens) :]
    inp = torch.tensor([outputs[0].prompt_tokens + comp], device=device)
    lab = torch.tensor([[-100] * len(outputs[0].prompt_tokens) + comp], device=device)

    with torch.no_grad():
        old_lp = seq_log_prob(old_policy, inp, lab)
    new_out = policy(input_ids=inp, labels=lab)
    new_lp = -new_out.loss * (lab != -100).sum()
    ratio = torch.exp(new_lp - old_lp)
    clipped = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS)
    pg_loss = -(torch.min(ratio, clipped) * adv).mean()

    # -- Optimise base policy -----------------------------------------
    opt_policy.zero_grad(set_to_none=True)
    pg_loss.backward()
    opt_policy.step()
    old_policy.load_state_dict(policy.state_dict())

    # keep rollouts for later draft fine-tune
    cached_rollouts.append((outputs[0].prompt_tokens, comp))

    # -- Fine-tune draft model every N steps --------------------------
    if (step + 1) % FINETUNE_STEPS == 0 and cached_rollouts:
        ea.draft_model.train()
        sample = random.sample(
            cached_rollouts, max(1, int(len(cached_rollouts) * DRAFT_FRAC))
        )
        d_loss = 0.0
        for p_ids, c_ids in sample:
            seq = torch.tensor([p_ids + c_ids], device=device)
            mask = (seq != tok.pad_token_id).long()
            h = (
                policy(
                    input_ids=seq,
                    attention_mask=mask,
                    output_hidden_states=True,
                    return_dict=True,
                )
                .hidden_states[-2]
                .squeeze(0)
            )
            inp_h, tgt_h = h[:-1].unsqueeze(0), h[1:].unsqueeze(0)
            d_loss += F.mse_loss(ea.draft_model(inp_h)[0], tgt_h)
        d_loss /= len(sample)
        opt_draft.zero_grad(set_to_none=True)
        d_loss.backward()
        opt_draft.step()
        ea.draft_model.eval()
        cached_rollouts.clear()

    # -- Broadcast fresh weights (policy + draft) ---------------------
    broadcast_weights(policy, "policy")
    broadcast_weights(ea.draft_model, "draft")

    if (step + 1) % 10 == 0:
        print(
            f"[{step + 1:03d}] reward {rewards.mean():+6.2f} "
            f"| pg_loss {pg_loss.item():.4f}"
        )

print("✅ done.")
