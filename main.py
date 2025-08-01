"""
grpo_speculative_training.py  —  **GRPO w/out KL (ratio‑clipped), timing report, chat prompts, subsampled draft fine‑tune**

Restored complete script and ensured **timing information is printed** after
training. Highlights:
• Old‑policy snapshot for ratio‑clipped GRPO (ε = 0.2).
• No KL penalty.
• Chat‑formatted prompts (Qwen ChatML).
• Generation vs. update timing captured per step and summarised at the end.
• Draft speculative model fine‑tuned on a random fraction (`DRAFT_FRACTION`) of
  cached rollouts every `FINETUNE_STEPS`.
"""

import copy
import random
import time
from typing import List, Tuple

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.optim import Adam
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from eagle.model.ea_model import EaModel


# ───────────────────────────────────────────────────────────────────────────
# Hyper‑parameters
# ───────────────────────────────────────────────────────────────────────────
NUM_STEPS = 100
BASE_MODEL = "Qwen/Qwen3-4B"
SPECULATIVE_MODEL = "AngelSlim/Qwen3-4B_eagle3"
DATASET = "trl-lib/tldr"
GROUP_SIZE = 4  # rollouts per prompt
MAX_NEW_TOKENS = 100
TEMPERATURE = 0.7
TOP_P = 0.95
CLIP_EPS = 0.2  # PPO‑style ratio clip
SYSTEM_PROMPT = "Summarize the following text in about 100 characters."
LR = 1e-5
DRAFT_LR = 1e-4
FINETUNE_STEPS = 5
DRAFT_FRACTION = 0.3  # train draft on 30% of cached rollouts
EPS = 1e-8

# Uncomment if you want to limit visible GPUs
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


def reward_fn(completions) -> List[float]:
    """Reward = −|len(summary) − 100| (closer to 100 chars is better)."""
    return [-abs(len(c.outputs[0].text) - 100) for c in completions]


def get_seq_log_prob(model, inp_ids, lab_ids):
    """Return summed log‑prob per sequence (−mean_nll × seq_len)."""
    with torch.no_grad():
        out = model(input_ids=inp_ids, labels=lab_ids)
    seq_lens = (lab_ids != -100).sum(dim=1)
    return -out.loss * seq_lens.float()


def build_messages(user_text: str):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]


# ───────────────────────────────────────────────────────────────────────────
# Model setup
# ───────────────────────────────────────────────────────────────────────────
print("⏳ Initialising models …")

llm = LLM(
    model=BASE_MODEL,
    tensor_parallel_size=1,
    dtype="fp16",
    gpu_ids=[0],
    trust_remote_code=True,
    speculative_config={
        "model": SPECULATIVE_MODEL,
        "num_speculative_tokens": 5,
        "method": "eagle",
    },
)

device1 = torch.device("cuda:1")
ref_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map={"": device1},
    trust_remote_code=True,
)
ref_model.train()

# Frozen old‑policy snapshot for ratio clipping
old_model = copy.deepcopy(ref_model).eval()
for p in old_model.parameters():
    p.requires_grad = False

# Tokeniser
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

# Draft (speculative) model wrapper

ea_wrapper = EaModel(
    base_model=ref_model,
    base_model_name_or_path=BASE_MODEL,
    ea_model_path=SPECULATIVE_MODEL,
)

ea_wrapper.draft_model.to(device1).train()

optimizer = Adam(ref_model.parameters(), lr=LR)
draft_optimizer = Adam(ea_wrapper.draft_model.parameters(), lr=DRAFT_LR)

# Dataset
train_ds = load_dataset(DATASET, split="train").select(range(1000))
user_texts = [ex["prompt"] for ex in train_ds]

cached_rollouts: List[Tuple[List[int], List[int]]] = []
step_times: List[Tuple[float, float]] = []  # (generation_time, update_time)

print("🚀 Starting training loop …")

# ───────────────────────────────────────────────────────────────────────────
# Training loop
# ───────────────────────────────────────────────────────────────────────────
for step in range(NUM_STEPS):
    messages = build_messages(user_texts[step])

    # ─ Generation timing ─
    t0 = time.perf_counter()

    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False)
    prompt_ids = tokenizer(prompt_text).input_ids

    outputs = []
    for g in range(GROUP_SIZE):
        out = llm.generate(
            messages=[messages],
            prompt_token_ids=[prompt_ids],
            sampling_params=SamplingParams(
                temperature=TEMPERATURE,
                max_tokens=MAX_NEW_TOKENS,
                top_p=TOP_P,
                seed=step * GROUP_SIZE + g,
            ),
            request_id=[f"{step}-{g}"],
        )
        outputs.append(out[0])

    gen_time = time.perf_counter() - t0

    # ─ Update timing ─
    t1 = time.perf_counter()

    rollouts = []
    for out in outputs:
        comp_ids = out.token_ids[len(out.prompt_tokens) :]
        rollouts.append((out.prompt_tokens, comp_ids))
        cached_rollouts.append((out.prompt_tokens, comp_ids))

    rewards = torch.tensor(reward_fn(outputs), device=device1)
    advantages = rewards - rewards.mean()
    advantages = (advantages - advantages.mean()) / (advantages.std() + EPS)

    inp_t, lab_t = [], []
    for p_ids, c_ids in rollouts:
        seq = torch.tensor(p_ids + c_ids, device=device1)
        lab = torch.tensor([-100] * len(p_ids) + c_ids, device=device1)
        inp_t.append(seq)
        lab_t.append(lab)

    padded_inp = torch.nn.utils.rnn.pad_sequence(
        inp_t, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    padded_lab = torch.nn.utils.rnn.pad_sequence(
        lab_t, batch_first=True, padding_value=-100
    )

    # Log‑probs under old and new policy
    with torch.no_grad():
        old_logp = get_seq_log_prob(old_model, padded_inp, padded_lab)

    new_out = ref_model(input_ids=padded_inp, labels=padded_lab)
    seq_lens = (padded_lab != -100).sum(dim=1)
    new_logp = -new_out.loss * seq_lens.float()

    # Ratio‑clipped PG loss
    ratio = torch.exp(new_logp - old_logp)
    clipped_ratio = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS)
    pg_loss = -(torch.min(ratio, clipped_ratio) * advantages).mean()

    optimizer.zero_grad(set_to_none=True)
    pg_loss.backward()
    optimizer.step()

    # Refresh old policy
    old_model.load_state_dict(ref_model.state_dict())

    # Sync updated weights to vLLM (GPU‑0)
    with torch.no_grad():
        w0 = [
            (n, p.data.to("cuda:0", non_blocking=True))
            for n, p in ref_model.named_parameters()
        ]
    llm.model_runner.model.load_weights(w0)

    # Draft fine‑tune periodically on a subsample
    if (step + 1) % FINETUNE_STEPS == 0 and cached_rollouts:
        sample_size = max(1, int(len(cached_rollouts) * DRAFT_FRACTION))
        sampled = random.sample(cached_rollouts, sample_size)

        ea_wrapper.draft_model.train()
        d_loss = 0.0
        for p_ids, c_ids in sampled:
            seq = torch.tensor([p_ids + c_ids], device=device1)
            mask = (seq != tokenizer.pad_token_id).long()
            hid = (
                ref_model(
                    input_ids=seq,
                    attention_mask=mask,
                    output_hidden_states=True,
                    return_dict=True,
                )
                .hidden_states[-2]
                .squeeze(0)
            )
            feat_in, feat_tgt = hid[:-1].unsqueeze(0), hid[1:].unsqueeze(0)
            pred = ea_wrapper.draft_model(feat_in)[0]
            d_loss += F.mse_loss(pred, feat_tgt)
        d_loss /= sample_size
        draft_optimizer.zero_grad(set_to_none=True)
        d_loss.backward()
        draft_optimizer.step()
        ea_wrapper.draft_model.eval()
        with torch.no_grad():
            dw = [
                (n, p.data.to("cuda:0", non_blocking=True))
                for n, p in ea_wrapper.draft_model.named_parameters()
            ]
        llm.model_runner.speculative_model.load_weights(dw)
        cached_rollouts.clear()

    upd_time = time.perf_counter() - t1
    step_times.append((gen_time, upd_time))

    # ─ Logging every 10 steps ─
    if (step + 1) % 10 == 0:
        print(
            f"[step {step + 1:03d}] reward {rewards.mean():+.2f} | pg_loss {pg_loss.item():.4f} | "
            f"gen {gen_time:.3f}s | upd {upd_time:.3f}s"
        )

# ───────────────────────────────────────────────────────────────────────────
# Timing summary
# ───────────────────────────────────────────────────────────────────────────
print("\n🕒  Timing summary (seconds):")
for idx, (g, u) in enumerate(step_times, 1):
    print(f"Step {idx:03d}: generation {g:.3f} | update {u:.3f}")

gen_avg = sum(g for g, _ in step_times) / len(step_times)
upd_avg = sum(u for _, u in step_times) / len(step_times)
print(
    f"\nAverage over {NUM_STEPS} steps → generation: {gen_avg:.3f}s | update: {upd_avg:.3f}s"
)
print("✅ Training complete.")
