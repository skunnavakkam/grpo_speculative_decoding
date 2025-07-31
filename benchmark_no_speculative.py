"""
grpo_basic_training.py — GRPO w/out speculative decoding (no KL, ratio-clipped), timing report, chat prompts.

This version removes the eagle speculative model and any draft-model fine-tuning. It keeps the ratio-clipped GRPO core and timing instrumentation.
"""

import copy
import time
from typing import List, Tuple

import torch
from datasets import load_dataset
from torch.optim import Adam
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

# ───────────────────────────────────────────────────────────────────────────
# Hyper-parameters
# ───────────────────────────────────────────────────────────────────────────
NUM_STEPS = 100
BASE_MODEL = "Qwen/Qwen3-4B"
DATASET = "trl-lib/tldr"
GROUP_SIZE = 4  # rollouts per prompt
MAX_NEW_TOKENS = 100
TEMPERATURE = 0.7
TOP_P = 0.95
CLIP_EPS = 0.2  # PPO-style ratio clip
SYSTEM_PROMPT = "Summarize the following text in about 100 characters."
LR = 1e-5
EPS = 1e-8


def reward_fn(completions) -> List[float]:
    """Reward = −|len(summary) − 100| (closer to 100 chars is better)."""
    return [-abs(len(c.outputs[0].text) - 100) for c in completions]


def get_seq_log_prob(model, inp_ids, lab_ids):
    """Return summed log-prob per sequence (−mean_nll × seq_len)."""
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
    # No speculative_config → pure decoding
)

device1 = torch.device("cuda:1")
ref_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map={"": device1},
    trust_remote_code=True,
)
ref_model.train()

# Frozen old-policy snapshot for ratio clipping
old_model = copy.deepcopy(ref_model).eval()
for p in old_model.parameters():
    p.requires_grad = False

# Tokeniser
print("🔤 Loading tokenizer …")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

optimizer = Adam(ref_model.parameters(), lr=LR)

# Dataset
print("🗂️  Loading dataset …")
train_ds = load_dataset(DATASET, split="train").select(range(1000))
user_texts = [ex["prompt"] for ex in train_ds]

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

    # Log-probs under old and new policy
    with torch.no_grad():
        old_logp = get_seq_log_prob(old_model, padded_inp, padded_lab)

    new_out = ref_model(input_ids=padded_inp, labels=padded_lab)
    seq_lens = (padded_lab != -100).sum(dim=1)
    new_logp = -new_out.loss * seq_lens.float()

    # Ratio-clipped PG loss
    ratio = torch.exp(new_logp - old_logp)
    clipped_ratio = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS)
    pg_loss = -(torch.min(ratio, clipped_ratio) * advantages).mean()

    optimizer.zero_grad(set_to_none=True)
    pg_loss.backward()
    optimizer.step()

    # Refresh old policy
    old_model.load_state_dict(ref_model.state_dict())

    # Sync updated weights to vLLM (GPU-0)
    with torch.no_grad():
        w0 = [
            (n, p.data.to("cuda:0", non_blocking=True))
            for n, p in ref_model.named_parameters()
        ]
    llm.model_runner.model.load_weights(w0)

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
