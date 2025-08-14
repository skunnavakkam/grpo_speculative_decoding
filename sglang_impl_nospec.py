"""
The RL framework relies on

a) doing rollouts
b) doing weight updates
c) updating the model

"""

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import trange
import torch
import sglang as sgl
from openai import OpenAI




def get_dataset():
    dataset = load_dataset("trl-lib/tldr")
    prompts = []
    for example in dataset["train"]:
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that summarizes text.",
            },
            {
                "role": "user",
                "content": f"Please summarize this text:\n{example['prompt']}",
            },
        ]
        prompts.append(messages)

    return prompts


def reward_len(text):
    return -abs(20 - len(text))


def get_models():
    bm = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", torch_dtype=torch.bfloat16).to("cuda:1")
    bm_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")

    return bm, bm_tokenizer

client = OpenAI(base_url="http://127.0.0.1:30000/v1", api_key="EMPTY")

def get_rollouts(messages):
    # Generate 4 rollouts
    resp = client.chat.completions.create(
        model="Qwen/Qwen3-4B",
        messages=messages,
        temperature=0.7,
        top_p=0.95,
        max_tokens=100,
        n=4,  # ask SGLang for 4 rollouts in one request
    )
    return [choice.message.content for choice in resp.choices]

if __name__ == "__main__":
    base_model, base_model_tokenizer = get_models()
    prompts = get_dataset()

    # HYPERPARAMS
    num_training_steps = 100
    eps = 0.1
    bm_optimizer = torch.optim.AdamW(base_model.parameters(), lr=1e-5)

    for step in trange(num_training_steps):
        prompt_chat_format = base_model_tokenizer.apply_chat_template(
            prompts[step], tokenize=False, add_generation_prompt=True
        )

        rollouts = get_rollouts(prompts[step])

        rewards = torch.tensor([reward_len(rollout) for rollout in rollouts]).to(torch.float32)
        r_mean = rewards.mean()
        r_std = rewards.std()

        advantage = (rewards - r_mean) / (r_std + 1e-8)

        responses = [prompt_chat_format + rollout for rollout in rollouts]

        tokens = base_model_tokenizer(responses, return_tensors="pt", padding=True).to(
            "cuda:1"
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

        pg_loss = -torch.min(
            ratio * advantage, torch.clamp(ratio, 1 - eps, 1 + eps) * advantage
        ).mean()

        bm_optimizer.zero_grad()
        pg_loss.backward()
        bm_optimizer.step()
