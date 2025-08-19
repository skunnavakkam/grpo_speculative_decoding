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
import os
import requests

from openai import OpenAI
from specforge import AutoDraftModelConfig, AutoEagle3DraftModel, OnlineEagle3Model
from specforge.data import (
    build_eagle3_dataset,
    prepare_dp_dataloaders,
    generate_vocab_mapping_file,
)
from specforge.distributed import get_dp_group


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


def get_base_model():
    bm = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-4B", torch_dtype=torch.bfloat16
    ).to("cuda:1")
    bm_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")

    return bm, bm_tokenizer


def get_draft_model():
    draft_model_config = AutoDraftModelConfig.from_file("qwen_3_4b_eagle3_config.json")

    draft_model = (
        AutoEagle3DraftModel(
            draft_model_config=draft_model_config,
        )
        .to("cuda:1")
        .to(torch.bfloat16)
    )

    draft_model.load_embedding(
        "Qwen/Qwen3-4B", embedding_key="model.embed_tokens.weight"
    )
    draft_model.freeze_embedding()

    print("initialized draft model")

    return draft_model, draft_model_config


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
    base_model, base_model_tokenizer = get_base_model()
    draft_model, draft_model_config = get_draft_model()
    prompts = get_dataset()

    # HYPERPARAMS
    num_training_steps = 100
    eps = 0.1
    bm_optimizer = torch.optim.AdamW(base_model.parameters(), lr=1e-5)

    generation_cache = []  # we then train off of this

    """
    generation_cache = [
        [
            {
                "role": "user",
                "content": "Please summarize this text:\n{text}",
            },
            {
                "role": "assistant",
                "content": "{rollout}",
            },
        ]
    ]
    """

    for step in trange(num_training_steps):
        prompt_chat_format = base_model_tokenizer.apply_chat_template(
            prompts[step], tokenize=False, add_generation_prompt=True
        )

        rollouts = get_rollouts(prompts[step])

        for rollout in rollouts:
            step = prompts[step] + [{"role": "assistant", "content": rollout}]
            generation_cache.append(step)

        rewards = torch.tensor([reward_len(rollout) for rollout in rollouts]).to(
            torch.float32
        )
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

        if step % 5 == 0:
            target_model = base_model

            train_dataset = build_eagle3_dataset(
                dataset=generation_cache,
                tokenizer=base_model_tokenizer,
                chat_template="qwen",
                max_length=2048,
                num_proc=8,
            )

            vocab_mapping_path = generate_vocab_mapping_file(
                dataset=train_dataset,
                target_vocab_size=draft_model_config.vocab_size,
                draft_vocab_size=draft_model_config.draft_vocab_size,
            )

            train_dataloader = prepare_dp_dataloaders(
                train_dataset,
                4,
                num_workers=4,
                shuffle=True,
                process_group=get_dp_group(),
            )

            draft_model.load_vocab_mapping(vocab_mapping_path)

            # Create OnlineEagle3Model for training
            eagle3_model = OnlineEagle3Model(
                target_model=target_model,
                draft_model=draft_model,
                length=7,  # TTT length
                attention_backend="flex_attention",
            )

            # Setup optimizer and scheduler for draft model training
            draft_optimizer = torch.optim.AdamW(draft_model.parameters(), lr=1e-4)

            # Train the draft model on the generated data
            draft_model.train()
            num_draft_epochs = 1  # Train for a few epochs on current data

            for epoch in range(num_draft_epochs):
                epoch_losses = []
                epoch_acces = [[] for _ in range(eagle3_model.length)]

                for batch_data in train_dataloader:
                    draft_optimizer.zero_grad()

                    # Move data to correct device
                    input_ids = batch_data["input_ids"].to("cuda:1")
                    attention_mask = batch_data["attention_mask"].to("cuda:1")
                    loss_mask = batch_data["loss_mask"].to("cuda:1")

                    # Forward pass through Eagle3 model
                    plosses, _, acces = eagle3_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        loss_mask=loss_mask,
                    )

                    # Calculate weighted loss (same as in the reference implementation)
                    ploss_weight = [0.8**i for i in range(len(plosses))]
                    ploss = sum(
                        [ploss_weight[i] * plosses[i] for i in range(len(plosses))]
                    )

                    # Backward pass and optimization
                    ploss.backward()
                    draft_optimizer.step()

                    # Collect metrics
                    epoch_losses.append(ploss.item())
                    for i in range(len(acces)):
                        epoch_acces[i].append(acces[i])

                # Print training metrics
                avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
                print(
                    f"Draft model training - Step {step}, Epoch {epoch + 1}, Avg Loss: {avg_loss:.4f}"
                )

                for i in range(len(epoch_acces)):
                    if epoch_acces[i]:
                        avg_acc = sum(epoch_acces[i]) / len(epoch_acces[i])
                        print(
                            f"Draft model training - Step {step}, Epoch {epoch + 1}, Position {i}, Acc: {avg_acc:.4f}"
                        )

            print(f"Completed draft model training at step {step}")

            # Save the model state
            epoch_output_dir = os.path.join("draft_model_checkpoints", f"step_{step}")
            os.makedirs(epoch_output_dir, exist_ok=True)

            # Save draft model state
            draft_model_state = {
                k: v
                for k, v in draft_model.state_dict().items()
                if "embed" not in k.lower()
            }

            # Save training state
            training_state = {
                "optimizer_state_dict": draft_optimizer.state_dict(),
                "step": step,
            }

            # Save both model and training state
            draft_model.save_pretrained(epoch_output_dir, state_dict=draft_model_state)
            torch.save(
                training_state, os.path.join(epoch_output_dir, "training_state.pt")
            )
            print(f"Saved model checkpoint to {epoch_output_dir}")

            # clear the generation cache
            generation_cache = []

            url = f"http://localhost:30000/update_weights_from_disk"
            data = {"speculative_draft_model_path": epoch_output_dir}
            response = requests.post(url, json=data)

            print(response.text)
