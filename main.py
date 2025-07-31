from vllm import LLM, SamplingParams
from datasets import load_dataset
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import Adam
from eagle.model.ea_model import EaModel
import torch.nn.functional as F

NUM_STEPS = 5
BASE_MODEL = "Qwen/Qwen3-4B"
SPECULATIVE_MODEL = "AngelSlim/Qwen3-4B_eagle3"
DATASET = "trl-lib/tldr"
FINETUNE_STEPS = 5
NUM_GENERATIONS = 4

SYSTEM_PROMPT = "Summarize the following text in about 100 characters."


def reward_fn(completions, **kwargs):
    return [-abs(len(completion.outputs[0].text) - 100) for completion in completions]


# We have two GPUs. One for inference, one for updates. We store the vLLM model on cuda:0 and the other model on cuda:1.

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main():
    # Initialize vLLM engine
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

    # Load HuggingFace reference model for policy updates
    device = torch.device("cuda:1")
    ref_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map={"": device},
        trust_remote_code=True,
    )
    ref_tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
    )

    ref_eagle = EaModel(
        base_model=ref_model,
        base_model_name_or_path=BASE_MODEL,
        ea_model_path=SPECULATIVE_MODEL,
    )

    # Prepare dataset and prompts
    dataset = load_dataset(DATASET)
    dataset = dataset.select(range(1000))
    prompts = [f"{SYSTEM_PROMPT}\n{item['prompt']}" for item in dataset["train"]]

    # Optimizer for reference model
    optimizer = Adam(ref_model.parameters(), lr=1e-5)
    eagle_optimizer = Adam(ref_eagle.draft_model.parameters(), lr=1e-4)

    eps = 1e-8
    cached_rollouts = []

    for idx in range(NUM_STEPS):
        # Tokenize prompt
        encoding = ref_tokenizer(prompts[idx], return_tensors="pt")
        prompt_ids = encoding.input_ids[0].tolist()

        # Decode back to text for logging (optional)
        prompt_text = ref_tokenizer.decode(prompt_ids, skip_special_tokens=True)

        # Generate with vLLM using both text and token IDs
        outputs = llm.generate(
            prompts=[prompt_text],
            prompt_token_ids=[prompt_ids],
            sampling_params=SamplingParams(
                temperature=0.7,
                max_tokens=100,
                top_p=0.95,
                seed=idx,
            ),
            request_id=[str(idx)],
        )

        # Extract only the new token IDs per completion
        completions = []
        for out in outputs:
            rollout_ids = out.token_ids[len(out.prompt_tokens) :]
            completions.append((out.prompt_tokens, rollout_ids))

        cached_rollouts.extend(completions)

        # Compute rewards and advantages
        rewards = torch.tensor(reward_fn([r for _, r in completions]))
        mean, std = rewards.mean(), rewards.std()
        advantages = (rewards - mean) / (std + eps)

        # Prepare inputs for policy gradient
        inputs, labels = [], []
        for (p_ids, r_ids), adv in zip(completions, advantages):
            inputs.append(torch.tensor(p_ids + r_ids, device=device))
            labels.append(torch.tensor([-100] * len(p_ids) + r_ids, device=device))

        padded_inputs = torch.nn.utils.rnn.pad_sequence(
            inputs, batch_first=True, padding_value=ref_tokenizer.pad_token_id
        )
        padded_labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )

        # Compute log-prob with reference model
        out = ref_model(input_ids=padded_inputs, labels=padded_labels)
        log_prob = -out.loss
        pg_loss = -(advantages.to(log_prob.device) * log_prob).mean()

        optimizer.zero_grad()
        pg_loss.backward()
        optimizer.step()

        # Sync updated weights back into vLLM engine
        with torch.no_grad():  # no need for autograd here
            new_weights = [
                (n, p.data.to("cuda:0", non_blocking=True))  # copy directly to GPU0
                for n, p in ref_model.named_parameters()
            ]
        llm.model_runner.model.load_weights(new_weights)

        # Placeholder for speculative fine-tuning
        if idx % FINETUNE_STEPS == 0 and idx > 0 and cached_rollouts:
            ref_eagle.draft_model.train()
            total_loss = 0.0
            for p_ids, r_ids in cached_rollouts:
                # build a single sequence [prompt + completion]
                seq_ids = torch.tensor([p_ids + r_ids], device=device)
                attn_mask = (seq_ids != ref_tokenizer.pad_token_id).long()

                # get the true hidden states from your updated ref_model
                out = ref_model(
                    input_ids=seq_ids,
                    attention_mask=attn_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )
                # second-to-last layer features: (batch, seq_len, hidden_dim)
                features = out.hidden_states[-2].squeeze(0)

                # inputs to draft head: all but last token’s features
                feat_in = features[:-1].unsqueeze(0)  # (1, seq_len-1, hid)
                # targets: next-step features
                feat_tgt = features[1:].unsqueeze(0)  # (1, seq_len-1, hid)

                # draft_model returns (logits, …) or just preds depending on version
                pred_features = ref_eagle.draft_model(feat_in)[0]

                loss = F.mse_loss(pred_features, feat_tgt)
                total_loss += loss

            total_loss = total_loss / len(cached_rollouts)
            eagle_optimizer.zero_grad()
            total_loss.backward()
            eagle_optimizer.step()
            ref_eagle.draft_model.eval()

            # now sync the updated draft head into vLLM’s speculative model
            new_ea_weights = [
                (n, p.data.to("cuda:0", non_blocking=True))
                for n, p in ref_eagle.draft_model.named_parameters()
            ]
            llm.model_runner.speculative_model.load_weights(new_ea_weights)

            # clear cache
            cached_rollouts = []


if __name__ == "__main__":
    main()
