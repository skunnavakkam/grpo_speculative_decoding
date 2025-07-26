import vllm
import time
import matplotlib.pyplot as plt


def main():
    PROMPT = [{"role": "user", "content": "What is the capital of France?"}]

    # Non-speculative
    print("STARTING NON-SPECULATIVE")
    start_time = time.time()

    model_name = "Qwen/Qwen3-4B"
    model = vllm.LLM(
        model=model_name,
        tokenizer=model_name,
        max_seq_len=1024,
    )

    results = model.generate(PROMPT, temperature=0.0, max_tokens=100)
    print(results[0].outputs[0].text)
    end_time = time.time()
    print(f"Non-speculative time: {end_time - start_time} seconds")

    model.reset()

    # Speculative
    print("STARTING SPECULATIVE")
    start_time = time.time()
    model_name = "Qwen/Qwen3-4B"
    model = vllm.LLM(
        model=model_name,
        tokenizer=model_name,
        max_seq_len=1024,
        speculative_config={
            "model": "Qwen/Qwen3-0.6B",
            "num_speculative_tokens": 5,
        },
    )
    sampling_params = vllm.SamplingParams(temperature=0.0, max_tokens=100)
    results = model.generate(PROMPT, sampling_params)
    print(results[0].outputs[0].text)
    end_time = time.time()
    print(f"Speculative time: {end_time - start_time} seconds")

    # plot the results as a bar chart
    plt.bar(
        ["Non-speculative", "Speculative"],
        [end_time - start_time, end_time - start_time],
    )
    plt.ylabel("Time (seconds)")
    plt.title("Speculative vs Non-speculative")
    plt.savefig("speculative_vs_non_speculative.png")


if __name__ == "__main__":
    main()
