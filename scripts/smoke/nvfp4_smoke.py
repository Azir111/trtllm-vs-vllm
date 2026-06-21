from tensorrt_llm import LLM, SamplingParams


def main():
    llm = LLM(
        model="/workspace/models/llama31-8b-nvfp4",
        backend="pytorch",
        attn_backend="FLASHINFER",
        tensor_parallel_size=1,
    )
    prompts = ["Hello, my name is", "The capital of France is", "The future of AI is"]
    params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=64)

    for out in llm.generate(prompts, params):
        print(f"{out.prompt!r} -> {out.outputs[0].text!r}")


if __name__ == "__main__":
    main()
