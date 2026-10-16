"""Minimal vLLM smoke test: load the model and sample one completion."""
from vllm import LLM, SamplingParams
llm = LLM(model="checkpoints_merged/rl_thinking_mix", dtype="bfloat16",
          gpu_memory_utilization=0.6, enforce_eager=True, max_model_len=4096)
tok = llm.get_tokenizer()
q = ("Below is a description of a bulk material. The chemical formula is Rb2SnCl6. "
     "The spacegroup number is 225. Generate a description of the lengths and angles of "
     "the lattice vectors and then the element type and coordinates for each atom within "
     "the lattice:")
text = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                               add_generation_prompt=True)
out = llm.generate([text], SamplingParams(temperature=1.0, top_p=0.7, top_k=50,
                                          max_tokens=3584, n=2))
for i, o in enumerate(out[0].outputs):
    print(f"--- sample {i} ({len(o.token_ids)} tokens) ---")
    print(o.text[-600:])
print("SMOKE OK")
