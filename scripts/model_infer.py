from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import torch

model_id = "checkpoints_merged/rl_thinking_mix"

tokenizer = AutoTokenizer.from_pretrained(model_id)
config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    config=config,
    torch_dtype=torch.bfloat16,
    #attn_implementation="flash_attention_2",
    trust_remote_code=True
)

print(model.device)

messages = [
    {"role": "user", "content": "Below is a description of a bulk material. The chemical formula is TiMoB2. The bulk_modulus is in [150, 300]. The shear_modulus is greater or equal than 200. Generate a description of the lengths and angles of the lattice vectors and then the element type and coordinates for each atom within the lattice:"},
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(text)
model_inputs = tokenizer(text, return_tensors="pt").to(model.device)

generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=2048
)
generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(generated_text)