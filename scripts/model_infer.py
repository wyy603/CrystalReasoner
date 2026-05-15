from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import torch

model_id = "checkpoints_merged/rl_thinking_mix"

tokenizer = AutoTokenizer.from_pretrained(model_id)
config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    config=config,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

messages = [
    {"role": "user", "content": "Below is a description of a bulk material. The chemical formula is NaCl. The bulk_modulus is about 100 GPa. Generate a description of the lengths and angles of the lattice vectors and then the element type and coordinates for each atom within the lattice:"},
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
model_inputs = tokenizer(text, return_tensors="pt").to(model.device)

generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=2048,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
    use_cache=True,
)
generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]
print(generated_text)

def get_structure(generated_text: str):
    import re
    from pymatgen.core import Lattice, Structure

    cif_match = re.search(r'<CIF>(.*?)</CIF>', generated_text, re.DOTALL)
    if cif_match:
        generated_text = cif_match.group(1)

    lines = [line.strip() for line in generated_text.strip().split('\n') if line.strip()]
    if lines and not re.match(r'^[-+0-9.eE\s]+$', lines[0]):
        lines = lines[1:]

    lengths = list(map(float, lines[0].split()))
    angles = list(map(float, lines[1].split()))
    lattice = Lattice.from_parameters(*lengths, *angles)

    species = []
    coords = []
    for line in lines[2:]:
        parts = line.split()
        species.append(parts[0])
        coords.append([float(parts[2]), float(parts[3]), float(parts[4])])

    structure = Structure(lattice, species, coords)
    return structure

print(get_structure(generated_text))
