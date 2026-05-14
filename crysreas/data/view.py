import shelve
import tyro
import torch
from dataclasses import dataclass
from typing import List, Dict, Any
from crysreas import Config
from .prompt_generator import get_info
from verl.utils import hf_tokenizer
from crysreas.utils.crystal import SimpleCrystal

@dataclass
class Args:
    """Arguments for viewing material structure data."""
    db_path: str = str(Config.DATA_PATH / "MP_shelve")

def main(args: Args):
    """Main entry point for interactive material structure viewing."""
    print("--- Running View Structures Function ---")
    view_structures(args.db_path)

def view_structures(db_path: str):
    """Interactive shell to view material data stored in the shelve database."""
    tokenizer = hf_tokenizer("MegaScience/Qwen2.5-3B-MegaScience", trust_remote_code=True)
    try:
        # 1. Open the shelve database (read-only)
        with shelve.open(db_path, 'r') as db:
            mp_ids: List[str] = list(db.keys())
            count = len(mp_ids)

            if not mp_ids:
                print(f"No material data found in the database '{db_path}'.")
                return

            print(f"The database '{db_path}' contains a total of {count} materials.")
            print("---")
            print("Available Material IDs:")
            if len(mp_ids) > 20:
                display_ids = mp_ids[:20]
                print(", ".join(display_ids) + ", ... (omitted)")
            else:
                print(", ".join(mp_ids))
            print("---")
            print("Commands: Enter ID, 'all' for summary, 'q' to quit. Prefix ID with 'd' for full prompt description.")

            # 2. Interactive query loop
            while True:
                user_input = input(">> Enter ID / 'all' / 'q': ").strip().lower()

                if user_input == 'q':
                    print("Exiting view mode.")
                    break

                elif user_input == 'all':
                    print("\n--- Summary of All Materials ---")
                    for mp_id in mp_ids:
                        doc_data: Dict[str, Any] = db[mp_id]
                        title_info = f"ID: {mp_id}, E_above_hull: {doc_data.get('energy_above_hull', 'N/A'):.3f} eV/atom"
                        print(f"**{title_info}**")
                        print(f"  Stability: {'Stable' if doc_data.get('is_stable') else 'Unstable'}")
                        print(f"  Band Gap: {doc_data.get('band_gap', 'N/A'):.3f} eV, {'Metallic' if doc_data.get('is_metal') else 'Non-metallic'}")
                        desc = doc_data.get("robocrys_description", "N/A")
                        print(f"  Structure Description (Robocrys): {desc[:100]}...")
                    print("--------------------------------\n")

                elif user_input in mp_ids:
                    print(f"\n--- Material Details: {user_input} ---")
                    doc_data: Dict[str, Any] = db[user_input]
                    for key, value in doc_data.items():
                        print(f"  {key:<25}: {value}")
                    print(SimpleCrystal(doc_data["structure"]).to_simple_no_sym())
                    print("----------------------------------\n")

                elif user_input.startswith('d') and user_input[1:] in mp_ids:
                    mp_id = user_input[1:]
                    print(f"\n--- Material Description: {mp_id} ---")
                    doc_data: Dict[str, Any] = db[mp_id]
                    frame = get_info(doc_data, "conditional+thinking")
                    prompt = frame["question"]
                    response = frame["answer"]
                    prompt_chat = [{"role": "user", "content": prompt}]
                    prompt_chat_str = tokenizer.apply_chat_template(
                        prompt_chat, add_generation_prompt=True, tokenize=False
                    )
                    response_chat_str = response + tokenizer.eos_token
                    print(prompt_chat_str + response_chat_str)

                    prompt_ids_output = tokenizer(prompt_chat_str, return_tensors="pt", add_special_tokens=False)
                    prompt_ids = prompt_ids_output["input_ids"][0]
                    response_ids_output = tokenizer(response_chat_str, return_tensors="pt", add_special_tokens=False)
                    response_ids = response_ids_output["input_ids"][0]
                    input_ids = torch.cat((prompt_ids, response_ids), dim=-1)
                    print("Token Count:", input_ids.shape[0])

                else:
                    print(f"⚠️ Unrecognized input or ID '{user_input}' not in database.")

    except FileNotFoundError:
        print(f"❌ Error: Database file '{db_path}' not found.")
    except Exception as e:
        print(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    tyro.cli(main)