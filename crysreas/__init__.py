import os
import warnings
from dotenv import load_dotenv, find_dotenv
from pathlib import Path

def load_env():
    env_path = find_dotenv()
    load_dotenv(env_path)

load_env()

if os.getenv("AI4SCI_SUPPRESS_MATTERGEN_WARNINGS", "1") == "1":
    warnings.filterwarnings("ignore", category=Warning, module=r"mattergen(\.|$)")

def find_project_root(marker_filename: str = 'pyproject.toml') -> Path:
    current_dir = Path(__file__).resolve().parent
    
    for parent in [current_dir] + list(current_dir.parents):
        if (parent / marker_filename).exists():
            return parent
            
    raise FileNotFoundError(f"Project root marker '{marker_filename}' not found. Cannot determine base path.")

class Config:
    """Stores all application configuration paths."""
    
    try:
        PROJECT_ROOT: Path = find_project_root('pyproject.toml')
    except FileNotFoundError as e:
        print(f"Warning: {e}")
        PROJECT_ROOT: Path = Path(os.getcwd())
    
    DATA_PATH: Path = PROJECT_ROOT / 'assets' / 'MP'
    CHECKPOINT_PATH: Path = PROJECT_ROOT / 'checkpoints' / 'latest'
    VERL_ROOT_PATH: Path = PROJECT_ROOT / "submodules" / "verl"
    CRYSTALTEXTLLM_DATA_PATH: Path = PROJECT_ROOT / "submodules/crystal-text-llm/data/basic"
    DATA_PATH.mkdir(parents=True, exist_ok=True)
    API_KEY = {
        'DEEPSEEK': os.getenv('DEEPSEEK_API_KEY'),
        'GEMINI': os.getenv('GEMINI_API_KEY'),
        'MP': os.getenv('MP_API_KEY')
    }

if __name__ == '__main__':
    print(f"Project Root: {Config.PROJECT_ROOT}")
    print(f"Data Path:    {Config.DATA_PATH}")
    print(f"MP API Key:   {Config.API_KEY['MP']}")
    print(f"Data Path Exists: {Config.DATA_PATH.exists()}")
