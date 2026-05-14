export FORCE_CUDA=1
export MAX_JOBS=8

uv venv .venv --python 3.12
source .venv/bin/activate

uv pip install pip
uv pip install -e .
uv pip install -r requirements.txt
uv pip install ninja
bash submodules/verl/scripts/install_vllm_sglang_mcore.sh

uv pip install -e "submodules/torch-sim[mace,test]"

uv pip install torch-cluster -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
uv pip install torch-scatter -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
uv pip install torch-sparse -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
uv pip install -e submodules/mattergen

uv pip install e3nn==0.4.4
uv pip install mattersim==1.2.0
uv pip install "numpy<2.2"
uv pip install "ase==3.25.0"

cd ../apex
APEX_CPP_EXT=1 APEX_CUDA_EXT=1 pip install -v --no-build-isolation .