# Database

MP_shelve is downloaded by `crysreas/data/download.py`, and modified by scripts in 

MLIP predicted values are inserted by `crysreas/data/mlip/insert_properties.py`.

# Splits

split_cdvae.json is copied from ``assets/cdvae/split.json``.

Other jsons are calculated by ``crysreas/data/generate_split.py``.

Parquets are calculated by ``crysreas/metric_process``, they contain the values predicted by MLIP.

The logic of generating these jsons:
- `split_cdvae.json` is used for SFT, `split_rl.json` is used for RL, `split_elastic.json` for elasticity conditioned generation, `split_cte.json` for thermal expansion conditioned generation. For RL related jsons, their test sets are small.
- We first evaluate the structures and put the predicted values in `xxx.parquet`, then select those pass the MLIP test and put in the new json file.
- `xxx_generation.json` is for generation, the size of test set is larger than when training.