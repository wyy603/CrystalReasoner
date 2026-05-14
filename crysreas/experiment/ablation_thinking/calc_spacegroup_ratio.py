from pathlib import Path

import pandas as pd


def compute_spacegroup_ratio(
    input_path: Path,
    output_path: Path,
) -> None:
    """Compute true ratio of spacegroup_consistency for each (mp_id, variant)."""
    df = pd.read_parquet(input_path)
    prompt_has_spacegroup = (
        df["original_prompt"]
        .astype(str)
        .str.contains(r"space\s*group", case=False, regex=True, na=False)
    )
    df = df[prompt_has_spacegroup]

    # Normalize values to boolean True/False in a robust way.
    col = df["spacegroup_consistency"]
    if pd.api.types.is_bool_dtype(col):
        is_true = col.fillna(False)
    else:
        is_true = col.astype(str).str.strip().str.lower().eq("true")

    result = (
        df.assign(_is_true=is_true)
        .groupby(["mp_id", "variant"], as_index=False, sort=False)["_is_true"]
        .mean()
        .rename(columns={"_is_true": "spacegroup_ratio"})
    )

    result.to_csv(output_path, index=False)
    print(f"Wrote {len(result)} rows to {output_path}")


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    input_file = base_dir / "ablation_results.parquet"
    output_file = base_dir / "ablation_spacegroup_ratio.csv"
    compute_spacegroup_ratio(input_file, output_file)
