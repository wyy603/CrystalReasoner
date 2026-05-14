import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

_EXPERIMENT_DIR = Path(__file__).resolve().parent
_OUTPUT_DIR = _EXPERIMENT_DIR

binary_metrics_list = [
    'structure_validity', 'smact_validity', 'composition_consistency',
    'is_stable', 'is_novel', 'is_unique', 'stable_unique_novel', 
    'spacegroup_consistency'
]

def main():
    # compare to prior methods
    full_metrics = [
        'structure_validity', 'smact_validity', 'composition_consistency',
        'is_stable', 'is_novel', 'is_unique', 'stable_unique_novel', 
        'spacegroup_consistency', 'Energy', 'elastic_reward_all'
    ]
    full_comparison_files = {
        "With_Thinking": "checkpoints_merged/thinking/conditional+thinking.parquet",
        "RL_With_Thinking": "checkpoints_merged/rl_thinking_mix/conditional+thinking.parquet",
        "No_Thinking": "checkpoints_merged/no_thinking/conditional+thinking.parquet",
        "plaid_wyckoff": "checkpoints_merged/plaid_wyckoff/plaid_wyckoff_generation+no_thinking.parquet",
        "crystaltextllm": "checkpoints_merged/crystaltextllm/crystaltextllm_generation+no_thinking.parquet",
        "plaid_wyckoff_8": "checkpoints_merged/plaid_wyckoff_8/plaid_wyckoff_8_generation+no_thinking.parquet",
        "crystaltextllm_8": "checkpoints_merged/crystaltextllm_8/crystaltextllm_8_generation+no_thinking.parquet",
        #"RL_No_Thinking": "checkpoints_merged/rl_no_thinking_mix/conditional+thinking.parquet",
    }
    run_comparison_job(
        job_name="Full Comparison",
        file_paths=full_comparison_files,
        metrics=full_metrics,
        output_csv="model_comparison_metrics.csv",
        output_pdf="energy_distribution_curve.pdf",
        clip_l=-1,
        clip_r=10
    )

    #full comparison
    # full_metrics = [
    #     'structure_validity', 'smact_validity', 'composition_consistency',
    #     'is_stable', 'is_novel', 'is_unique', 'stable_unique_novel', 
    #     'spacegroup_consistency', 'Energy', 'elastic_reward_all'
    # ]
    # full_comparison_files = {
    #     #"With_Thinking": "checkpoints_merged/thinking/conditional+thinking.parquet",
    #     "RL_With_Thinking": "checkpoints_merged/rl_thinking_mix/conditional+thinking.parquet",
    #     #"No_Thinking": "checkpoints_merged/no_thinking/conditional+thinking.parquet",
    #     "RL_No_Thinking": "checkpoints_merged/rl_no_thinking_mix/conditional+thinking.parquet",
    # }
    # run_comparison_job(
    #     job_name="Full Comparison",
    #     file_paths=full_comparison_files,
    #     metrics=full_metrics,
    #     output_csv="model_comparison_metrics.csv",
    #     output_pdf="energy_distribution_curve.pdf",
    #     clip_l=-1,
    #     clip_r=10
    # )

    # rl thinking novel
    # full_metrics = [
    #     'structure_validity', 'smact_validity', 'composition_consistency',
    #     'is_stable', 'is_novel', 'is_unique', 'stable_unique_novel', 
    #     'spacegroup_consistency', 'Energy', 'elastic_reward_all'
    # ]
    # full_comparison_files = {
    #     "RL_With_Thinking": "checkpoints_merged/rl_thinking_mix/conditional+thinking.parquet",
    #     "RL_Thinking_Novel": "checkpoints_merged/rl_thinking_novel/conditional+thinking.parquet",
    # }
    # run_comparison_job(
    #     job_name="Full Comparison",
    #     file_paths=full_comparison_files,
    #     metrics=full_metrics,
    #     output_csv=str(_EXPERIMENT_DIR / "model_comparison_metrics_novel.csv"),
    #     output_pdf=str(_EXPERIMENT_DIR / "energy_distribution_curve_novel.pdf"),
    #     clip_l=-1,
    #     clip_r=10
    # )

    # # different rewards
    # different_rewards = {
    #     "SFTed": "checkpoints_merged/thinking/conditional+thinking.parquet",
    #     "Validity Only": "checkpoints_merged/thinking_only_validity/conditional+thinking.parquet",
    #     "Energy Only": "checkpoints_merged/thinking_only_energy/conditional+thinking.parquet",
    #     "Mixed": "checkpoints_merged/rl_thinking_mix/conditional+thinking.parquet",
    # }
    # run_comparison_job(
    #     job_name="Different Rewards Comparison",
    #     file_paths=different_rewards,
    #     metrics=full_metrics,
    #     output_csv="different_rewards_comparison.csv",
    #     clip_l=-1,
    #     clip_r=10
    # )

    # # spacegroup
    # spacegroup_files = {
    #     "Spacegroup_Thinking": "checkpoints_merged/spacegroup_thinking/spacegroup+thinking.parquet",
    #     "RL_Thinking_Mix": "checkpoints_merged/rl_thinking_mix/spacegroup+thinking.parquet",
    # }
    # run_comparison_job(
    #     job_name="Spacegroup Comparison",
    #     file_paths=spacegroup_files,
    #     metrics=["spacegroup_consistency"],
    #     output_csv="spacegroup_consistency_specific_comparison.csv"
    # )

    # elastic
    # elastic_files = {
    #     "rl_thinking": "checkpoints_merged/rl_thinking_mix/elastic+thinking.parquet",
    #     "elastic_reward": "checkpoints_merged/rl_elastic_thinking_new/elastic+thinking.parquet",
    # }
    # run_comparison_job(
    #     job_name="Elastic Comparison",
    #     file_paths=elastic_files,
    #     metrics=["structure_validity", "smact_validity", "composition_consistency", "elastic_reward_all", "is_stable","is_unique","is_novel","stable_unique_novel"],
    #     output_csv=str(_EXPERIMENT_DIR / "elastic_reward_comparison.csv")
    # )

    # cte
    # elastic_files = {
    #     "rl_thinking": "checkpoints_merged/rl_thinking_mix/cte+thinking.parquet",
    #     "cte_reward": "checkpoints_merged/rl_cte_thinking/cte+thinking.parquet",
    # }
    # run_comparison_job(
    #     job_name="Thermal Expansion Comparison",
    #     file_paths=elastic_files,
    #     metrics=["structure_validity", "smact_validity", "cte_reward_all"],
    #     output_csv=str(_EXPERIMENT_DIR / "cte_reward_comparison.csv")
    # )

def run_comparison_job(
    job_name: str, 
    file_paths: Dict[str, str], 
    metrics: List[str], 
    output_csv: str, 
    output_pdf: Optional[str] = None, 
    clip_l: float = -1, 
    clip_r: float = 10
):
    """Run a specific comparison job."""
    print(f"\nRunning Job: {job_name}...")
    output_csv = _resolve_output_path(output_csv)
    output_pdf = _resolve_output_path(output_pdf) if output_pdf is not None else None
    
    # 1. Process data
    res_df, plot_df = process_datasets(file_paths, metrics, clip_l, clip_r)
    
    # 2. Save summary statistics
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(output_csv, index=False)
    print(f"Saved: {output_csv}")

    # 3. Plotting (only if output_pdf is provided and Energy was processed)
    if output_pdf and not plot_df.empty:
        plot_results(plot_df, list(file_paths.keys()), output_pdf)

def compute(df: pd.DataFrame, key: str, group_key="mp_id", mode="bool") -> str:
    """Compute mean and standard deviation percentage for boolean metrics."""
    if key not in df.columns:
        return "N/A"
    
    df_temp = df[[group_key, key]].copy()
    if mode == "bool":
        df_temp[key] = (df_temp[key] == True).astype(float)
    elif mode == "float":
        df_temp[key] = df_temp[key].astype(float)
    elif mode == "1":
        df_temp[key] = (df_temp[key] == 1).astype(float)
    elif mode == "2":
        df_temp[key] = (df_temp[key] == 2).astype(float)

    final_mean = df_temp.groupby(group_key)[key].mean().mean()
    
    group_stats = df_temp.groupby(group_key)[key].agg(['var', 'count'])
    group_stats['var'] = group_stats['var'].fillna(0)
    group_stats['mean_var_contribution'] = group_stats['var'] / group_stats['count']
    
    M = len(group_stats)
    total_var_s = group_stats['mean_var_contribution'].sum() / (M**2)
    sigma_s = np.sqrt(total_var_s)
    
    if mode == "float":
        return f"{final_mean:.2f} ± {sigma_s:.4f}"
    else:
        return f"{final_mean * 100:.2f}% ± {sigma_s * 100:.2f}%"

def _resolve_output_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return _OUTPUT_DIR / path

def process_datasets(file_paths: Dict[str, str], metrics: List[str], clip_l: float, clip_r: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Process datasets and generate summary statistics."""
    summary_stats = []
    all_valid_energies = []

    print("Processing datasets...")
    for name, path in file_paths.items():
        df = pd.read_parquet(path)
        total_count = len(df)
        
        stat = {"model name": name}
        for metric in metrics:
            if metric in binary_metrics_list:
                if metric in df.columns:
                    stat[metric] = compute(df, metric)
                else:
                    stat[metric] = "N/A"
            elif metric == "Energy":
                if 'energy_above_hull' in df.columns:
                    df_energy = df[
                        (~df['energy_above_hull'].isna()) & 
                        (df['energy_above_hull'] >= clip_l) & 
                        (df['energy_above_hull'] <= clip_r)
                    ]
                    stat["Energy"] = compute(df_energy, 'energy_above_hull', mode='float')
                    
                    energies = df_energy['energy_above_hull'].to_numpy()
                    temp_df = pd.DataFrame({
                        'Energy': energies,
                        'Model': name,
                        'Weight': 1.0 / total_count 
                    })
                    all_valid_energies.append(temp_df)
                else:
                    stat["Energy"] = "N/A"
            elif metric == 'elastic_reward_all':
                if metric in df.columns:
                    stat[metric] = compute(df, metric, mode='2')
                else:
                    stat[metric] = "N/A"
            elif metric == 'cte_reward_all':
                if metric in df.columns:
                    stat[metric] = compute(df, metric, mode='1')
                else:
                    stat[metric] = "N/A"
        
        summary_stats.append(stat)

    res_df = pd.DataFrame(summary_stats)
    plot_df = pd.concat(all_valid_energies) if all_valid_energies else pd.DataFrame()
    return res_df, plot_df

def plot_results(plot_df: pd.DataFrame, model_names: List[str], output_pdf: str):
    """Plot KDE distribution curves for Energy."""
    plt.figure(figsize=(12, 7), dpi=300)
    sns.set_style("whitegrid", {"grid.linestyle": "--"})

    colors = sns.color_palette("Set1", n_colors=len(model_names))
    sns.kdeplot(
        data=plot_df,
        x='Energy',
        hue='Model',
        weights='Weight',
        fill=True,
        alpha=0.2,
        linewidth=2.5,
        common_norm=False,
        palette=colors,
    )

    plt.title('Energy Above Hull Distribution (Normalized by Total Samples)', fontsize=15, pad=20)
    plt.xlabel('Energy Above Hull (eV/atom)', fontsize=12)
    plt.ylabel('Adjusted Density (Fraction of Total)', fontsize=12)

    x_min = plot_df['Energy'].quantile(0.001)
    x_max = plot_df['Energy'].quantile(0.99)
    plt.xlim(min(x_min, -0.05), x_max * 1.2) 

    plt.tight_layout()
    Path(output_pdf).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_pdf)
    print(f"Saved: {output_pdf}")

if __name__ == "__main__":
    main()
