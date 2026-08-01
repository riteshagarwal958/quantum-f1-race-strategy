import os
import pandas as pd

DQN_REWARDS_CSV = "results/dqn_results_test.csv"
DQN_ACTIONS_CSV = "results/dqn_action_analysis_test.csv"
QRL_REWARDS_CSV = "qrl_logs/test_2025/qrl_reward_comparison.csv"
QRL_ACTIONS_CSV = "qrl_logs/test_2025/qrl_action_breakdown.csv"

OUT_DIR = "results"


def _normalize_keys(df):
    """Make track/year/driver mergeable across the two CSV sources
    (DQN's `year` column is an int, QRL's is a string from
    _split_track_year, so both get cast to str; track/driver are
    lowercased+stripped in case of casing differences)."""
    df = df.copy()
    df["track"] = df["track"].astype(str).str.strip()
    df["year"] = df["year"].astype(str).str.strip()
    df["driver"] = df["driver"].astype(str).str.strip()
    return df


def load_and_merge():
    for path in (DQN_REWARDS_CSV, QRL_REWARDS_CSV):
        if not os.path.exists(path):
            raise SystemExit(
                f"Missing: {path}\n"
                "Run main_dqn.py and main_qrl.py (both include a test-year "
                "evaluation step) before this script."
            )

    dqn_r = _normalize_keys(pd.read_csv(DQN_REWARDS_CSV))
    qrl_r = _normalize_keys(pd.read_csv(QRL_REWARDS_CSV))

    rewards = dqn_r.merge(
        qrl_r[["track", "year", "driver", "random_reward", "real_strategy_reward", "qrl_reward"]],
        on=["track", "year", "driver"],
        suffixes=("_dqn_eval", "_qrl_eval"),
        how="outer",
        indicator=True,
    )

    if (rewards["_merge"] != "both").any():
        unmatched = rewards.loc[rewards["_merge"] != "both", ["track", "year", "driver", "_merge"]]
        print("WARNING: some (track, year, driver) rows didn't match between "
              "the two CSVs -- check for typos/missing scenarios:")
        print(unmatched.to_string(index=False))
    rewards = rewards.drop(columns=["_merge"])

    rewards["baseline_agreement_diff"] = (
        rewards["real_strategy_reward_dqn_eval"] - rewards["real_strategy_reward_qrl_eval"]
    ).round(3)
    rewards["qrl_minus_dqn"] = (rewards["qrl_reward"] - rewards["dqn_reward"]).round(3)
    rewards["qrl_vs_dqn_pct"] = (
        (rewards["qrl_reward"] - rewards["dqn_reward"]) / rewards["dqn_reward"].abs() * 100
    ).round(1)

    actions = None
    if os.path.exists(DQN_ACTIONS_CSV) and os.path.exists(QRL_ACTIONS_CSV):
        dqn_a = _normalize_keys(pd.read_csv(DQN_ACTIONS_CSV))
        qrl_a = _normalize_keys(pd.read_csv(QRL_ACTIONS_CSV))
        actions = dqn_a.merge(
            qrl_a, on=["track", "year", "driver"], suffixes=("_dqn", "_qrl"), how="outer"
        )
    else:
        print("NOTE: action-breakdown CSVs not both found -- skipping action comparison.\n"
              "(If you just fixed the evaluate_qrl.py action_rows bug, re-run main_qrl.py "
              "first so qrl_action_breakdown.csv reflects the QRL agent's own actions.)")

    return rewards, actions


def summarize(rewards):
    lines = []
    lines.append("=" * 60)
    lines.append("DQN vs QRL -- Held-out Test Year Comparison")
    lines.append("=" * 60)
    lines.append(rewards[[
        "track", "year", "driver", "random_reward_dqn_eval", "real_strategy_reward_dqn_eval",
        "dqn_reward", "qrl_reward", "qrl_minus_dqn", "qrl_vs_dqn_pct",
    ]].to_string(index=False))
    lines.append("")
    n_qrl_wins = (rewards["qrl_minus_dqn"] > 0).sum()
    n_dqn_wins = (rewards["qrl_minus_dqn"] < 0).sum()
    lines.append(f"QRL beat DQN on {n_qrl_wins}/{len(rewards)} scenarios; "
                 f"DQN beat QRL on {n_dqn_wins}/{len(rewards)}.")
    lines.append(f"Mean reward -- DQN: {rewards['dqn_reward'].mean():.2f}  "
                 f"QRL: {rewards['qrl_reward'].mean():.2f}  "
                 f"Real strategy: {rewards['real_strategy_reward_dqn_eval'].mean():.2f}  "
                 f"Random: {rewards['random_reward_dqn_eval'].mean():.2f}")
    max_baseline_drift = rewards["baseline_agreement_diff"].abs().max()
    if max_baseline_drift > 0.5:
        lines.append(f"\nNOTE: real_strategy_reward differs by up to "
                     f"{max_baseline_drift:.2f} between the DQN-side and QRL-side "
                     f"eval runs for the same scenario -- expected to be ~identical "
                     f"since it's a deterministic replay of the historical strategy "
                     f"in the same env. Worth checking if this is large.")
    return "\n".join(lines)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rewards, actions = load_and_merge()

    rewards.to_csv(os.path.join(OUT_DIR, "agent_comparison_rewards.csv"), index=False)
    print(f"Saved -> {OUT_DIR}/agent_comparison_rewards.csv")

    if actions is not None:
        actions.to_csv(os.path.join(OUT_DIR, "agent_comparison_actions.csv"), index=False)
        print(f"Saved -> {OUT_DIR}/agent_comparison_actions.csv")

    summary = summarize(rewards)
    print("\n" + summary)
    with open(os.path.join(OUT_DIR, "agent_comparison_summary.txt"), "w") as f:
        f.write(summary + "\n")
    print(f"\nSaved -> {OUT_DIR}/agent_comparison_summary.txt")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        labels = rewards["track"] + " " + rewards["year"]
        x = range(len(rewards))
        width = 0.2
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar([i - 1.5 * width for i in x], rewards["random_reward_dqn_eval"], width, label="Random")
        ax.bar([i - 0.5 * width for i in x], rewards["real_strategy_reward_dqn_eval"], width, label="Real strategy")
        ax.bar([i + 0.5 * width for i in x], rewards["dqn_reward"], width, label="DQN")
        ax.bar([i + 1.5 * width for i in x], rewards["qrl_reward"], width, label="QRL")
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel("Episode reward")
        ax.set_title("DQN vs QRL vs baselines (held-out test year)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "agent_comparison_rewards.png"), dpi=150)
        print(f"Saved -> {OUT_DIR}/agent_comparison_rewards.png")
    except ImportError:
        print("(matplotlib not installed -- skipping chart; CSVs/summary are still saved)")


if __name__ == "__main__":
    main()
