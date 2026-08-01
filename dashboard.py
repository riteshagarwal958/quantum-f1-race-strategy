
import glob
import os
import pickle
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    from f1_strategy_env import F1StrategyEnv
    HAS_ENV = True
except ImportError:
    HAS_ENV = False

try:
    from data_loader import STATE_FEATURES
    HAS_DATA_LOADER = True
except ImportError:
    STATE_FEATURES = []
    HAS_DATA_LOADER = False

try:
    from stable_baselines3 import DQN
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False

try:
    import torch
    from main_qrl import (
        QRLActorCritic,
        RunningNormalizer,
        NUM_QUBITS,
        NUM_LAYERS,
        NUM_FEATURES,
        NUM_ACTIONS,
        ENCODER_HIDDEN,
        DEVICE,
    )
    HAS_QRL_DEPS = True
except ImportError:
    HAS_QRL_DEPS = False


# ----------------------------- Constants -----------------------------

TRACKS = ["Silverstone", "Monza", "Monaco"]
DRIVER = "HAM"
TRAIN_YEARS = [2022, 2023, 2024]
TEST_YEARS = [2025]

DATA_DIR = "data"
MODEL_DIR = "models"
RESULTS_DIR = "results"
TB_LOG_DIR = "tensorboard_logs"
QRL_LOG_DIR = "qrl_logs"

COMPOUND_NAMES = {0: "Soft", 1: "Medium", 2: "Hard"}
ACTION_NAMES = {0: "STAY", 1: "PIT_SOFT", 2: "PIT_MEDIUM", 3: "PIT_HARD"}
COMPOUND_TO_ACTION = {0: 1, 1: 2, 2: 3}

# Documentation shown on the "Environment & Reward Design" tab. Kept as plain
# data so it renders even when data_loader.py / f1_strategy_env.py aren't
# importable in the current environment.
STATE_FEATURE_DESCRIPTIONS = {
    "remaining_laps_norm": "Fraction of race laps still remaining (1.0 at race start)",
    "compound_soft": "1.0 if currently on Soft tyres, else 0.0",
    "compound_medium": "1.0 if currently on Medium tyres, else 0.0",
    "compound_hard": "1.0 if currently on Hard tyres, else 0.0",
    "tyre_age_norm": "Tyre age in laps, normalized by MAX_TYRE_LIFE",
    "lap_delta_norm": "Last lap time vs. median race pace, normalized",
    "position_norm": "Race position, normalized so P1 = 1.0 and last place = 0.0",
    "gap_ahead_norm": "Time gap to the car ahead, normalized by GAP_SCALE",
    "gap_behind_norm": "Time gap to the car behind, normalized by GAP_SCALE",
    "gap_to_leader_norm": "Time gap to the race leader, normalized by GAP_SCALE",
    "pits_done_norm": "Pit stops taken so far, normalized by MAX_STOPS",
    "safety_car_active": "1.0 if a safety car is out this lap, else 0.0",
    "deg_penalty_norm": "Current tyre-degradation pace penalty, normalized by MAX_DEG_PENALTY",
}

# Circuit design notes for the QRL policy. These describe the ansatz actually
# implemented in main_qrl.py (RY-RZ-RY rotations, entanglement cycling every
# 3 layers) -- edit here if that design changes.
QUANTUM_CIRCUIT_NOTES = {
    "encoding": "Race-state features are encoded as per-qubit rotation angles ahead of the "
                "variational layers (angle encoding), so every qubit starts carrying real "
                "race-state information rather than a fixed reference state.",
    "ansatz": "RY-RZ-RY three-rotation ansatz on every qubit in every layer, giving each qubit "
              "full single-qubit (SU(2)) expressivity rather than the more limited RY-RZ pattern.",
    "entanglement": "The entangling layer alternates by layer index (layer % 3): a forward chain "
                     "of CNOTs, then a reverse chain, then a full pairwise entangling layer, "
                     "cycling every 3 layers.",
}


PS_TECHNICAL_CONSTRAINTS = [
    "All race data sourced from the FastF1 API -- no synthetic datasets for core benchmarking",
    "VQC policy network implemented with PennyLane (or Qiskit)",
    "Classical DQN baseline trained under identical environment conditions as the QRL agent",
    "Open-source frameworks and tools only; no proprietary APIs or paid services",
    "The system must run on consumer-grade hardware",
    "Quantum circuit design, reward function, and training methodology must be documented",
]

PS_EVALUATION_CRITERIA = [
    "Strategy performance vs. the classical DQN baseline and historical team strategy",
    "Technical correctness of the VQC architecture, circuit design, and environment implementation",
    "Training convergence quality and stability",
    "Accuracy and clarity of the replay visualization",
    "Code quality, modularity, and reproducibility",
    "Depth of analysis and report quality",
]

PS_BONUS_EXTENSIONS = [
    "Interactive dashboard for race selection and replay",
    "Weather-adaptive strategy integration",
    "Multi-race curriculum learning",
    "Multi-agent strategy simulation",
    "Comparative benchmarking across multiple VQC architectures",
]

PS_MANUAL_DELIVERABLES = [
    "Public GitHub repository (source code, setup instructions, training scripts, replay system)",
    "Screen-recorded demo video of a full race simulation",
    "Final technical report (reward curves, convergence plots, sample-efficiency, strategy divergence)",
    "Final presentation slides summarizing methodology and results",
]


# ----------------------------- Cached loaders -----------------------------

@st.cache_data(show_spinner=False)
def load_csv(path):
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def discover_scenarios():
    """Scan data/ for '<track>_<year>_<driver>.pkl' files."""
    if not os.path.isdir(DATA_DIR):
        return []
    entries = []
    for fname in sorted(os.listdir(DATA_DIR)):
        if not fname.endswith(".pkl"):
            continue
        stem = fname[:-4]
        parts = stem.split("_")
        if len(parts) < 3 or not parts[-2].isdigit():
            continue
        driver = parts[-1]
        year = parts[-2]
        track = "_".join(parts[:-2])
        entries.append((track, year, driver, fname))
    return entries


@st.cache_data(show_spinner=False)
def load_scenario_file(path):
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_resource(show_spinner="Loading DQN model...")
def load_dqn_model():
    if not HAS_SB3:
        return None
    stem = os.path.join(MODEL_DIR, "dqn_f1_strategist_final")
    if not os.path.exists(stem + ".zip"):
        return None
    return DQN.load(stem)


@st.cache_resource(show_spinner="Loading QRL model...")
def load_qrl_model():
    if not HAS_QRL_DEPS:
        return None
    path = os.path.join(MODEL_DIR, "qrl_f1_strategist_final.pt")
    if not os.path.exists(path):
        return None
 
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    policy = QRLActorCritic(NUM_QUBITS, NUM_LAYERS, NUM_FEATURES, NUM_ACTIONS, ENCODER_HIDDEN).to(DEVICE)
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.eval()
    normalizer = RunningNormalizer(shape=(NUM_FEATURES,))
    normalizer.mean = checkpoint["norm_mean"]
    normalizer.var = checkpoint["norm_var"]
    return policy, normalizer


# ----------------------------- Episode runner -----------------------------

def real_strategy_fn_factory(scenario):
    """Replays the historical team's actual pit-stop / compound calls."""
    compounds = scenario.actual_compounds
    n_laps = scenario.n_laps
    pit_at_lap = {}
    prev_compound = scenario.start_compound
    for lap in range(1, n_laps + 1):
        comp = compounds.get(lap, prev_compound)
        if comp != prev_compound:
            pit_at_lap[lap] = comp
        prev_compound = comp

    def fn(obs, env):
        lap_number = env.current_lap + 1
        if lap_number in pit_at_lap:
            return COMPOUND_TO_ACTION[pit_at_lap[lap_number]]
        return 0

    return fn


def dqn_fn_factory(model):
    def fn(obs, env):
        action, _ = model.predict(obs, deterministic=True)
        return int(action)
    return fn


def qrl_fn_factory(policy, normalizer):
    def fn(obs, env):
        probs = policy.get_action_probs(obs, normalizer)
        return int(probs.argmax().item())
    return fn


def run_and_log_episode(scenario, action_fn):
    """Runs one deterministic episode, returns (per-lap DataFrame, pits, finish position, total reward)."""
    env = F1StrategyEnv(scenario=scenario)
    obs, _ = env.reset()
    done = False
    cum_reward = 0.0
    rows = []
    while not done:
        lap_number = env.current_lap + 1
        under_sc = lap_number in env.safety_car_laps
        action = action_fn(obs, env)
        obs, reward, terminated, truncated, info = env.step(action)
        cum_reward += reward
        rows.append({
            "lap": lap_number,
            "position": info["position"],
            "tyre_compound": COMPOUND_NAMES.get(info["tyre_compound"], "?"),
            "tyre_age": info["tyre_age"],
            "gap_ahead": round(info["gap_ahead"], 2),
            "gap_behind": round(info["gap_behind"], 2),
            "pit_stop": action != 0,
            "safety_car": under_sc,
            "reward": round(reward, 3),
            "cumulative_reward": round(cum_reward, 3),
            "total_time": round(info["total_time"], 1),
        })
        done = terminated or truncated
    return pd.DataFrame(rows), info["pits_done"], info["position"], cum_reward


@st.cache_data(show_spinner="Running lap-by-lap replay...")
def get_replay_logs(fname, has_dqn, has_qrl):
    scenario = load_scenario_file(os.path.join(DATA_DIR, fname))
    logs = {}
    logs["Real Strategy"] = run_and_log_episode(scenario, real_strategy_fn_factory(scenario))
    if has_dqn:
        model = load_dqn_model()
        if model is not None:
            logs["DQN"] = run_and_log_episode(scenario, dqn_fn_factory(model))
    if has_qrl:
        bundle = load_qrl_model()
        if bundle is not None:
            policy, normalizer = bundle
            logs["QRL"] = run_and_log_episode(scenario, qrl_fn_factory(policy, normalizer))
    return scenario, logs


# ----------------------------- Quantum circuit diagram helper -----------------------------

def entanglement_pairs_for_layer(num_qubits, layer_idx):
    """Mirrors the alternating entanglement schedule used in main_qrl.py:
    layer_idx % 3 == 0 -> forward chain, 1 -> reverse chain, 2 -> full pairwise."""
    if num_qubits < 2:
        return [], "none (single qubit)"
    pattern = layer_idx % 3
    if pattern == 0:
        pairs = [(i, i + 1) for i in range(num_qubits - 1)]
        name = "forward chain"
    elif pattern == 1:
        pairs = [(i, i - 1) for i in range(num_qubits - 1, 0, -1)]
        name = "reverse chain"
    else:
        pairs = [(i, j) for i in range(num_qubits) for j in range(i + 1, num_qubits)]
        name = "full pairwise"
    return pairs, name


def build_circuit_figure(num_qubits, layer_idx):
    """Schematic (not simulation) diagram of one variational layer: angle
    encoding -> RY-RZ-RY per qubit -> entangling layer for this layer index."""
    pairs, pattern_name = entanglement_pairs_for_layer(num_qubits, layer_idx)

    x_encode, x_ry1, x_rz, x_ry2, x_ent = 0.0, 1.1, 2.2, 3.3, 4.6
    fig = go.Figure()

    for q in range(num_qubits):
        fig.add_shape(type="line", x0=-0.5, x1=x_ent + 0.9, y0=q, y1=q,
                      line=dict(color="#999999", width=1.5))
        fig.add_annotation(x=-0.7, y=q, text=f"q{q}", showarrow=False, xanchor="right",
                            font=dict(size=12))

    def gate_column(x, label, color):
        for q in range(num_qubits):
            fig.add_shape(type="rect", x0=x - 0.3, x1=x + 0.3, y0=q - 0.3, y1=q + 0.3,
                          line=dict(color=color, width=1), fillcolor=color, opacity=0.9)
            fig.add_annotation(x=x, y=q, text=label, showarrow=False,
                                font=dict(color="white", size=11))

    gate_column(x_encode, "Enc", "#636EFA")
    gate_column(x_ry1, "RY", "#EF553B")
    gate_column(x_rz, "RZ", "#00CC96")
    gate_column(x_ry2, "RY", "#EF553B")

    for a, b in pairs:
        lo, hi = min(a, b), max(a, b)
        fig.add_shape(type="line", x0=x_ent, x1=x_ent, y0=lo, y1=hi,
                      line=dict(color="#AB63FA", width=3))
        fig.add_shape(type="circle", x0=x_ent - 0.09, x1=x_ent + 0.09, y0=a - 0.09, y1=a + 0.09,
                      fillcolor="#AB63FA", line=dict(color="#AB63FA"))
        fig.add_shape(type="circle", x0=x_ent - 0.12, x1=x_ent + 0.12, y0=b - 0.12, y1=b + 0.12,
                      fillcolor="white", line=dict(color="#AB63FA", width=3))

    fig.update_xaxes(visible=False, range=[-1.3, x_ent + 1.3])
    fig.update_yaxes(visible=False, range=[num_qubits - 0.3, -1.0])
    fig.update_layout(
        height=110 + 65 * max(num_qubits, 1),
        margin=dict(l=10, r=10, t=40, b=10),
        title=f"Layer {layer_idx}  ·  entanglement: {pattern_name}",
        plot_bgcolor="white",
        showlegend=False,
    )
    return fig, pattern_name


# ----------------------------- Page setup -----------------------------

st.set_page_config(page_title="F1 Race Strategy RL Dashboard", page_icon="🏎️", layout="wide")

st.title("🏎️ F1 Race Strategy RL — Dashboard")
st.caption(
    "IITISoC 2026 — Quantum Computing × Machine Learning: benchmarking a Quantum RL (VQC/PPO) "
    "agent against a classical DQN baseline and real historical team strategy."
)

with st.sidebar:
    st.header("Project setup")
    st.markdown(f"**Driver:** {DRIVER}")
    st.markdown(f"**Tracks:** {', '.join(TRACKS)}")
    st.markdown(f"**Train years:** {', '.join(map(str, TRAIN_YEARS))}")
    st.markdown(f"**Held-out test year:** {', '.join(map(str, TEST_YEARS))}")
    st.divider()
    st.markdown("**Component availability**")
    st.markdown(f"- F1StrategyEnv: {'✅' if HAS_ENV else '❌ (f1_strategy_env.py not importable)'}")
    st.markdown(f"- data_loader (state features): {'✅' if HAS_DATA_LOADER else '❌ not importable'}")
    st.markdown(f"- DQN (Stable-Baselines3): {'✅' if HAS_SB3 else '❌ not installed'}")
    st.markdown(f"- QRL (PennyLane/Qiskit + torch): {'✅' if HAS_QRL_DEPS else '❌ not installed'}")
    st.markdown(f"- Trained DQN model: {'✅' if load_dqn_model() is not None else '❌ not found'}")
    st.markdown(f"- Trained QRL model: {'✅' if load_qrl_model() is not None else '❌ not found'}")
    st.divider()
    if st.button("Clear cache / reload data"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

(
    tab_overview,
    tab_ps,
    tab_rewards,
    tab_actions,
    tab_convergence,
    tab_qcircuit,
    tab_env_reward,
    tab_replay,
    tab_divergence,
    tab_raw,
) = st.tabs([
    "Overview",
    "Problem Statement Coverage",
    "Reward Comparison",
    "Pit Stop Analysis",
    "Training Convergence",
    "Quantum Circuit",
    "Environment & Reward Design",
    "Lap-by-Lap Replay",
    "Strategy Divergence",
    "Raw Data",
])


# ----------------------------- Overview -----------------------------

with tab_overview:
    st.subheader("What this project benchmarks")
    st.markdown(
        """
Real historical Formula 1 telemetry (via FastF1) is used to build a custom race-strategy
environment. Three agents are compared against each other in that environment:

- **Real historical strategy** — the actual pit-stop and tyre calls the team made
- **Classical baseline (DQN)** — a Stable-Baselines3 DQN agent, trained on the same curriculum
- **Quantum RL agent (QRL)** — a Variational-Quantum-Circuit policy trained with PPO

All three are trained (or replayed) on **2022–2024** data across Silverstone, Monza, and Monaco,
then evaluated on the **held-out 2025** season to check generalization.
        """
    )
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Tracks", len(TRACKS))
    col2.metric("Train seasons", len(TRAIN_YEARS))
    col3.metric("Held-out test season", TEST_YEARS[0])
    col4.metric("Actions", "STAY / PIT→Soft / PIT→Medium / PIT→Hard")

    st.subheader("Deliverable → dashboard section")
    st.table(pd.DataFrame([
        {"PS deliverable": "Reward curves, convergence plots, sample efficiency", "Section": "Training Convergence"},
        {"PS deliverable": "Strategy performance vs classical baseline & real team strategy", "Section": "Reward Comparison"},
        {"PS deliverable": "Lap-by-lap replay: QRL / DQN / real strategy calls, position & tyre changes", "Section": "Lap-by-Lap Replay"},
        {"PS deliverable": "Strategy divergence analysis", "Section": "Strategy Divergence"},
        {"PS deliverable": "Quantum circuit design: qubits, gates, entanglement, encoding", "Section": "Quantum Circuit"},
        {"PS deliverable": "Reward function & environment documentation", "Section": "Environment & Reward Design"},
        {"PS deliverable": "Deliverables / constraints / evaluation-criteria checklist", "Section": "Problem Statement Coverage"},
        {"PS deliverable": "Final race summaries / comparative benchmarking", "Section": "Reward Comparison + Raw Data"},
    ]))


# ----------------------------- Problem statement coverage -----------------------------

with tab_ps:
    st.header("Problem Statement Coverage")
    st.caption(
        "Maps this project against the IITISoC 2026 'Quantum RL Agent for F1 Race Strategy "
        "Optimization' problem statement: deliverables, technical constraints, and evaluation criteria."
    )

    st.subheader("Mandatory deliverables")
    dqn_ok = load_dqn_model() is not None
    qrl_ok = load_qrl_model() is not None
    scenarios_ok = len(discover_scenarios()) > 0
    summary_ok = os.path.exists(os.path.join(RESULTS_DIR, "agent_comparison_summary.txt"))

    auto_deliverables = [
        ("Custom Gym environment built from real FastF1 telemetry", scenarios_ok),
        ("Trained classical DQN baseline", dqn_ok),
        ("Trained Quantum RL (VQC) agent", qrl_ok),
        ("Live lap-by-lap replay: QRL vs. DQN vs. real strategy", scenarios_ok),
        ("Strategy divergence analysis", scenarios_ok),
        ("Benchmark report (reward curves, convergence, comparisons)", summary_ok),
    ]
    for label, done in auto_deliverables:
        # No explicit `key` here on purpose: these are disabled/read-only and
        # must keep reflecting the freshly computed `done` value on every
        # rerun rather than sticking to whatever it was the first time.
        st.checkbox(label, value=done, disabled=True)
    st.caption("Auto-checked from files found in `data/`, `models/`, and `results/`.")

    st.markdown("**Not verifiable from inside the app** — tick these off yourself as you finish them "
                "(these checkboxes only persist for your current browser session):")
    for item in PS_MANUAL_DELIVERABLES:
        st.checkbox(item, key=f"manual_deliv_{item}")

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Technical constraints")
        for c in PS_TECHNICAL_CONSTRAINTS:
            st.markdown(f"- {c}")
    with col_b:
        st.subheader("Evaluation criteria")
        for c in PS_EVALUATION_CRITERIA:
            st.markdown(f"- {c}")

    with st.expander("Optional / bonus extensions"):
        for b in PS_BONUS_EXTENSIONS:
            st.markdown(f"- {b}")
        st.caption("The interactive dashboard bonus item is this app itself.")


# ----------------------------- Reward comparison -----------------------------

with tab_rewards:
    st.header("Reward Comparison")
    split = st.radio("Data split", ["Train (2022–2024)", "Test (2025, held-out)"], horizontal=True)

    if split.startswith("Train"):
        df = load_csv(os.path.join(RESULTS_DIR, "dqn_results_train.csv"))
        if df is None:
            st.warning("`results/dqn_results_train.csv` not found — run `main_dqn.py` first.")
        else:
            df = df.copy()
            df["track_year"] = df["track"].astype(str) + " " + df["year"].astype(str)
            value_cols = [c for c in ["random_reward", "real_strategy_reward", "dqn_reward"] if c in df.columns]
            melted = df.melt(id_vars=["track_year"], value_vars=value_cols, var_name="agent", value_name="reward")
            fig = px.bar(melted, x="track_year", y="reward", color="agent", barmode="group",
                         template="plotly_white", title="Random vs Real Strategy vs DQN — Train years")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(df, use_container_width=True)
            st.caption("Note: the QRL agent is only evaluated against the held-out test year in this "
                       "pipeline (see main_qrl.py) — switch to the Test tab for the 3-way comparison.")
    else:
        comp_df = load_csv(os.path.join(RESULTS_DIR, "agent_comparison_rewards.csv"))
        if comp_df is None:
            st.warning(
                "`results/agent_comparison_rewards.csv` not found — run `compare_agents.py` after "
                "both `main_dqn.py` and `main_qrl.py` have produced their test-year (2025) results."
            )
            fallback = load_csv(os.path.join(RESULTS_DIR, "dqn_results_test.csv"))
            if fallback is not None:
                st.caption("Showing DQN-only test results in the meantime:")
                fallback = fallback.copy()
                fallback["track_year"] = fallback["track"].astype(str) + " " + fallback["year"].astype(str)
                value_cols = [c for c in ["random_reward", "real_strategy_reward", "dqn_reward"] if c in fallback.columns]
                melted = fallback.melt(id_vars=["track_year"], value_vars=value_cols, var_name="agent", value_name="reward")
                fig = px.bar(melted, x="track_year", y="reward", color="agent", barmode="group", template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(fallback, use_container_width=True)
        else:
            comp_df = comp_df.copy()
            comp_df["track_year"] = comp_df["track"].astype(str) + " " + comp_df["year"].astype(str)
            value_cols = [c for c in [
                "random_reward_dqn_eval", "real_strategy_reward_dqn_eval", "dqn_reward", "qrl_reward"
            ] if c in comp_df.columns]
            melted = comp_df.melt(id_vars=["track_year"], value_vars=value_cols, var_name="agent", value_name="reward")
            fig = px.bar(melted, x="track_year", y="reward", color="agent", barmode="group",
                         template="plotly_white", title="Random vs Real Strategy vs DQN vs QRL — Test year (2025)")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(comp_df, use_container_width=True)

            summary_path = os.path.join(RESULTS_DIR, "agent_comparison_summary.txt")
            if os.path.exists(summary_path):
                with open(summary_path) as f:
                    st.text(f.read())

            png_path = os.path.join(RESULTS_DIR, "agent_comparison_rewards.png")
            if os.path.exists(png_path):
                st.image(png_path, caption="results/agent_comparison_rewards.png")


# ----------------------------- Pit stop / action analysis -----------------------------

with tab_actions:
    st.header("Pit Stop / Action Analysis")
    split = st.radio("Data split", ["Train (2022–2024)", "Test (2025, held-out)"], horizontal=True, key="actions_split")

    if split.startswith("Train"):
        df = load_csv(os.path.join(RESULTS_DIR, "dqn_action_analysis_train.csv"))
        if df is None:
            st.warning("`results/dqn_action_analysis_train.csv` not found — run `main_dqn.py` first.")
        else:
            df = df.copy()
            df["track_year"] = df["track"].astype(str) + " " + df["year"].astype(str)
            action_cols = [c for c in ACTION_NAMES.values() if c in df.columns]
            melted = df.melt(id_vars=["track_year"], value_vars=action_cols, var_name="action", value_name="count")
            fig = px.bar(melted, x="track_year", y="count", color="action", barmode="stack",
                         template="plotly_white", title="DQN action breakdown — Train years")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(df, use_container_width=True)
    else:
        dqn_a = load_csv(os.path.join(RESULTS_DIR, "dqn_action_analysis_test.csv"))
        qrl_a = load_csv(os.path.join(QRL_LOG_DIR, "test_2025", "qrl_action_breakdown.csv"))

        cols = st.columns(2)
        with cols[0]:
            st.subheader("DQN")
            if dqn_a is None:
                st.warning("`results/dqn_action_analysis_test.csv` not found.")
            else:
                dqn_a = dqn_a.copy()
                dqn_a["track_year"] = dqn_a["track"].astype(str) + " " + dqn_a["year"].astype(str)
                action_cols = [c for c in ACTION_NAMES.values() if c in dqn_a.columns]
                melted = dqn_a.melt(id_vars=["track_year"], value_vars=action_cols, var_name="action", value_name="count")
                fig = px.bar(melted, x="track_year", y="count", color="action", barmode="stack", template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(dqn_a, use_container_width=True)

        with cols[1]:
            st.subheader("QRL")
            if qrl_a is None:
                st.warning(f"`{QRL_LOG_DIR}/test_2025/qrl_action_breakdown.csv` not found — run `main_qrl.py`.")
            else:
                qrl_a = qrl_a.copy()
                qrl_a["track_year"] = qrl_a["track"].astype(str) + " " + qrl_a["year"].astype(str)
                action_cols = [c for c in ACTION_NAMES.values() if c in qrl_a.columns]
                melted = qrl_a.melt(id_vars=["track_year"], value_vars=action_cols, var_name="action", value_name="count")
                fig = px.bar(melted, x="track_year", y="count", color="action", barmode="stack", template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(qrl_a, use_container_width=True)


# ----------------------------- Training convergence -----------------------------

with tab_convergence:
    st.header("Training Convergence")
    st.caption("Convergence speed and sample efficiency, DQN vs QRL — as requested in the problem statement.")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("QRL (PPO) reward history")
        qrl_hist_path = os.path.join(QRL_LOG_DIR, "qrl_reward_history.npy")
        if os.path.exists(qrl_hist_path):
            hist = np.load(qrl_hist_path)
            hist_df = pd.DataFrame({"update": np.arange(1, len(hist) + 1), "avg_reward": hist})
            hist_df["smoothed (10-update MA)"] = hist_df["avg_reward"].rolling(10, min_periods=1).mean()
            fig = px.line(hist_df, x="update", y=["avg_reward", "smoothed (10-update MA)"], template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"`{qrl_hist_path}` not found yet — run `main_qrl.py` to generate it.")

    with col2:
        st.subheader("DQN (Stable-Baselines3) training curve")
        try:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
            tb_available = True
        except ImportError:
            tb_available = False

        if not tb_available:
            st.info(
                "Install `tensorboard` (`pip install tensorboard`) to plot the DQN training curve "
                f"here, or run `tensorboard --logdir {TB_LOG_DIR}` separately."
            )
        elif not os.path.isdir(TB_LOG_DIR):
            st.info(f"No `{TB_LOG_DIR}/` directory found yet — run `main_dqn.py` to generate it.")
        else:
            run_dirs = sorted(glob.glob(os.path.join(TB_LOG_DIR, "DQN_F1*")))
            if not run_dirs:
                st.info(f"No `DQN_F1*` runs found under `{TB_LOG_DIR}/`.")
            else:
                latest_run = run_dirs[-1]
                ea = EventAccumulator(latest_run)
                ea.Reload()
                tag_candidates = [t for t in ea.Tags().get("scalars", []) if "ep_rew_mean" in t]
                if tag_candidates:
                    tag = tag_candidates[0]
                    events = ea.Scalars(tag)
                    tb_df = pd.DataFrame({"step": [e.step for e in events], "reward": [e.value for e in events]})
                    fig = px.line(tb_df, x="step", y="reward", template="plotly_white",
                                  title=f"{tag} — {os.path.basename(latest_run)}")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info(f"No episode-reward scalar found in `{latest_run}`. "
                            f"Available tags: {ea.Tags().get('scalars', [])}")


# ----------------------------- Quantum circuit design -----------------------------

with tab_qcircuit:
    st.header("Quantum Circuit Design")
    st.caption(
        "Documents the VQC policy's qubit count, gate selection, entanglement topology, and encoding "
        "strategy — the design-justification deliverable from the problem statement's 'Quantum RL Agent' "
        "scope of work."
    )

    if HAS_QRL_DEPS:
        num_qubits, num_layers = NUM_QUBITS, NUM_LAYERS
        st.success(f"Loaded live config from `main_qrl.py`: {num_qubits} qubits, {num_layers} variational layers.")
    else:
        num_qubits, num_layers = 4, 3
        st.warning(
            "`main_qrl.py` / PennyLane / torch aren't importable here — showing the documented design "
            "with illustrative qubit/layer counts, not values read from your actual config."
        )

    metric_cols = st.columns(4)
    metric_cols[0].metric("Qubits", num_qubits)
    metric_cols[1].metric("Variational layers", num_layers)
    if HAS_QRL_DEPS:
        metric_cols[2].metric("Input features", NUM_FEATURES)
        metric_cols[3].metric("Actions", NUM_ACTIONS)

    st.subheader("Design justification")
    st.markdown(f"""
- **Encoding strategy:** {QUANTUM_CIRCUIT_NOTES['encoding']}
- **Ansatz / gate selection:** {QUANTUM_CIRCUIT_NOTES['ansatz']}
- **Entanglement topology:** {QUANTUM_CIRCUIT_NOTES['entanglement']}
    """)

    st.subheader("Layer-by-layer entanglement schedule")
    layer_pick = st.slider("Inspect layer", 0, max(num_layers - 1, 0), 0, key="qcircuit_layer_slider")
    fig, pattern_name = build_circuit_figure(num_qubits, layer_pick)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"Layer {layer_pick} uses **{pattern_name}** entanglement. The schedule cycles every 3 layers: "
        "forward chain → reverse chain → full pairwise."
    )

    with st.expander("Why alternate entanglement patterns across layers?"):
        st.markdown(
            "A single fixed entangling pattern repeated in every layer limits which correlations between "
            "race-state features (e.g. tyre age vs. gap-to-leader) the circuit can represent. Cycling "
            "between a forward chain, a reverse chain, and a full pairwise layer spreads entangling "
            "capacity across all qubit pairs over the course of the circuit while keeping any single "
            "layer shallow and trainable."
        )


# ----------------------------- Environment & reward design -----------------------------

with tab_env_reward:
    st.header("Environment & Reward Design")
    st.caption(
        "Documents the RL environment's observation space and reward shaping — the 'reward function' "
        "and environment documentation the problem statement asks teams to provide."
    )

    st.subheader("Observation space (13 features)")
    if STATE_FEATURES:
        feat_df = pd.DataFrame({
            "feature": STATE_FEATURES,
            "description": [STATE_FEATURE_DESCRIPTIONS.get(f, "—") for f in STATE_FEATURES],
        })
        st.dataframe(feat_df, use_container_width=True)
    else:
        st.info("`data_loader.py` not importable here — feature list unavailable.")

    st.subheader("Reward shaping")
    if HAS_ENV:
        E = F1StrategyEnv
        reward_df = pd.DataFrame([
            {"term": "Pace reward", "value": E.PACE_SCALE,
             "meaning": "Reward per second faster than median pace (zeroed out under safety car)"},
            {"term": "Position reward", "value": E.POSITION_SCALE,
             "meaning": "Per-lap reward for track position, scaled toward P1"},
            {"term": "Finish bonus", "value": E.FINISH_SCALE,
             "meaning": "One-off bonus on the final lap, scaled by finishing position"},
            {"term": "Extra-stop penalty", "value": E.EXTRA_STOP_PENALTY,
             "meaning": "Penalty applied per pit stop beyond the first"},
            {"term": "Minimum stint length", "value": E.MIN_STINT_LAPS,
             "meaning": "Stints shorter than this are penalized (3 laps if under safety car)"},
            {"term": "Short-stint penalty", "value": E.SHORT_STINT_PENALTY,
             "meaning": "Penalty per lap short of the minimum stint length"},
            {"term": "Safety-car pit factor", "value": E.SC_PIT_FACTOR,
             "meaning": "Pit-stop time loss is multiplied by this under safety car"},
            {"term": "Safety-car pace factor", "value": E.SC_PACE_FACTOR,
             "meaning": "Lap time is multiplied by this under safety car"},
            {"term": "Degradation cliff rate", "value": E.CLIFF_RATE,
             "meaning": "Extra pace loss per lap once tyre age passes the per-track cliff age"},
        ])
        st.dataframe(reward_df, use_container_width=True)
        st.caption(
            "The cliff age itself (`CLIFF_AGE`) is fit per track/driver from real historical stint "
            "lengths in `data_loader.py`, not a single fixed constant across all scenarios."
        )
    else:
        st.info("`f1_strategy_env.py` not importable here — reward constants unavailable.")


# ----------------------------- Lap-by-lap replay -----------------------------

with tab_replay:
    st.header("Lap-by-Lap Race Replay")

    if not HAS_ENV:
        st.error("`f1_strategy_env.py` could not be imported — place this dashboard file in the same "
                 "directory as your project scripts.")
    else:
        entries = discover_scenarios()
        if not entries:
            st.warning(f"No scenario files found in `{DATA_DIR}/` — run `generate_data.py` first.")
        else:
            labels = [f"{track.title()} {year}" for track, year, driver, fname in entries]
            choice = st.selectbox("Select race scenario", labels)
            idx = labels.index(choice)
            track, year, driver, fname = entries[idx]

            has_dqn = load_dqn_model() is not None
            has_qrl = load_qrl_model() is not None
            scenario, logs = get_replay_logs(fname, has_dqn, has_qrl)

            if not has_dqn:
                st.info("DQN model not found at `models/dqn_f1_strategist_final.zip` — showing Real Strategy only "
                         "for DQN's slot. Train it with `main_dqn.py`.")
            if not has_qrl:
                st.info("QRL model not found (`models/qrl_f1_strategist_final.pt`) or torch/qiskit unavailable — "
                         "train it with `main_qrl.py` to include it here.")

            combined = []
            for name, (df, pits, finish_pos, total_r) in logs.items():
                d = df.copy()
                d["agent"] = name
                combined.append(d)
            combined_df = pd.concat(combined, ignore_index=True)

            st.subheader(f"Final summary — {choice}")
            summary_cols = st.columns(len(logs))
            for col, (name, (df, pits, finish_pos, total_r)) in zip(summary_cols, logs.items()):
                col.metric(name, f"P{finish_pos}", f"{pits} pit stops · reward {total_r:.1f}")

            max_lap = int(combined_df["lap"].max())

            # --- Scrub / auto-play controls ---
            if "lap_cursor" not in st.session_state or st.session_state.lap_cursor > max_lap:
                st.session_state.lap_cursor = max_lap

            play_cols = st.columns([1, 1, 3])
            autoplay = play_cols[0].checkbox("▶ Auto-play", key="autoplay_toggle")
            play_speed = play_cols[1].select_slider(
                "Speed", options=[0.15, 0.3, 0.6, 1.0], value=0.3,
                format_func=lambda s: f"{1/s:.1f} lap/s", key="autoplay_speed",
            )

            
            if autoplay:
                time.sleep(play_speed)
                st.session_state.lap_cursor = (st.session_state.lap_cursor % max_lap) + 1
                st.rerun()

            lap_cursor = st.slider("Scrub to lap", 1, max_lap, key="lap_cursor")

            st.subheader("Position over the race")
            fig_pos = px.line(combined_df, x="lap", y="position", color="agent", markers=True, template="plotly_white")
            fig_pos.update_yaxes(autorange="reversed", title="Position (P1 at top)")
            for sc_lap in sorted(scenario.safety_car_laps):
                fig_pos.add_vline(x=sc_lap, line_dash="dot", line_color="orange", opacity=0.5)
            fig_pos.add_vline(x=lap_cursor, line_color="red")
            st.plotly_chart(fig_pos, use_container_width=True)
            st.caption("Dotted orange lines = safety car laps. Red line = current scrub position.")

            st.subheader("Tyre age & pit stops")
            fig_tyre = px.line(combined_df, x="lap", y="tyre_age", color="agent", template="plotly_white")
            pit_points = combined_df[combined_df["pit_stop"]]
            if not pit_points.empty:
                fig_tyre.add_scatter(
                    x=pit_points["lap"], y=pit_points["tyre_age"], mode="markers",
                    marker=dict(size=11, symbol="triangle-down", color="black"),
                    name="Pit stop",
                )
            st.plotly_chart(fig_tyre, use_container_width=True)

            st.subheader(f"State at lap {lap_cursor}")
            state_cols = st.columns(len(logs))
            for col, (name, (df, *_rest)) in zip(state_cols, logs.items()):
                row = df[df["lap"] == lap_cursor]
                col.markdown(f"**{name}**")
                if not row.empty:
                    r = row.iloc[0]
                    col.write(f"Position: P{int(r['position'])}")
                    col.write(f"Tyre: {r['tyre_compound']} (age {int(r['tyre_age'])})")
                    col.write(f"Gap ahead/behind: {r['gap_ahead']}s / {r['gap_behind']}s")
                    if r["safety_car"]:
                        col.warning("Safety Car")
                    if r["pit_stop"]:
                        col.success("PIT STOP this lap")

            st.subheader("Pit stop log (strategy divergence at a glance)")
            pit_table = combined_df[combined_df["pit_stop"]][["agent", "lap", "tyre_compound"]].reset_index(drop=True)
            st.dataframe(pit_table, use_container_width=True)

            st.download_button(
                "Download full lap-by-lap log (CSV)",
                combined_df.to_csv(index=False),
                file_name=f"replay_{track}_{year}.csv",
            )


# ----------------------------- Strategy divergence sweep -----------------------------

with tab_divergence:
    st.header("Strategy Divergence — Full Sweep")
    st.caption(
        "Compares pit-lap timing and finishing position across Real / DQN / QRL for every scenario found in data/."
    )

    if not HAS_ENV:
        st.error("`f1_strategy_env.py` could not be imported.")
    else:
        entries = discover_scenarios()
        filter_cols = st.columns(2)
        with filter_cols[0]:
            year_filter = st.multiselect(
                "Years to include",
                sorted({e[1] for e in entries}),
                default=[str(y) for y in TEST_YEARS if str(y) in {e[1] for e in entries}] or sorted({e[1] for e in entries}),
            )
        with filter_cols[1]:
            track_filter = st.multiselect(
                "Tracks to include",
                sorted({e[0].title() for e in entries}),
                default=sorted({e[0].title() for e in entries}),
            )
        if st.button("Run divergence sweep"):
            filtered = [e for e in entries if e[1] in year_filter and e[0].title() in track_filter]
            if not filtered:
                st.warning("No scenarios match the selected year(s)/track(s).")
            else:
                has_dqn = load_dqn_model() is not None
                has_qrl = load_qrl_model() is not None
                rows = []
                progress = st.progress(0.0)
                for i, (track, year, drv, fname) in enumerate(filtered):
                    _, logs = get_replay_logs(fname, has_dqn, has_qrl)
                    real_df = logs["Real Strategy"][0]
                    real_pits = set(real_df.loc[real_df["pit_stop"], "lap"])
                    row = {
                        "track": track.title(),
                        "year": year,
                        "real_pit_laps": sorted(real_pits),
                        "real_finish_pos": logs["Real Strategy"][2],
                    }
                    for name in ("DQN", "QRL"):
                        if name in logs:
                            agent_df = logs[name][0]
                            agent_pits = set(agent_df.loc[agent_df["pit_stop"], "lap"])
                            row[f"{name}_pit_laps"] = sorted(agent_pits)
                            row[f"{name}_finish_pos"] = logs[name][2]
                            row[f"{name}_vs_real_pit_lap_diff"] = len(agent_pits.symmetric_difference(real_pits))
                            row[f"{name}_vs_real_finish_diff"] = abs(logs[name][2] - logs["Real Strategy"][2])
                    rows.append(row)
                    progress.progress((i + 1) / len(filtered))

                div_df = pd.DataFrame(rows)
                st.dataframe(div_df, use_container_width=True)
                st.download_button(
                    "Download divergence table (CSV)",
                    div_df.to_csv(index=False),
                    file_name="strategy_divergence.csv",
                )

                diff_cols = [c for c in div_df.columns if c.endswith("_pit_lap_diff")]
                if diff_cols:
                    div_df["track_year"] = div_df["track"] + " " + div_df["year"].astype(str)
                    melted = div_df.melt(id_vars=["track_year"], value_vars=diff_cols,
                                          var_name="agent", value_name="pit_lap_divergence")
                    fig = px.bar(melted, x="track_year", y="pit_lap_divergence", color="agent",
                                 barmode="group", template="plotly_white",
                                 title="Pit-lap divergence from real strategy (lower = more similar)")
                    st.plotly_chart(fig, use_container_width=True)


# ----------------------------- Raw data explorer -----------------------------

with tab_raw:
    st.header("Raw Data Explorer")
    st.caption("Browse any CSV your pipeline has produced so far.")

    candidate_paths = []
    for base_dir in (RESULTS_DIR, QRL_LOG_DIR):
        if os.path.isdir(base_dir):
            for root, _, files in os.walk(base_dir):
                for fn in files:
                    if fn.endswith(".csv"):
                        candidate_paths.append(os.path.join(root, fn))

    if not candidate_paths:
        st.info(f"No CSV files found yet under `{RESULTS_DIR}/` or `{QRL_LOG_DIR}/`.")
    else:
        chosen = st.selectbox("Choose a file", sorted(candidate_paths))
        df = load_csv(chosen)
        if df is not None:
            st.dataframe(df, use_container_width=True)
            st.download_button("Download this CSV", df.to_csv(index=False), file_name=os.path.basename(chosen))
