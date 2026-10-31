"""
Quantum Reinforcement Learning Agent for F1 Strategy (High Performance, Optimized)
"""

import os
import math
import pickle
import argparse
from dataclasses import dataclass, asdict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Operator

from f1_strategy_env import F1StrategyEnv

TRACKS = ["Silverstone", "Monza", "Monaco"]
DRIVER = "HAM"

# Years used to build the training set (curriculum runs across all track x year
# combinations, treated as independent "tracks" by the reward normalizer /
# exposure-balancing logic below).
TRAIN_YEARS = [2022, 2023, 2024]

# Held-out year used only for final evaluation / baseline comparison after
# training completes. Never appears in scenario_list / track_names passed to train().
TEST_YEAR = 2025

DATA_DIR = "data"
MODEL_DIR = "models"
LOG_DIR = "qrl_logs"

NUM_QUBITS = 4
NUM_LAYERS = 4
NUM_ACTIONS = 4
NUM_FEATURES = 13
ENCODER_HIDDEN = 64
NUM_ENTANGLING_PATTERNS = 3

GAMMA = 0.99
GAE_LAMBDA = 0.95
LEARNING_RATE = 0.0003
QUANTUM_LR_MULT = 2.0
ENTROPY_COEFF = 0.02
ENTROPY_COEFF_MIN = 0.001
VALUE_COEFF = 0.5
PPO_CLIP = 0.2
PPO_EPOCHS = 8
MINIBATCH_SIZE = 64
KL_TARGET = 0.02
WARMUP_STEPS = 20

BATCH_EPISODES = 64
NUM_EPISODES = 12000
SEED = 42

REWARD_NORM_WARMUP_UPDATES = 5
REWARD_NORM_CLIP = 10.0

# Curriculum / exploration
ENTROPY_BOOST_UPDATES = 10

SEARCH_N_TRIALS = 20
SEARCH_EPISODES = 300

@dataclass
class HParams:
    lr: float = LEARNING_RATE
    quantum_lr_mult: float = QUANTUM_LR_MULT
    entropy_coeff: float = ENTROPY_COEFF
    entropy_coeff_min: float = ENTROPY_COEFF_MIN
    value_coeff: float = VALUE_COEFF
    ppo_clip: float = PPO_CLIP
    ppo_epochs: int = PPO_EPOCHS
    minibatch_size: int = MINIBATCH_SIZE
    kl_target: float = KL_TARGET
    warmup_steps: int = WARMUP_STEPS
    batch_episodes: int = BATCH_EPISODES
    gae_lambda: float = GAE_LAMBDA
    gamma: float = GAMMA
    reward_norm_warmup_updates: int = REWARD_NORM_WARMUP_UPDATES
    reward_norm_clip: float = REWARD_NORM_CLIP

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

class RunningNormalizer:
    def __init__(self, shape, eps=1e-8):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = eps

    def update(self, x):
        x = np.asarray(x, dtype=np.float64)
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + (delta ** 2) * self.count * batch_count / tot_count
        new_var = m2 / tot_count
        self.mean, self.var, self.count = new_mean, new_var, tot_count

    def normalize(self, x):
        x = np.asarray(x, dtype=np.float64)
        return ((x - self.mean) / (np.sqrt(self.var) + 1e-8)).astype(np.float32)

class RewardNormalizer:

    def __init__(self, gamma, n_envs=1, eps=1e-8, clip=10.0, warmup_updates=0):
        self.gamma = gamma
        self.eps = eps
        self.clip = clip
        self.warmup_updates = warmup_updates
        self.ret_rms = RunningNormalizer(shape=())
        self.returns = np.zeros(n_envs, dtype=np.float64)
        self._update_count = 0

    def start_update(self):
        """Call once per PPO update / trajectory batch, before normalize()."""
        self._update_count += 1

    def normalize(self, rewards, dones):
        """rewards, dones: flat lists, one entry per transition, contiguous
        per-environment (same layout as collect_trajectories' output, or a
        per-track slice of it)."""
        rewards = np.asarray(rewards, dtype=np.float64)
        dones = np.asarray(dones, dtype=bool)

        n_envs_this_call = int(np.sum(dones))
        if n_envs_this_call > 0 and self.returns.shape[0] != n_envs_this_call:
            self.returns = np.zeros(n_envs_this_call, dtype=np.float64)

        normed = np.empty_like(rewards)
        env_idx = 0
        i = 0
        n = len(rewards)
        while i < n:
            self.returns[env_idx] = self.returns[env_idx] * self.gamma + rewards[i]
            self.ret_rms.update(np.array([[self.returns[env_idx]]]))
            normed[i] = rewards[i]
            if dones[i]:
                self.returns[env_idx] = 0.0
                env_idx += 1
            i += 1

        if self._update_count <= self.warmup_updates:
            return rewards.tolist()

        std = math.sqrt(self.ret_rms.var[0]) + self.eps
        normed = rewards / std
        normed = np.clip(normed, -self.clip, self.clip)
        return normed.tolist()

class VariationalQuantumLayer(nn.Module):
    def __init__(self, num_qubits, num_layers):
        super().__init__()
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.dim = 2 ** num_qubits
 
        self.thetas = nn.Parameter(torch.randn(num_layers, num_qubits) * 0.1)
        self.phis = nn.Parameter(torch.randn(num_layers, num_qubits) * 0.1)
        self.omegas = nn.Parameter(torch.randn(num_layers, num_qubits) * 0.1)
 
        self.register_buffer('cnot_patterns', self._build_cnot_patterns())
        self.register_buffer('pauli_z_signs', self._build_pauli_z_signs())
 
        # Precompute the fixed reshape geometry (stride/pairs) per qubit
        # once, instead of recomputing 2**q / dim//(2*stride) every forward
        # call inside the loop.
        self.strides = [2 ** q for q in range(num_qubits)]
        self.pairs = [self.dim // (2 * st) for st in self.strides]
 
    def _build_cnot_patterns(self):
        forward_qc = QuantumCircuit(self.num_qubits)
        for q in range(self.num_qubits - 1):
            forward_qc.cx(q, q + 1)
 
        reverse_qc = QuantumCircuit(self.num_qubits)
        for q in range(self.num_qubits - 1, 0, -1):
            reverse_qc.cx(q, q - 1)
 
        pairwise_qc = QuantumCircuit(self.num_qubits)
        for i in range(self.num_qubits):
            for j in range(i + 1, self.num_qubits):
                pairwise_qc.cx(i, j)
 
        forward_mat = torch.tensor(Operator(forward_qc).data, dtype=torch.cfloat)
        reverse_mat = torch.tensor(Operator(reverse_qc).data, dtype=torch.cfloat)
        pairwise_mat = torch.tensor(Operator(pairwise_qc).data, dtype=torch.cfloat)
 
        return torch.stack([forward_mat, reverse_mat, pairwise_mat], dim=0)
 
    def _build_pauli_z_signs(self):
        signs = torch.ones(self.dim, self.num_qubits)
        for i in range(self.dim):
            for q in range(self.num_qubits):
                if (i >> q) & 1:
                    signs[i, q] = -1.0
        return signs
 
    def forward(self, x):
        batch_size = x.shape[0]
        device = x.device
 
        state = torch.zeros(batch_size, self.dim, dtype=torch.cfloat, device=device)
        state[:, 0] = 1.0
 
        # --- Vectorized trig precompute for ALL (layer, qubit) pairs at once ---
        # Shapes: (batch, num_layers, num_qubits)
        theta = self.thetas.unsqueeze(0) + x.unsqueeze(1) * math.pi
        phi = self.phis.unsqueeze(0)
        omega = self.omegas.unsqueeze(0)
 
        c = torch.cos(theta * 0.5).to(torch.cfloat)
        s = torch.sin(theta * 0.5).to(torch.cfloat)
        ep = torch.exp(1j * phi.to(torch.cfloat) * 0.5).expand(batch_size, -1, -1)
        em = torch.exp(-1j * phi.to(torch.cfloat) * 0.5).expand(batch_size, -1, -1)
        co = torch.cos(omega * 0.5).to(torch.cfloat).expand(batch_size, -1, -1)
        so = torch.sin(omega * 0.5).to(torch.cfloat).expand(batch_size, -1, -1)
 
        for layer in range(self.num_layers):
            for q in range(self.num_qubits):
                c_ = c[:, layer, q].view(batch_size, 1, 1)
                s_ = s[:, layer, q].view(batch_size, 1, 1)
                ep_ = ep[:, layer, q].view(batch_size, 1, 1)
                em_ = em[:, layer, q].view(batch_size, 1, 1)
                co_ = co[:, layer, q].view(batch_size, 1, 1)
                so_ = so[:, layer, q].view(batch_size, 1, 1)
 
                stride, pairs = self.strides[q], self.pairs[q]
                state_view = state.view(batch_size, pairs, 2, stride)
                s0 = state_view[:, :, 0, :]
                s1 = state_view[:, :, 1, :]
 
                mid_s0 = em_ * c_ * s0 - em_ * s_ * s1
                mid_s1 = ep_ * s_ * s0 + ep_ * c_ * s1
 
                new_s0 = co_ * mid_s0 - so_ * mid_s1
                new_s1 = so_ * mid_s0 + co_ * mid_s1
 
                state = torch.stack([new_s0, new_s1], dim=2).reshape(batch_size, self.dim)
 
            pattern = self.cnot_patterns[layer % NUM_ENTANGLING_PATTERNS]
            state = torch.matmul(state, pattern.T)
 
        probs_full = (state * state.conj()).real
        expectations = probs_full @ self.pauli_z_signs
        return expectations

class QRLActorCritic(nn.Module):
    def __init__(self, num_qubits, num_layers, num_features, num_actions, hidden=64):
        super().__init__()
        self.num_qubits = num_qubits

        self.encoder_trunk = nn.Sequential(
            nn.Linear(num_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        self.angle_head = nn.Sequential(
            nn.Linear(hidden, num_qubits),
            nn.LayerNorm(num_qubits),
            nn.Tanh(),
        )

        self.quantum = VariationalQuantumLayer(num_qubits, num_layers)

        combined_dim = hidden + num_qubits

        self.actor = nn.Sequential(
            nn.Linear(combined_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_actions),
        )

        self.critic = nn.Sequential(
            nn.Linear(combined_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        features = self.encoder_trunk(x)
        angles = self.angle_head(features)
        q_out = self.quantum(angles)
        combined = torch.cat([features, q_out], dim=-1)

        logits = self.actor(combined)
        value = self.critic(combined)
        probs = torch.softmax(logits, dim=-1)
        return probs, value.squeeze(-1)

    def get_action_probs(self, obs_np, obs_normalizer=None):
        if obs_normalizer is not None:
            obs_np = obs_normalizer.normalize(obs_np[None, :])[0]
        x = torch.FloatTensor(obs_np).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            probs, _ = self.forward(x)
        return probs.squeeze(0)

    def evaluate_actions(self, states, actions):
        probs, values = self.forward(states)
        dist = Categorical(probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, values, entropy

QRLPolicy = QRLActorCritic

def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
        nn.init.constant_(module.bias, 0.0)

def apply_policy_init(policy):
    policy.apply(init_weights)
    nn.init.orthogonal_(policy.actor[-1].weight, gain=0.01)
    nn.init.constant_(policy.actor[-1].bias, 0.0)

def compute_gae(rewards, values, dones, gamma, lam):
    advantages = []
    gae = 0
    next_value = 0
    for t in reversed(range(len(rewards))):
        if dones[t]:
            next_value = 0
            gae = 0
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
        next_value = values[t]
    return advantages

def collect_trajectories(policy, envs, obs_normalizer):
    n_envs = len(envs)
    states_buf = [[] for _ in range(n_envs)]
    actions_buf = [[] for _ in range(n_envs)]
    rewards_buf = [[] for _ in range(n_envs)]
    values_buf = [[] for _ in range(n_envs)]
    log_probs_buf = [[] for _ in range(n_envs)]
    dones_buf = [[] for _ in range(n_envs)]
    last_info = None

    obs_list = []
    for env in envs:
        obs, _ = env.reset()
        obs_list.append(obs)

    active = list(range(n_envs))
    while active:
        raw_batch = np.array([obs_list[i] for i in active])
        obs_normalizer.update(raw_batch)
        norm_batch = obs_normalizer.normalize(raw_batch)
        state_t = torch.FloatTensor(norm_batch).to(DEVICE)

        with torch.no_grad():
            probs, values = policy.forward(state_t)
        dist = Categorical(probs)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)

        still_active = []
        for b, env_idx in enumerate(active):
            env = envs[env_idx]
            action = actions[b].item()

            states_buf[env_idx].append(obs_list[env_idx].copy())
            actions_buf[env_idx].append(action)
            values_buf[env_idx].append(values[b].item())
            log_probs_buf[env_idx].append(log_probs[b].item())

            obs, reward, terminated, truncated, info = env.step(action)
            rewards_buf[env_idx].append(reward)
            done = terminated or truncated
            dones_buf[env_idx].append(done)
            last_info = info

            if not done:
                obs_list[env_idx] = obs
                still_active.append(env_idx)

        active = still_active

    all_states, all_actions, all_rewards = [], [], []
    all_values, all_log_probs, all_dones = [], [], []
    for i in range(n_envs):
        all_states.extend(states_buf[i])
        all_actions.extend(actions_buf[i])
        all_rewards.extend(rewards_buf[i])
        all_values.extend(values_buf[i])
        all_log_probs.extend(log_probs_buf[i])
        all_dones.extend(dones_buf[i])

    return (all_states, all_actions, all_rewards, all_values,
            all_log_probs, all_dones, last_info)

def ppo_update(policy, optimizer, states, actions, old_log_probs, advantages, returns,
               old_values, ppo_clip, ppo_epochs, entropy_coeff, value_coeff,
               minibatch_size, obs_normalizer, kl_target):
    states_np = obs_normalizer.normalize(np.array(states))
    states_t = torch.FloatTensor(states_np).to(DEVICE)
    actions_t = torch.LongTensor(actions).to(DEVICE)
    old_log_probs_t = torch.FloatTensor(old_log_probs).to(DEVICE)
    advantages_t = torch.FloatTensor(advantages).to(DEVICE)
    returns_t = torch.FloatTensor(returns).to(DEVICE)
    old_values_t = torch.FloatTensor(old_values).to(DEVICE)

    if advantages_t.std() > 1e-6:
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

    n = states_t.shape[0]
    stats = {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0,
             'approx_kl': 0.0, 'clip_fraction': 0.0, 'grad_norm_quantum': 0.0}
    total_updates = 0

    for epoch in range(ppo_epochs):
        idx = torch.randperm(n, device=DEVICE)
        epoch_kl_sum = 0.0
        epoch_batches = 0

        for start in range(0, n, minibatch_size):
            mb_idx = idx[start:start + minibatch_size]

            log_probs, values, entropy = policy.evaluate_actions(states_t[mb_idx], actions_t[mb_idx])

            ratio = torch.exp(log_probs - old_log_probs_t[mb_idx])
            surr1 = ratio * advantages_t[mb_idx]
            surr2 = torch.clamp(ratio, 1.0 - ppo_clip, 1.0 + ppo_clip) * advantages_t[mb_idx]
            policy_loss = -torch.min(surr1, surr2).mean()

            values_clipped = old_values_t[mb_idx] + torch.clamp(
                values - old_values_t[mb_idx], -ppo_clip, ppo_clip)
            value_loss_unclipped = (values - returns_t[mb_idx]) ** 2
            value_loss_clipped = (values_clipped - returns_t[mb_idx]) ** 2
            value_loss = torch.max(value_loss_unclipped, value_loss_clipped).mean()

            entropy_loss = -entropy.mean()

            loss = policy_loss + value_coeff * value_loss + entropy_coeff * entropy_loss

            optimizer.zero_grad()
            loss.backward()

            grad_norm_q_sq = 0.0
            for p in policy.quantum.parameters():
                if p.grad is not None:
                    grad_norm_q_sq += p.grad.norm().item() ** 2
            grad_norm_q = grad_norm_q_sq ** 0.5

            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
            optimizer.step()

            with torch.no_grad():
                approx_kl = (old_log_probs_t[mb_idx] - log_probs).mean().item()
                clip_frac = ((ratio - 1.0).abs() > ppo_clip).float().mean().item()

            stats['policy_loss'] += policy_loss.item()
            stats['value_loss'] += value_loss.item()
            stats['entropy'] += entropy.mean().item()
            stats['approx_kl'] += approx_kl
            stats['clip_fraction'] += clip_frac
            stats['grad_norm_quantum'] += grad_norm_q
            total_updates += 1

            epoch_kl_sum += approx_kl
            epoch_batches += 1

        if epoch_batches > 0 and (epoch_kl_sum / epoch_batches) > kl_target:
            break

    if total_updates > 0:
        for k in stats:
            stats[k] /= total_updates
    return stats


def get_active_track_indices(step, num_updates, num_tracks):
    """Ramp in scenarios over the course of training, generalized to any
    num_tracks (originally hardcoded to 1 -> 2 -> 3 for a 3-track setup).
    Now activates roughly a third, two-thirds, then all of num_tracks,
    so a 9-scenario (3 track x 3 year) run ramps 3 -> 6 -> 9 instead of
    getting stuck on a single scenario for the first third of training."""
    frac = step / max(num_updates, 1)
    if frac < 0.33:
        n_active = max(1, round(num_tracks / 3))
    elif frac < 0.66:
        n_active = max(1, round(2 * num_tracks / 3))
    else:
        n_active = num_tracks
    return list(range(min(n_active, num_tracks)))

def _episode_boundaries(dones):
    """Split a flat, per-env-contiguous dones list into (start, end) index
    ranges, one per environment/episode, in the same order the envs were
    processed in collect_trajectories. Every env is guaranteed to contribute
    exactly one completed episode per training batch (fresh envs each batch,
    collection runs until all are done), so this recovers the per-env
    segment boundaries needed to route each env's transitions to the right
    per-track reward normalizer."""
    boundaries = []
    start = 0
    for i, d in enumerate(dones):
        if d:
            boundaries.append((start, i + 1))
            start = i + 1
    return boundaries

def save_checkpoint(path, policy, obs_normalizer):
    torch.save({
        'model_state_dict': policy.state_dict(),
        'norm_mean': obs_normalizer.mean,
        'norm_var': obs_normalizer.var,
    }, path)

def train(hparams: HParams, scenario_list, num_episodes, seed=SEED,
          save_checkpoints=True, verbose=True, trial=None, log_tag="",
          track_names=None):

    torch.manual_seed(seed)
    np.random.seed(seed)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    if track_names is None:
        track_names = [f"track_{i}" for i in range(len(scenario_list))]

    policy = QRLActorCritic(NUM_QUBITS, NUM_LAYERS, NUM_FEATURES, NUM_ACTIONS,
                             hidden=ENCODER_HIDDEN).to(DEVICE)
    apply_policy_init(policy)

    if verbose:
        num_params = sum(p.numel() for p in policy.parameters())
        q_params = (policy.quantum.thetas.numel() + policy.quantum.phis.numel()
                    + policy.quantum.omegas.numel())
        print(f"{log_tag}Quantum params: {q_params} | Classical params: "
              f"{num_params - q_params} | Total: {num_params}")
        print(f"{log_tag}PPO: {num_episodes} episodes, batch={hparams.batch_episodes}, "
              f"epochs={hparams.ppo_epochs}, minibatch={hparams.minibatch_size}")
        print(f"{log_tag}Training tracks ({len(track_names)}): {track_names}")

    quantum_params = list(policy.quantum.parameters())
    quantum_ids = {id(p) for p in quantum_params}
    classical_params = [p for p in policy.parameters() if id(p) not in quantum_ids]

    optimizer = optim.Adam([
        {'params': classical_params, 'lr': hparams.lr},
        {'params': quantum_params, 'lr': hparams.lr * hparams.quantum_lr_mult},
    ])

    num_updates = max(1, num_episodes // hparams.batch_episodes)

    def lr_lambda(step):
        if step < hparams.warmup_steps:
            return (step + 1) / hparams.warmup_steps
        progress = (step - hparams.warmup_steps) / max(1, (num_updates - hparams.warmup_steps))
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    obs_normalizer = RunningNormalizer(shape=(NUM_FEATURES,))

    reward_normalizers = {}

    num_tracks = len(scenario_list)
    track_exposure_counts = [0] * num_tracks

    entropy_boost_remaining = 0
    prev_active_track_count = 0

    reward_history = []
    best_reward = -float("inf")

    for step in range(num_updates):
        active_indices = get_active_track_indices(step, num_updates, num_tracks)

        batch_track_idx = []
        for _ in range(hparams.batch_episodes):
            chosen = min(active_indices, key=lambda idx: track_exposure_counts[idx])
            batch_track_idx.append(chosen)
            track_exposure_counts[chosen] += 1

        batch_envs = [F1StrategyEnv(scenario=scenario_list[idx]) for idx in batch_track_idx]

        (states, actions, rewards, values,
         log_probs, dones, info) = collect_trajectories(policy, batch_envs, obs_normalizer)

        # Route each env's transitions to its own track's reward normalizer.
        boundaries = _episode_boundaries(dones)
        track_segments = {}
        for env_number, (s, e) in enumerate(boundaries):
            t_idx = batch_track_idx[env_number]
            track_segments.setdefault(t_idx, []).append((s, e))

        rewards_for_gae = [0.0] * len(rewards)
        for t_idx, segments in track_segments.items():
            t_name = track_names[t_idx]
            if t_name not in reward_normalizers:
                reward_normalizers[t_name] = RewardNormalizer(
                    gamma=hparams.gamma,
                    clip=hparams.reward_norm_clip,
                    warmup_updates=hparams.reward_norm_warmup_updates,
                )
            normalizer = reward_normalizers[t_name]

            track_rewards, track_dones = [], []
            for (s, e) in segments:
                track_rewards.extend(rewards[s:e])
                track_dones.extend(dones[s:e])

            normalizer.start_update()
            normed = normalizer.normalize(track_rewards, track_dones)

            pos = 0
            for (s, e) in segments:
                length = e - s
                rewards_for_gae[s:e] = normed[pos:pos + length]
                pos += length

        advantages = compute_gae(rewards_for_gae, values, dones, hparams.gamma, hparams.gae_lambda)
        returns = [a + v for a, v in zip(advantages, values)]

        # Entropy schedule with a curriculum-change boost.
        if len(active_indices) > prev_active_track_count:
            entropy_boost_remaining = ENTROPY_BOOST_UPDATES
        prev_active_track_count = len(active_indices)

        base_entropy_coeff = max(hparams.entropy_coeff_min,
                                  hparams.entropy_coeff * (1 - step / num_updates))
        if entropy_boost_remaining > 0:
            entropy_coeff_now = max(base_entropy_coeff, hparams.entropy_coeff)
            entropy_boost_remaining -= 1
        else:
            entropy_coeff_now = base_entropy_coeff

        stats = ppo_update(policy, optimizer, states, actions, log_probs,
                            advantages, returns, values, hparams.ppo_clip, hparams.ppo_epochs,
                            entropy_coeff_now, hparams.value_coeff, hparams.minibatch_size,
                            obs_normalizer, hparams.kl_target)
        scheduler.step()

        ep_rewards = []
        ep_reward = 0
        for r, d in zip(rewards, dones):
            ep_reward += r
            if d:
                ep_rewards.append(ep_reward)
                ep_reward = 0
        avg_reward = float(np.mean(ep_rewards)) if ep_rewards else 0.0
        reward_history.append(avg_reward)

        if verbose:
            lr_classical = optimizer.param_groups[0]['lr']
            lr_quantum = optimizer.param_groups[1]['lr']
            n_tracks_active = len(active_indices)
            print(
                f"{log_tag}[Batch {step + 1}/{num_updates} completed] "
                f"R={avg_reward:.1f} | PolicyLoss={stats['policy_loss']:.4f} | "
                f"ValueLoss={stats['value_loss']:.4f} | Entropy={stats['entropy']:.3f} | "
                f"KL={stats['approx_kl']:.4f} | ClipFrac={stats['clip_fraction']:.2f} | "
                f"GradNormQ={stats['grad_norm_quantum']:.5f} | "
                f"Tracks={n_tracks_active} | EntCoeff={entropy_coeff_now:.4f} | "
                f"Pits={info['pits_done']} P{info['position']} | "
                f"lr_c={lr_classical:.6f} lr_q={lr_quantum:.6f}"
            )

        if avg_reward > best_reward:
            best_reward = avg_reward
            if save_checkpoints:
                save_checkpoint(os.path.join(MODEL_DIR, "qrl_best.pt"), policy, obs_normalizer)

        if trial is not None:
            trial.report(avg_reward, step)
            if trial.should_prune():
                import optuna
                raise optuna.TrialPruned()

    if verbose:
        exposure_str = ", ".join(
            f"{track_names[i]}={track_exposure_counts[i]}" for i in range(num_tracks)
        )
        print(f"{log_tag}Final per-track episode exposure: {exposure_str}")

    if save_checkpoints:
        save_checkpoint(os.path.join(MODEL_DIR, "qrl_f1_strategist_final.pt"), policy, obs_normalizer)
        np.save(os.path.join(LOG_DIR, "qrl_reward_history.npy"), np.array(reward_history))

    final_score = float(np.mean(reward_history[-5:])) if reward_history else -float("inf")

    return {
        'policy': policy,
        'obs_normalizer': obs_normalizer,
        'reward_history': reward_history,
        'best_reward': best_reward,
        'final_score': final_score,
        'track_exposure_counts': dict(zip(track_names, track_exposure_counts)),
    }


def objective(trial, scenario_list, track_names=None):
    import optuna

    hparams = HParams(
        lr=trial.suggest_float("lr", 1e-5, 1e-2, log=True),
        entropy_coeff=trial.suggest_float("entropy_coeff", 1e-4, 5e-2, log=True),
        ppo_clip=trial.suggest_float("ppo_clip", 0.1, 0.4),
        batch_episodes=trial.suggest_categorical("batch_episodes", [8, 16, 32, 64]),
        gae_lambda=trial.suggest_float("gae_lambda", 0.80, 0.99),
        ppo_epochs=trial.suggest_int("ppo_epochs", 2, 10),
        value_coeff=trial.suggest_float("value_coeff", 0.1, 1.0),
        quantum_lr_mult=trial.suggest_float("quantum_lr_mult", 0.5, 5.0, log=True),
        kl_target=trial.suggest_float("kl_target", 0.005, 0.05, log=True),
    )

    result = train(hparams, scenario_list, num_episodes=SEARCH_EPISODES,
                    seed=SEED, save_checkpoints=False, verbose=False,
                    trial=trial, log_tag=f"[trial {trial.number}] ",
                    track_names=track_names)

    print(f"[trial {trial.number}] score={result['final_score']:.3f} "
          f"best={result['best_reward']:.3f} params={trial.params}")

    return result['final_score']

def run_hyperparameter_search(scenario_list, track_names=None, n_trials=SEARCH_N_TRIALS,
                               search_episodes=SEARCH_EPISODES,
                               train_after_search=True):
    import optuna

    global SEARCH_EPISODES
    SEARCH_EPISODES = search_episodes

    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
    )
    study.optimize(lambda t: objective(t, scenario_list, track_names=track_names), n_trials=n_trials)

    print("\n" + "=" * 60)
    print(f"Search done. Best score: {study.best_value:.3f}")
    print(f"Best hyperparameters: {study.best_params}")
    print("=" * 60)

    if train_after_search:
        print(f"\nRe-running a full {NUM_EPISODES}-episode training with the "
              f"best hyperparameters found...")
        best_hparams = HParams(**study.best_params)
        result = train(best_hparams, scenario_list, num_episodes=NUM_EPISODES,
                        seed=SEED, save_checkpoints=True, verbose=True,
                        track_names=track_names)
        print(f"\nFull run with best hyperparameters -> best_reward="
              f"{result['best_reward']:.2f}")

    return study

def load_scenario(track, year):
    """Load a single pickled scenario for (track, year). Raises a clear error
    telling the user to run generate_data.py if the file is missing -- this
    now applies per (track, year) pair rather than per track only."""
    path = os.path.join(DATA_DIR, f"{track.lower()}_{year}_{DRIVER.lower()}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Scenario not found: {path}\n"
            f"Run generate_data.py first (it must include year={year})."
        )
    with open(path, "rb") as f:
        return pickle.load(f)

def load_scenarios_for_years(years, tracks=TRACKS):
    """Build an ordered {'<track>_<year>': scenario} dict for every
    (track, year) combination. Keys double as the `track_names` passed into
    train()/curriculum/reward-normalizer logic, so each (track, year) pair is
    treated as an independent training "track"."""
    scenarios = {}
    for year in years:
        for track in tracks:
            key = f"{track}_{year}"
            scenarios[key] = load_scenario(track, year)
    return scenarios

def final_evaluation(policy, obs_normalizer, scenarios):
    print("\nFinal evaluation:")
    for name, scenario in scenarios.items():
        env = F1StrategyEnv(scenario=scenario)
        obs, _ = env.reset()
        total, done, pit_laps = 0.0, False, []
        while not done:
            probs = policy.get_action_probs(obs, obs_normalizer)
            action = probs.argmax().item()
            if action != 0:
                pit_laps.append(env.current_lap + 1)
            obs, r, term, trunc, info = env.step(action)
            total += r
            done = term or trunc
        print(f"  {name:<20} R={total:.2f} pits={info['pits_done']} ")
        print(f"laps={pit_laps} P{info['position']}")

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # --- Training data: 2022, 2023, 2024 across all tracks ---
    train_scenarios = load_scenarios_for_years(TRAIN_YEARS, TRACKS)
    scenario_list = list(train_scenarios.values())
    track_names = list(train_scenarios.keys())

    print(f"Hybrid QRL (PPO, optimized): {NUM_QUBITS} qubits, {NUM_LAYERS} layers")
    print(f"Training years: {TRAIN_YEARS} | Tracks: {TRACKS} "
          f"| Total training scenarios: {len(scenario_list)}")

    hparams = HParams()
    result = train(hparams, scenario_list, num_episodes=NUM_EPISODES, seed=SEED,
                    save_checkpoints=True, verbose=True, track_names=track_names)

    print(f"\nSaved -> {os.path.join(MODEL_DIR, 'qrl_f1_strategist_final.pt')}")
    print(f"Best: {result['best_reward']:.2f}")

    # --- Held-out test data: 2025, never seen during training ---
    print(f"\n{'=' * 60}")
    print(f"Evaluating on held-out TEST year: {TEST_YEAR}")
    print(f"{'=' * 60}")
    test_scenarios = load_scenarios_for_years([TEST_YEAR], TRACKS)

    final_evaluation(result['policy'], result['obs_normalizer'], test_scenarios)

    from evaluate_qrl import run_baseline_comparison
    run_baseline_comparison(
        result['policy'], result['obs_normalizer'], test_scenarios, DEVICE,
        out_dir=os.path.join(LOG_DIR, f"test_{TEST_YEAR}"),
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "search"], default="train")
    parser.add_argument("--search-trials", type=int, default=SEARCH_N_TRIALS)
    parser.add_argument("--search-episodes", type=int, default=SEARCH_EPISODES)
    parser.add_argument("--no-train-after-search", action="store_true")
    args = parser.parse_args()

    if args.mode == "search":
        os.makedirs(MODEL_DIR, exist_ok=True)
        os.makedirs(LOG_DIR, exist_ok=True)
        train_scenarios = load_scenarios_for_years(TRAIN_YEARS, TRACKS)
        scenario_list = list(train_scenarios.values())
        track_names = list(train_scenarios.keys())
        run_hyperparameter_search(
            scenario_list,
            track_names=track_names,
            n_trials=args.search_trials,
            search_episodes=args.search_episodes,
            train_after_search=not args.no_train_after_search,
        )
    else:
        main()
