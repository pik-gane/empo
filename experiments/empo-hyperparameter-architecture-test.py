#!/usr/bin/env python3
"""
EMPO Phase 2: Comparing Training Methods
Ground Truth (Backward Induction) vs Simultaneous Updates vs Inner-Outer Loop
"""

import sys, os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless server
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from collections import defaultdict
import time
import types
import pickle
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Setup paths
# ============================================================
sys.path.insert(0, 'src')
sys.path.insert(0, 'vendor/multigrid')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"PyTorch: {torch.__version__}")
print(f"Device: {DEVICE}")

SAVE_DIR = 'results'
os.makedirs(SAVE_DIR, exist_ok=True)

# ============================================================
# Patch ReachCellGoal BEFORE any computation
# ============================================================
from empo.world_specific_helpers.multigrid import ReachCellGoal

ReachCellGoal.__eq__ = lambda self, other: (
    isinstance(other, ReachCellGoal) and self.target_pos == other.target_pos
)
ReachCellGoal.__hash__ = lambda self: hash(('ReachCellGoal', self.target_pos))
print("[✓] ReachCellGoal __eq__/__hash__ patched")

# ============================================================
# Patch unreachable goals (assertion → graceful fallback)
# ============================================================
import empo.backward_induction.phase2 as _phase2

_orig_process = _phase2._rp_process_single_state

def _patched_process(state_index, state, states, state_transitions, Vh_values, Vr_values,
                     human_agent_indices, robot_agent_indices, robot_action_profiles,
                     possible_goal_generator, num_agents, num_actions, action_powers,
                     human_policy_prior, beta_r, gamma_h, gamma_r, zeta, xi, eta, terminal_Vr,
                     slice_cache=None, use_indexed=False, vres0=None, compute_successor_probs=True):
    try:
        return _orig_process(
            state_index, state, states, state_transitions, Vh_values, Vr_values,
            human_agent_indices, robot_agent_indices, robot_action_profiles,
            possible_goal_generator, num_agents, num_actions, action_powers,
            human_policy_prior, beta_r, gamma_h, gamma_r, zeta, xi, eta, terminal_Vr,
            slice_cache=slice_cache, use_indexed=use_indexed, vres0=vres0,
            compute_successor_probs=compute_successor_probs)
    except (AssertionError, ValueError) as e:
        if "No goal achievable" in str(e) or "xh=0" in str(e):
            vh_results = {agent_idx: {} for agent_idx in human_agent_indices}
            return vh_results, terminal_Vr, None, {}
        raise

_phase2._rp_process_single_state = _patched_process
print("[✓] Unreachable goals patch applied")

# ============================================================
# Configuration
# ============================================================
# EMPO parameters (normative — define the solution)
BETA_R = 5.0
BETA_H = 10.0
GAMMA_R = 0.99
GAMMA_H = 0.99
ZETA = 2.0
XI = 1.0
ETA = 1.0

# Training parameters
NUM_TRAINING_STEPS = 20000
MAX_STEPS = 10  # env max_steps override

# ============================================================
# Load Environment
# ============================================================
from gym_multigrid.multigrid import MultiGridEnv, SmallActions

ENV_YAML = "multigrid_worlds/jobst_challenges/asymmetric_freeing_simple.yaml"

# Override max_steps in the yaml
import re
with open(ENV_YAML, 'r') as f:
    yaml_content = f.read()
yaml_content = re.sub(r'max_steps:\s*\d+', f'max_steps: {MAX_STEPS}', yaml_content)
with open(ENV_YAML, 'w') as f:
    f.write(yaml_content)

env = MultiGridEnv(config_file=ENV_YAML, partial_obs=False, actions_set=SmallActions)
env.reset()

print(f"\nEnvironment: asymmetric_freeing_simple")
print(f"  Grid: {env.width}x{env.height}")
print(f"  Agents: {len(env.agents)} total")
print(f"  Humans: {env.human_agent_indices}")
print(f"  Robots: {env.robot_agent_indices}")
print(f"  Actions per agent: {env.action_space.n}")
print(f"  Max steps: {env.max_steps}")

# ============================================================
# Phase 1: Human Policy Prior
# ============================================================
from empo.backward_induction import compute_human_policy_prior

print("\n[Phase 1] Computing human policy prior...")
t0 = time.time()

human_policy_prior = compute_human_policy_prior(
    world_model=env,
    human_agent_indices=env.human_agent_indices,
    possible_goal_generator=env.possible_goal_generator,
    beta_h=BETA_H,
    gamma_h=GAMMA_H,
)

print(f"  Done in {time.time()-t0:.1f}s")

# ============================================================
# Goal Sampler
# ============================================================
from empo.possible_goal import TabularGoalSampler

initial_state = env.get_state()
human_idx = env.human_agent_indices[0]
goals_and_weights = list(env.possible_goal_generator.generate(initial_state, human_idx))
goals = [g for g, w in goals_and_weights]
weights = [w for g, w in goals_and_weights]

goal_sampler = TabularGoalSampler(goals, weights=weights)
print(f"  Goals per human: {len(goal_sampler.goals)}")

# ============================================================
# Phase 2 Method 1: Ground Truth (Backward Induction)
# ============================================================
from empo.backward_induction import compute_robot_policy

print("\n[Ground Truth] Computing exact robot policy...")
t0 = time.time()

gt_policy = compute_robot_policy(
    world_model=env,
    human_agent_indices=env.human_agent_indices,
    robot_agent_indices=env.robot_agent_indices,
    human_policy_prior=human_policy_prior,
    possible_goal_generator=env.possible_goal_generator,
    beta_r=BETA_R,
    gamma_r=GAMMA_R,
    gamma_h=GAMMA_H,
    zeta=ZETA,
    xi=XI,
    eta=ETA,
)

t_gt = time.time() - t0
num_states = len(gt_policy.values)
print(f"  Done in {t_gt:.1f}s")
print(f"  States with policy: {num_states}")

gt_non_uniform = {s: p for s, p in gt_policy.values.items()
                  if max(p.values()) - min(p.values()) > 0.01}
print(f"  Non-uniform policies: {len(gt_non_uniform)} "
      f"({100*len(gt_non_uniform)/max(1,num_states):.0f}% of states)")

# Save ground truth
with open(f'{SAVE_DIR}/gt_policy.pkl', 'wb') as f:
    pickle.dump(gt_policy, f)
print(f"  Saved to {SAVE_DIR}/gt_policy.pkl")

# ============================================================
# Phase 2 Method 2: Simultaneous Updates
# ============================================================
from empo.learning_based.phase2.config import Phase2Config
from empo.learning_based.multigrid.phase2.trainer import (
    train_multigrid_phase2, MultiGridPhase2Trainer, create_phase2_networks
)

config_simul = Phase2Config(
    beta_r=BETA_R,
    gamma_r=GAMMA_R,
    gamma_h=GAMMA_H,
    zeta=ZETA,
    xi=XI,
    eta=ETA,
    x_h_use_network=False,
    use_lookup_tables=False,
    use_model_based_targets=False,
    warmup_v_h_e_steps=2000,
    warmup_q_r_steps=2000,
    beta_r_rampup_steps=4000,
    num_training_steps=NUM_TRAINING_STEPS,
    steps_per_episode=env.max_steps,
    batch_size=64,
    buffer_size=10000,
)

print(f"\n[Simultaneous] Training ({NUM_TRAINING_STEPS} steps)...")
env.reset()
t0 = time.time()

q_r_simul, nets_simul, hist_simul, trainer_simul = train_multigrid_phase2(
    world_model=env,
    human_agent_indices=env.human_agent_indices,
    robot_agent_indices=env.robot_agent_indices,
    human_policy_prior=human_policy_prior,
    goal_sampler=goal_sampler,
    config=config_simul,
    device=DEVICE,
    verbose=True,
    tensorboard_dir=f"{SAVE_DIR}/tb_simul",
)

t_simul = time.time() - t0
print(f"  Simultaneous training: {t_simul:.1f}s")

trainer_simul.save_all_networks(f'{SAVE_DIR}/checkpoint_simul.pt')
with open(f'{SAVE_DIR}/hist_simul.pkl', 'wb') as f:
    pickle.dump(hist_simul, f)

# ============================================================
# Phase 2 Method 3: Inner-Outer Loop
# ============================================================
def _apply_inner_outer_patch(trainer, inner_steps=5, tol=0.005):
    """Monkey-patch the trainer to use inner-outer loop."""
    trainer._inner_loop_steps = inner_steps
    trainer._inner_loop_tol = tol
    trainer._inner_loop_verbose = False
    trainer._original_training_step = trainer.training_step

    def _io_training_step(self):
        config = self.config
        step = self.training_step_count
        active = config.get_active_networks(step)
        if 'q_r' not in active or 'v_h_e' not in active:
            return self._original_training_step()

        x_h_bs = config.x_h_batch_size or config.batch_size
        if len(self.replay_buffer) < max(config.batch_size, x_h_bs):
            return {}, {}, {}

        batch = self.replay_buffer.sample(config.batch_size)
        x_h_batch = self.replay_buffer.sample(x_h_bs) if x_h_bs > config.batch_size else batch

        # --- INNER LOOP: V_h^e only ---
        prev_preds = None
        for inner_step in range(self._inner_loop_steps):
            losses, _ = self.compute_losses(batch, x_h_batch)
            inner_nets = []
            if 'v_h_e' in self.optimizers and losses.get('v_h_e') is not None and losses['v_h_e'].requires_grad:
                inner_nets.append(('v_h_e', losses['v_h_e']))
            if config.x_h_use_network and 'x_h' in active and 'x_h' in self.optimizers:
                if losses.get('x_h') is not None and losses['x_h'].requires_grad:
                    inner_nets.append(('x_h', losses['x_h']))
            if not inner_nets:
                break

            for n, _ in inner_nets:
                self.optimizers[n].zero_grad()
            for i, (n, l) in enumerate(inner_nets):
                l.backward(retain_graph=(i < len(inner_nets)-1))
            self._apply_adaptive_lr_scaling()

            net_map = {'v_h_e': self.networks.v_h_e}
            if config.x_h_use_network and self.networks.x_h is not None:
                net_map['x_h'] = self.networks.x_h
            for n, l in inner_nets:
                self.update_counts[n] += 1
                net = net_map.get(n)
                use_alr = config.lookup_use_adaptive_lr and net and hasattr(net, 'scale_gradients_by_update_count')
                lr = 1.0 if use_alr else config.get_learning_rate(n, step, self.update_counts[n])
                for pg in self.optimizers[n].param_groups:
                    pg['lr'] = lr
                if net:
                    cv = config.get_effective_grad_clip(n, lr)
                    if cv and cv > 0:
                        torch.nn.utils.clip_grad_norm_(net.parameters(), cv)
                self.optimizers[n].step()

            self.networks.v_h_e_target.load_state_dict(self.networks.v_h_e.state_dict())
            self.networks.v_h_e_target.eval()
            if config.x_h_use_network and self.networks.x_h_target is not None:
                self.networks.x_h_target.load_state_dict(self.networks.x_h.state_dict())
                self.networks.x_h_target.eval()

            with torch.no_grad():
                ss, hh, gg = [], [], []
                for t in batch:
                    for h, g in t.goals.items():
                        ss.append(t.state); hh.append(h); gg.append(g)
                if ss:
                    cur = self.networks.v_h_e.forward_batch(ss, gg, hh, self.env, self.device).squeeze().cpu().numpy()
                    if prev_preds is not None and len(cur) == len(prev_preds):
                        if np.max(np.abs(cur - prev_preds)) < self._inner_loop_tol:
                            break
                    prev_preds = cur.copy()
            self._add_new_lookup_params_to_optimizers()

        # --- OUTER LOOP: Q_r update ---
        losses, pred_stats = self.compute_losses(batch, x_h_batch)
        loss_vals, grad_norms = {}, {}
        outer_nets = []
        for n, l in losses.items():
            loss_vals[n] = l.item()
            if n in self.optimizers and l.requires_grad and n in active and n not in ('v_h_e', 'x_h'):
                outer_nets.append((n, l))

        full_map = {'q_r': self.networks.q_r, 'v_h_e': self.networks.v_h_e}
        if config.x_h_use_network and self.networks.x_h is not None:
            full_map['x_h'] = self.networks.x_h
        if config.u_r_use_network:
            full_map['u_r'] = self.networks.u_r
        if config.v_r_use_network:
            full_map['v_r'] = self.networks.v_r
        if config.use_rnd and self.networks.rnd is not None:
            full_map['rnd'] = self.networks.rnd.predictor

        for n, _ in outer_nets:
            self.optimizers[n].zero_grad()
        for i, (n, l) in enumerate(outer_nets):
            l.backward(retain_graph=(i < len(outer_nets)-1))
        self._apply_adaptive_lr_scaling()
        states = [t.state for t in batch]
        rnd_lr = self._apply_rnd_adaptive_lr_scaling(states, active)

        for n, l in outer_nets:
            self.update_counts[n] += 1
            net = full_map.get(n)
            use_alr = config.lookup_use_adaptive_lr and net and hasattr(net, 'scale_gradients_by_update_count')
            lr = 1.0 if use_alr else config.get_learning_rate(n, step, self.update_counts[n])
            for pg in self.optimizers[n].param_groups:
                pg['lr'] = lr
            if net:
                cv = config.get_effective_grad_clip(n, lr)
                if cv and cv > 0:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), cv)
            grad_norms[n] = self._compute_single_grad_norm(n)
            self.optimizers[n].step()

        self.update_target_networks()
        self._add_new_lookup_params_to_optimizers()
        if rnd_lr:
            pred_stats['rnd_adaptive_lr'] = rnd_lr
        return loss_vals, grad_norms, pred_stats

    trainer.training_step = types.MethodType(_io_training_step, trainer)


print(f"\n[Inner-Outer] Training ({NUM_TRAINING_STEPS} steps)...")
env.reset()

nets_io = create_phase2_networks(
    env=env, config=config_simul,
    num_robots=len(env.robot_agent_indices),
    num_actions=env.action_space.n,
    device=DEVICE,
)

trainer_io = MultiGridPhase2Trainer(
    env=env, networks=nets_io, config=config_simul,
    human_agent_indices=env.human_agent_indices,
    robot_agent_indices=env.robot_agent_indices,
    human_policy_prior=human_policy_prior,
    goal_sampler=goal_sampler,
    device=DEVICE, verbose=True,
    tensorboard_dir=f"{SAVE_DIR}/tb_io",
)

_apply_inner_outer_patch(trainer_io, inner_steps=5, tol=0.005)
print("[✓] Inner-outer loop patch applied")

t0 = time.time()
hist_io = trainer_io.train(NUM_TRAINING_STEPS)
t_io = time.time() - t0
print(f"  Inner-Outer training: {t_io:.1f}s")

trainer_io.save_all_networks(f'{SAVE_DIR}/checkpoint_io.pt')
with open(f'{SAVE_DIR}/hist_io.pkl', 'wb') as f:
    pickle.dump(hist_io, f)

# ============================================================
# Evaluation
# ============================================================
def evaluate_against_gt(trainer, gt_policy, beta_r, label):
    results = []
    for state, gt_adist in gt_policy.values.items():
        gt_probs = np.array([gt_adist.get((a,), 0.25) for a in range(4)])
        learned_q = trainer.get_q_r(state, env)
        learned_pi = trainer.get_pi_r(state, env, beta_r=beta_r)
        is_uniform = (np.max(gt_probs) - np.min(gt_probs)) < 0.01
        timestep = state[0] if isinstance(state, tuple) else -1
        results.append({
            'state': state, 'timestep': timestep,
            'gt_probs': gt_probs, 'gt_best': np.argmax(gt_probs),
            'learned_q': learned_q, 'learned_pi': learned_pi,
            'learned_best': np.argmax(learned_pi),
            'gt_is_uniform': is_uniform,
            'kl_div': np.sum(gt_probs * np.log((gt_probs + 1e-10) / (learned_pi + 1e-10))),
        })

    non_uniform = [r for r in results if not r['gt_is_uniform']]
    agreement = (sum(r['gt_best'] == r['learned_best'] for r in non_uniform)
                 / max(1, len(non_uniform)))
    q_std = np.std([q for r in results for q in r['learned_q']])
    mean_kl = np.mean([r['kl_div'] for r in non_uniform]) if non_uniform else 0

    by_t = defaultdict(list)
    for r in non_uniform:
        by_t[r['timestep']].append(r['gt_best'] == r['learned_best'])
    per_t = {t: np.mean(v) for t, v in sorted(by_t.items())}

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Total states:                {len(results)}")
    print(f"  Non-uniform states:          {len(non_uniform)}")
    print(f"  Policy agreement (non-unif): {agreement:.1%}")
    print(f"  Q_r std:                     {q_std:.4f}")
    print(f"  Mean KL divergence:          {mean_kl:.4f}")
    print(f"  Per-timestep agreement:")
    for t, a in per_t.items():
        n = len(by_t[t])
        bar = '█' * int(a * 20) + '░' * (20 - int(a * 20))
        print(f"    t={t}: {bar} {a:.0%} ({n} states)")

    return {
        'label': label, 'agreement': agreement, 'q_std': q_std,
        'mean_kl': mean_kl, 'per_timestep': per_t,
        'results': results, 'non_uniform': non_uniform,
    }


print("\n" + "=" * 60)
print("DIAGNOSTIC COMPARISON")
print("=" * 60)

eval_simul = evaluate_against_gt(trainer_simul, gt_policy, BETA_R, "SIMULTANEOUS")
eval_io = evaluate_against_gt(trainer_io, gt_policy, BETA_R, "INNER-OUTER LOOP")

# Save evaluation results
with open(f'{SAVE_DIR}/eval_results.pkl', 'wb') as f:
    pickle.dump({'simul': eval_simul, 'io': eval_io, 't_gt': t_gt, 't_simul': t_simul, 't_io': t_io}, f)

# ============================================================
# Plots
# ============================================================
sns.set_theme(style="whitegrid", font_scale=1.1)
fig = plt.figure(figsize=(18, 14))
gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.3)

# Plot 1: Policy Agreement
ax1 = fig.add_subplot(gs[0, 0])
methods = ['Random\nBaseline', 'Simultaneous', 'Inner-Outer']
agreements = [0.25, eval_simul['agreement'], eval_io['agreement']]
colors = ['#95a5a6', '#e74c3c', '#2ecc71']
bars = ax1.bar(methods, agreements, color=colors, edgecolor='white', linewidth=1.5)
ax1.set_ylim(0, 1)
ax1.set_ylabel('Policy Agreement with Ground Truth')
ax1.set_title('Best-Action Agreement\n(non-uniform states)', fontweight='bold')
ax1.axhline(y=0.25, color='gray', linestyle='--', alpha=0.5)
for bar, val in zip(bars, agreements):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f'{val:.0%}', ha='center', fontweight='bold', fontsize=12)

# Plot 2: Q_r Distribution
ax2 = fig.add_subplot(gs[0, 1])
q_simul = [q for r in eval_simul['results'] for q in r['learned_q']]
q_io = [q for r in eval_io['results'] for q in r['learned_q']]
ax2.hist(q_simul, bins=50, alpha=0.6, color='#e74c3c',
         label=f'Simultaneous (σ={eval_simul["q_std"]:.3f})', density=True)
ax2.hist(q_io, bins=50, alpha=0.6, color='#2ecc71',
         label=f'Inner-Outer (σ={eval_io["q_std"]:.3f})', density=True)
ax2.set_xlabel('Q_r value')
ax2.set_ylabel('Density')
ax2.set_title('Q_r Value Distribution', fontweight='bold')
ax2.legend(fontsize=9)

# Plot 3: Per-Timestep Agreement
ax3 = fig.add_subplot(gs[0, 2])
all_t = sorted(set(eval_simul['per_timestep'].keys()) | set(eval_io['per_timestep'].keys()))
ax3.plot(all_t, [eval_simul['per_timestep'].get(t, 0) for t in all_t],
         'o-', color='#e74c3c', label='Simultaneous', linewidth=2, markersize=6)
ax3.plot(all_t, [eval_io['per_timestep'].get(t, 0) for t in all_t],
         's-', color='#2ecc71', label='Inner-Outer', linewidth=2, markersize=6)
ax3.axhline(y=0.25, color='gray', linestyle='--', alpha=0.5)
ax3.set_xlabel('Timestep')
ax3.set_ylabel('Agreement')
ax3.set_ylim(-0.05, 1.05)
ax3.set_title('Agreement by Timestep', fontweight='bold')
ax3.legend()

# Plot 4: KL Divergence
ax4 = fig.add_subplot(gs[1, 0])
kl_simul = [r['kl_div'] for r in eval_simul['non_uniform']]
kl_io = [r['kl_div'] for r in eval_io['non_uniform']]
parts = ax4.violinplot([kl_simul, kl_io], positions=[0, 1], showmeans=True, showmedians=True)
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(['#e74c3c', '#2ecc71'][i])
    pc.set_alpha(0.6)
ax4.set_xticks([0, 1])
ax4.set_xticklabels(['Simultaneous', 'Inner-Outer'])
ax4.set_ylabel('KL Divergence from Ground Truth')
ax4.set_title('Policy Distance (lower is better)', fontweight='bold')

# Plot 5: Summary Table
ax5 = fig.add_subplot(gs[1, 1])
ax5.axis('off')
table_data = [
    ['Metric', 'Simultaneous', 'Inner-Outer', 'Target'],
    ['Agreement', f'{eval_simul["agreement"]:.1%}', f'{eval_io["agreement"]:.1%}', '>50%'],
    ['Q_r std', f'{eval_simul["q_std"]:.4f}', f'{eval_io["q_std"]:.4f}', '>> 0.03'],
    ['Mean KL', f'{eval_simul["mean_kl"]:.3f}', f'{eval_io["mean_kl"]:.3f}', '< 0.5'],
    ['Time', f'{t_simul:.0f}s', f'{t_io:.0f}s', '—'],
]
table = ax5.table(cellText=table_data, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 1.8)
for j in range(4):
    table[0, j].set_facecolor('#34495e')
    table[0, j].set_text_props(color='white', fontweight='bold')
for i in range(1, 5):
    table[i, 1].set_facecolor('#fde8e8')
    table[i, 2].set_facecolor('#e8fde8')
ax5.set_title('Summary', fontweight='bold', fontsize=13, pad=20)

# Plot 6: Training Loss Curves
ax6 = fig.add_subplot(gs[1, 2])
if hist_simul:
    steps_s = range(0, len(hist_simul) * 100, 100)
    ax6.plot(steps_s, [h.get('v_h_e', 0) for h in hist_simul],
             color='#e74c3c', alpha=0.4, linewidth=0.8, label='Simul V_h^e')
    ax6.plot(steps_s, [h.get('q_r', 0) for h in hist_simul],
             color='#e74c3c', alpha=0.8, linewidth=1.2, linestyle='--', label='Simul Q_r')
if hist_io:
    steps_io = range(0, len(hist_io) * 100, 100)
    ax6.plot(steps_io, [h.get('v_h_e', 0) for h in hist_io],
             color='#2ecc71', alpha=0.4, linewidth=0.8, label='I-O V_h^e')
    ax6.plot(steps_io, [h.get('q_r', 0) for h in hist_io],
             color='#2ecc71', alpha=0.8, linewidth=1.2, linestyle='--', label='I-O Q_r')
ax6.set_xlabel('Training Step')
ax6.set_ylabel('Loss')
ax6.set_title('Training Loss Curves', fontweight='bold')
ax6.legend(fontsize=8)
ax6.set_yscale('log')

fig.suptitle('EMPO Phase 2: Simultaneous vs Inner-Outer Loop',
             fontsize=16, fontweight='bold', y=0.98)
plt.savefig(f'{SAVE_DIR}/empo_comparison.png', dpi=150, bbox_inches='tight')
print(f"\n[✓] Plot saved to {SAVE_DIR}/empo_comparison.png")

# ============================================================
# Worst-Case Analysis
# ============================================================
action_names = ['↑ Up', '→ Right', '↓ Down', '← Left']
decisive = sorted(eval_io['non_uniform'],
                  key=lambda r: np.max(r['gt_probs']), reverse=True)[:10]

print(f"\n{'='*80}")
print(f"TOP 10 MOST DECISIVE GROUND-TRUTH STATES")
print(f"{'='*80}")

for i, r in enumerate(decisive):
    gt_best = action_names[r['gt_best']]
    s_idx = next(j for j, s in enumerate(eval_simul['results']) if s['state'] == r['state'])
    s_best = action_names[np.argmax(eval_simul['results'][s_idx]['learned_pi'])]
    io_best = action_names[r['learned_best']]
    gt_match_s = '✅' if r['gt_best'] == np.argmax(eval_simul['results'][s_idx]['learned_pi']) else '❌'
    gt_match_io = '✅' if r['gt_best'] == r['learned_best'] else '❌'

    print(f"\n  State {i+1} (t={r['timestep']}): GT best = {gt_best} "
          f"({r['gt_probs'][r['gt_best']]:.0%})")
    print(f"    GT probs:     [{', '.join(f'{p:.2f}' for p in r['gt_probs'])}]")
    print(f"    Simul policy: [{', '.join(f'{p:.2f}' for p in eval_simul['results'][s_idx]['learned_pi'])}] {gt_match_s}")
    print(f"    I-O policy:   [{', '.join(f'{p:.2f}' for p in r['learned_pi'])}] {gt_match_io}")

# Save config for reproducibility
with open(f'{SAVE_DIR}/config.pkl', 'wb') as f:
    pickle.dump(config_simul, f)

print(f"\n{'='*60}")
print(f"All results saved to {SAVE_DIR}/")
print(f"{'='*60}")
