"""
PufferLib-based Phase 2 trainer for EMPO.

This module orchestrates Phase 2 training using **PufferLib's PuffeRL** class
for the core PPO loop (rollout collection, advantage computation, clipped
surrogate update).  EMPO-specific auxiliary network training (V_h^e, X_h,
U_r) runs alongside PufferLib as a post-train hook.

Key PufferLib integration points:

- ``pufferlib.vector.make()`` creates vectorised environments.
- ``pufferlib.pufferl.PuffeRL`` drives ``evaluate()`` + ``train()``.
- ``EMPOActorCritic.forward(observations, state)`` → ``(logits, value)``
  follows PufferLib's expected policy interface.
- ``pufferlib.pytorch.sample_logits`` handles action sampling.

This module does NOT import or modify the existing DQN-path trainer
(``learning_based.phase2.trainer``).  It reads only base-class interfaces
(``Phase2Transition``, ``Phase2ReplayBuffer``, ``BaseHumanGoalAchievement-
Network``, etc.) that are stable public API.
"""

from __future__ import annotations

import copy
import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# TensorBoard (optional)
try:
    from torch.utils.tensorboard import SummaryWriter

    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False

try:
    import pufferlib
    import pufferlib.vector
    import pufferlib.pufferl
    import pufferlib.emulation
except ImportError:

    class _MissingPufferlibModule:
        """Shim raised at attribute access when ``pufferlib`` is not installed.

        Importing this module will succeed so that test collection and
        non-PPO code paths are not broken.  Any actual usage (attribute
        access) raises a clear ``RuntimeError``.
        """

        def __getattr__(self, name: str) -> Any:
            raise RuntimeError(
                "pufferlib is required to use "
                "empo.learning_based.phase2_ppo.trainer "
                "(the PPO-based Phase 2 trainer), but it is not installed. "
                "Install pufferlib to enable PPO-based Phase 2 training."
            )

    pufferlib = _MissingPufferlibModule()  # type: ignore[assignment]

# Read-only imports from existing DQN path (shared data structures)
from empo.learning_based.phase2.replay_buffer import (
    Phase2ReplayBuffer,
)
from empo.learning_based.phase2.human_goal_ability import (
    BaseHumanGoalAchievementNetwork,
)
from empo.learning_based.phase2.aggregate_goal_ability import (
    BaseAggregateGoalAbilityNetwork,
)
from empo.learning_based.phase2.intrinsic_reward_network import (
    BaseIntrinsicRewardNetwork,
)

from .actor_critic import EMPOActorCritic
from .config import PPOPhase2Config
from .diagnostics import compute_plasticity_diagnostics

logger = logging.getLogger(__name__)

# Floor for X_h values to prevent numerical instability in X_h^{-ξ}
# (division by zero / explosion when X_h → 0).  Matches the DQN path.
_X_H_MIN = 1e-3


# ======================================================================
# Container for auxiliary networks (PPO path)
# ======================================================================


@dataclass
class PPOAuxiliaryNetworks:
    """Container for the auxiliary networks used in the PPO path.

    These are the *same* base-class types as the DQN path but are
    instantiated independently.  The PPO trainer never touches
    ``Phase2Networks`` from the DQN path.
    """

    v_h_e: BaseHumanGoalAchievementNetwork
    x_h: Optional[BaseAggregateGoalAbilityNetwork] = None
    u_r: Optional[BaseIntrinsicRewardNetwork] = None

    # Frozen copies for reward computation during rollouts
    v_h_e_target: Optional[BaseHumanGoalAchievementNetwork] = None
    x_h_target: Optional[BaseAggregateGoalAbilityNetwork] = None
    u_r_target: Optional[BaseIntrinsicRewardNetwork] = None


# ======================================================================
# PPO Phase 2 Trainer  (PufferLib-backed)
# ======================================================================


class PPOPhase2Trainer:
    """PufferLib-backed PPO Phase 2 trainer.

    This trainer uses ``pufferlib.pufferl.PuffeRL`` for the core PPO loop
    and adds EMPO-specific auxiliary-network training on top.

    Parameters
    ----------
    actor_critic : EMPOActorCritic
        The PPO actor-critic network (follows PufferLib policy convention).
    auxiliary_networks : PPOAuxiliaryNetworks
        Auxiliary networks for V_h^e, X_h, U_r.
    config : PPOPhase2Config
        Full PPO Phase 2 configuration.
    device : str
        Torch device string (``'cpu'`` or ``'cuda'``).
    """

    def __init__(
        self,
        actor_critic: EMPOActorCritic,
        auxiliary_networks: PPOAuxiliaryNetworks,
        config: PPOPhase2Config,
        device: str = "cpu",
    ):
        self.actor_critic = actor_critic.to(device)
        self.auxiliary_networks = auxiliary_networks
        self.config = config
        self.device = device

        # Move auxiliary networks to the same device as the actor-critic
        # to prevent device-mismatch errors in forward() calls.
        for net_name in ("v_h_e", "x_h", "u_r"):
            net = getattr(auxiliary_networks, net_name, None)
            if net is not None and hasattr(net, "to"):
                net.to(device)

        # Auxiliary optimisers
        self.aux_optimizers: Dict[str, optim.Optimizer] = {}
        self.aux_optimizers["v_h_e"] = optim.Adam(
            auxiliary_networks.v_h_e.parameters(),
            lr=config.lr_v_h_e,
            weight_decay=config.v_h_e_weight_decay,
        )
        if auxiliary_networks.x_h is not None:
            self.aux_optimizers["x_h"] = optim.Adam(
                auxiliary_networks.x_h.parameters(),
                lr=config.lr_x_h,
                weight_decay=config.x_h_weight_decay,
            )
        if auxiliary_networks.u_r is not None:
            self.aux_optimizers["u_r"] = optim.Adam(
                auxiliary_networks.u_r.parameters(),
                lr=config.lr_u_r,
                weight_decay=config.u_r_weight_decay,
            )

        # Auxiliary replay buffer
        self.aux_replay_buffer = Phase2ReplayBuffer(capacity=config.aux_buffer_size)

        # Counters
        self.global_env_step: int = 0
        self.ppo_iteration: int = 0
        self.aux_training_step: int = 0  # cumulative warm-up + in-loop aux steps

        # TensorBoard writer (initialised lazily in train())
        self.writer: Optional["SummaryWriter"] = None

    # ------------------------------------------------------------------
    # Target-network management
    # ------------------------------------------------------------------

    def freeze_auxiliary_networks(self) -> None:
        """Create frozen copies of auxiliary networks for reward computation."""
        nets = self.auxiliary_networks
        nets.v_h_e_target = copy.deepcopy(nets.v_h_e)
        nets.v_h_e_target.eval()
        for p in nets.v_h_e_target.parameters():
            p.requires_grad = False
        if hasattr(nets.v_h_e_target, "to"):
            nets.v_h_e_target.to(self.device)
        if nets.x_h is not None:
            nets.x_h_target = copy.deepcopy(nets.x_h)
            nets.x_h_target.eval()
            for p in nets.x_h_target.parameters():
                p.requires_grad = False
            if hasattr(nets.x_h_target, "to"):
                nets.x_h_target.to(self.device)
        if nets.u_r is not None:
            nets.u_r_target = copy.deepcopy(nets.u_r)
            nets.u_r_target.eval()
            for p in nets.u_r_target.parameters():
                p.requires_grad = False
            if hasattr(nets.u_r_target, "to"):
                nets.u_r_target.to(self.device)

    # ------------------------------------------------------------------
    # Auxiliary training
    # ------------------------------------------------------------------

    def push_transition_to_aux_buffer(
        self,
        state: Any,
        next_state: Any,
        robot_action: tuple,
        goals: dict,
        goal_weights: dict,
        human_actions: list,
        transition_probs: Optional[dict],
        terminal: bool,
    ) -> None:
        """Push a single transition into the auxiliary replay buffer."""
        self.aux_replay_buffer.push(
            state=state,
            robot_action=robot_action,
            goals=goals,
            goal_weights=goal_weights,
            human_actions=human_actions,
            next_state=next_state,
            transition_probs_by_action=transition_probs,
            terminal=terminal,
        )

    def train_auxiliary_step(
        self,
        world_model: Any = None,
        active_networks: Optional[Set[str]] = None,
    ) -> Dict[str, float]:
        """Run one gradient step on the auxiliary networks.

        Mirrors the DQN path's ``compute_losses`` but uses the base-class
        ``forward()`` interface (single-item, not ``forward_batch``).

        Parameters
        ----------
        world_model : WorldModel or None
            A world model instance used for ``forward()`` calls on the
            auxiliary networks.  When ``None``, V_h^e and X_h forward passes
            are skipped (useful for unit testing the plumbing).
        active_networks : set[str] or None
            Subset of ``{"v_h_e", "x_h", "u_r"}`` to train on this step.
            When ``None`` (default), all available networks are trained
            (backward-compatible behaviour).

        Returns
        -------
        losses : dict[str, float]
        """
        if len(self.aux_replay_buffer) < self.config.batch_size:
            return {}

        batch = self.aux_replay_buffer.sample(self.config.batch_size)
        cfg = self.config
        nets = self.auxiliary_networks
        losses: Dict[str, float] = {}

        # -----------------------------------------------------------------
        # V_h^e loss: MSE between V_h^e(s, g_h) and TD target
        # -----------------------------------------------------------------
        if world_model is not None and (
            active_networks is None or "v_h_e" in active_networks
        ):
            v_h_e_preds: List[torch.Tensor] = []
            v_h_e_targets: List[torch.Tensor] = []

            for t in batch:
                for h_idx, goal in t.goals.items():
                    # Forward pass (online network)
                    pred = nets.v_h_e(t.state, world_model, h_idx, goal, self.device)
                    v_h_e_preds.append(pred.squeeze())

                    # Target from target network
                    with torch.no_grad():
                        target_net = nets.v_h_e_target or nets.v_h_e
                        # Check if goal achieved in next_state
                        achieved = (
                            goal.is_achieved(t.next_state)
                            if hasattr(goal, "is_achieved")
                            else 0
                        )
                        if achieved:
                            target = torch.tensor(1.0, device=self.device)
                        elif t.terminal:
                            target = torch.tensor(0.0, device=self.device)
                        else:
                            next_v = target_net(
                                t.next_state,
                                world_model,
                                h_idx,
                                goal,
                                self.device,
                            )
                            next_v = target_net.apply_hard_clamp(next_v)
                            target = cfg.gamma_h * next_v.squeeze()
                    v_h_e_targets.append(target)

            if v_h_e_preds:
                preds_t = torch.stack(v_h_e_preds)
                targets_t = torch.stack(v_h_e_targets)
                loss_v = ((preds_t - targets_t) ** 2).mean()

                opt = self.aux_optimizers["v_h_e"]
                opt.zero_grad()
                loss_v.backward()
                if cfg.v_h_e_grad_clip is not None:
                    nn.utils.clip_grad_norm_(
                        nets.v_h_e.parameters(), cfg.v_h_e_grad_clip
                    )
                opt.step()
                losses["v_h_e_loss"] = loss_v.item()

        # -----------------------------------------------------------------
        # X_h loss: MSE between X_h(s, h) and w_h * V_h^e(s, g_h)^ζ
        # -----------------------------------------------------------------
        if (
            nets.x_h is not None
            and world_model is not None
            and "x_h" in self.aux_optimizers
            and (active_networks is None or "x_h" in active_networks)
        ):
            x_h_preds: List[torch.Tensor] = []
            x_h_targets_list: List[torch.Tensor] = []

            for t in batch:
                for h_idx, goal in t.goals.items():
                    weight = t.goal_weights.get(h_idx, 1.0)
                    pred = nets.x_h(t.state, world_model, h_idx, self.device)
                    x_h_preds.append(pred.squeeze())

                    with torch.no_grad():
                        v_target_net = nets.v_h_e_target or nets.v_h_e
                        v_for_x = v_target_net(
                            t.state, world_model, h_idx, goal, self.device
                        )
                        v_for_x = v_target_net.apply_hard_clamp(v_for_x)
                        x_target = nets.x_h.compute_target(
                            v_for_x.squeeze(),
                            goal_weight=weight,
                        )
                    x_h_targets_list.append(x_target)

            if x_h_preds:
                xp = torch.stack(x_h_preds)
                xt = torch.stack(x_h_targets_list)
                loss_x = ((xp - xt) ** 2).mean()

                opt = self.aux_optimizers["x_h"]
                opt.zero_grad()
                loss_x.backward()
                if cfg.x_h_grad_clip is not None:
                    nn.utils.clip_grad_norm_(nets.x_h.parameters(), cfg.x_h_grad_clip)
                opt.step()
                losses["x_h_loss"] = loss_x.item()

        # -----------------------------------------------------------------
        # U_r loss: MSE between predicted y and target y = E_h[X_h^{-ξ}]
        # -----------------------------------------------------------------
        if (
            nets.u_r is not None
            and world_model is not None
            and "u_r" in self.aux_optimizers
            and (active_networks is None or "u_r" in active_networks)
        ):
            y_preds: List[torch.Tensor] = []
            y_targets: List[torch.Tensor] = []

            x_h_target_net = nets.x_h_target or nets.x_h
            for t in batch:
                y_pred, _ = nets.u_r(t.state, world_model, self.device)
                y_preds.append(y_pred.squeeze())

                with torch.no_grad():
                    x_vals: List[float] = []
                    if x_h_target_net is not None:
                        for h_idx in t.goals.keys():
                            xv = x_h_target_net(
                                t.state, world_model, h_idx, self.device
                            )
                            xv = x_h_target_net.apply_hard_clamp(xv)
                            x_vals.append(max(xv.squeeze().item(), _X_H_MIN))
                    if x_vals:
                        x_t = torch.tensor(x_vals, device=self.device)
                        y_t = (x_t ** (-cfg.xi)).mean()
                    else:
                        y_t = torch.tensor(1.0, device=self.device)
                y_targets.append(y_t)

            if y_preds:
                yp = torch.stack(y_preds)
                yt = torch.stack(y_targets)
                loss_u = ((yp - yt) ** 2).mean()

                opt = self.aux_optimizers["u_r"]
                opt.zero_grad()
                loss_u.backward()
                if cfg.u_r_grad_clip is not None:
                    nn.utils.clip_grad_norm_(nets.u_r.parameters(), cfg.u_r_grad_clip)
                opt.step()
                losses["u_r_loss"] = loss_u.item()

        return losses

    # ------------------------------------------------------------------
    # Auxiliary network ↔ environment synchronisation
    # ------------------------------------------------------------------

    @staticmethod
    def _sync_aux_nets_to_envs(vecenv: Any) -> None:
        """Ensure each env in the vectorised pool has the latest auxiliary_networks.

        After ``freeze_auxiliary_networks()`` creates frozen target copies
        (which are stored as new attributes on the ``PPOAuxiliaryNetworks``
        dataclass), the env instances that hold a reference to the *same*
        dataclass object automatically see the updated targets.

        This method is a no-op for the default wiring (where the env_creator
        closure captures the ``auxiliary_networks`` reference by object identity).
        It is kept as an explicit synchronisation point so that future
        implementations using multiprocessing backends can override it
        to serialise / ship the updated networks to worker processes.
        """
        # In the Serial backend the envs share the same process and the
        # same PPOAuxiliaryNetworks object, so frozen target updates are
        # immediately visible.  Nothing to do here.
        pass

    def _collect_aux_data_from_rollout(self, pufferl: Any, vecenv: Any) -> None:
        """Extract auxiliary transition data from environment aux buffers.

        Each :class:`EMPOWorldModelEnv` stores per-step auxiliary data in its
        ``_aux_buffer`` attribute.  Under the Serial backend the envs live
        in the same process, so we can read their buffers directly.  This
        avoids PufferLib's info aggregation which only handles numeric
        scalars.

        After reading, each env's ``_aux_buffer`` is drained so that
        transitions are not pushed twice.
        """
        # Access the underlying env instances through PufferLib's vecenv.
        # PufferLib Serial backend exposes ``envs`` as a list of env
        # instances.  ``single_env`` is used by some wrappers that hold
        # a single reference.
        envs = getattr(vecenv, "envs", None)
        if envs is None:
            envs = getattr(vecenv, "single_env", None)
            if envs is not None:
                envs = [envs]
        if not envs:
            return

        _MAX_UNWRAP_DEPTH = 20

        for env in envs:
            # Unwrap PufferLib emulation layers to reach EMPOWorldModelEnv.
            # Depth limit prevents infinite loops in misconfigured wrappers.
            inner = env
            depth = 0
            while hasattr(inner, "env") and depth < _MAX_UNWRAP_DEPTH:
                inner = inner.env
                depth += 1
            aux_buffer = getattr(inner, "_aux_buffer", None)
            if aux_buffer is None:
                continue

            while aux_buffer:
                info = aux_buffer.popleft()
                state = info.get("state")
                next_state = info.get("next_state")
                goals = info.get("goals")
                if state is None or next_state is None or goals is None:
                    continue

                goal_weights = info.get("goal_weights", {})
                human_actions = info.get("human_actions", [])
                transition_probs = info.get("transition_probs")

                # Prefer the combined terminal flag that accounts for both
                # environment termination and episode truncation (from
                # steps_per_episode).  Fall back to terminated-only for
                # backward compatibility with older aux-buffer schemas.
                terminal = info.get("terminal")
                if terminal is None:
                    terminated = info.get("terminated", False)
                    truncated = info.get("truncated", False)
                    terminal = bool(terminated or truncated)

                # robot_action should already be a per-robot tuple (stored
                # by EMPOWorldModelEnv._aux_buffer).  Normalise for safety.
                robot_action = info.get("robot_action", (0,))
                if isinstance(robot_action, int):
                    robot_action = (robot_action,)
                elif isinstance(robot_action, (list, tuple)):
                    robot_action = tuple(robot_action)

                self.push_transition_to_aux_buffer(
                    state=state,
                    next_state=next_state,
                    robot_action=robot_action,
                    goals=goals,
                    goal_weights=goal_weights,
                    human_actions=human_actions,
                    transition_probs=transition_probs,
                    terminal=terminal,
                )

    # ------------------------------------------------------------------
    # TensorBoard helpers
    # ------------------------------------------------------------------

    def _init_tensorboard(self) -> None:
        """Initialise the TensorBoard writer if configured."""
        tb_dir = self.config.tensorboard_dir
        if tb_dir is not None and HAS_TENSORBOARD:
            os.makedirs(tb_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=tb_dir)
        else:
            self.writer = None

    def _log_scalar(self, tag: str, value: float, step: int) -> None:
        if self.writer is not None:
            self.writer.add_scalar(tag, value, step)

    def _log_text(self, tag: str, text: str, step: int) -> None:
        if self.writer is not None:
            self.writer.add_text(tag, text, step)

    # ------------------------------------------------------------------
    # Plasticity diagnostics
    # ------------------------------------------------------------------

    @staticmethod
    def _unwrap_empo_env(env: Any) -> Any:
        """Unwrap PufferLib emulation layers to the env exposing ``_state_to_obs``."""
        inner = env
        for _ in range(20):
            if inner is None:
                return None
            if hasattr(inner, "_state_to_obs"):
                return inner
            inner = getattr(inner, "env", None)
        return None

    def _log_plasticity_diagnostics(self, vecenv: Any, step: int) -> None:
        """Compute and log plasticity-loss diagnostics for the actor-critic.

        Probes the current policy network with a batch of states drawn from
        the auxiliary replay buffer and logs dormant/dead-neuron fractions,
        the effective rank of the shared representation, and weight norms.
        Cheap enough to run periodically; skipped silently if no logging sink
        or insufficient data is available.
        """
        if self.writer is None:
            return
        n = min(
            len(self.aux_replay_buffer),
            self.config.plasticity_diagnostics_batch_size,
        )
        if n < 2:
            return

        env = self._unwrap_empo_env(vecenv.driver_env)
        if env is None:
            return

        batch = self.aux_replay_buffer.sample(n)
        try:
            obs = np.stack([env._state_to_obs(t.state) for t in batch])
        except NotImplementedError:
            # Base EMPOWorldModelEnv without an observation encoder.
            return

        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        metrics = compute_plasticity_diagnostics(
            self.actor_critic,
            obs_t,
            dormant_tau=self.config.plasticity_dormant_tau,
        )
        for k, v in metrics.items():
            self._log_scalar(f"Plasticity/{k}", v, step)


    # ------------------------------------------------------------------
    # Checkpoint save / load
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str) -> str:
        """Persist EMPO Phase 2 network state and counters to *path*.

        Saves the actor-critic and auxiliary network weights, their target
        copies, auxiliary optimiser states, and EMPO-specific training
        counters. This allows ``load_checkpoint`` to restore these
        components for evaluation or further fine-tuning, but does not
        include PuffeRL/PPO driver optimiser state, rollout buffers, or
        driver-specific training counters. PPO training will therefore
        restart from a fresh driver state after loading.

        Returns the actual path used (may differ if a tempdir fallback
        was needed).
        """
        nets = self.auxiliary_networks
        checkpoint: Dict[str, Any] = {
            "actor_critic": self.actor_critic.state_dict(),
            "v_h_e": nets.v_h_e.state_dict(),
            "global_env_step": self.global_env_step,
            "ppo_iteration": self.ppo_iteration,
            "aux_training_step": self.aux_training_step,
            "config": {
                "gamma_r": self.config.gamma_r,
                "gamma_h": self.config.gamma_h,
                "zeta": self.config.zeta,
                "xi": self.config.xi,
                "eta": self.config.eta,
            },
        }
        # Optional networks
        if nets.x_h is not None:
            checkpoint["x_h"] = nets.x_h.state_dict()
        if nets.u_r is not None:
            checkpoint["u_r"] = nets.u_r.state_dict()
        # Targets
        if nets.v_h_e_target is not None:
            checkpoint["v_h_e_target"] = nets.v_h_e_target.state_dict()
        if nets.x_h_target is not None:
            checkpoint["x_h_target"] = nets.x_h_target.state_dict()
        if nets.u_r_target is not None:
            checkpoint["u_r_target"] = nets.u_r_target.state_dict()
        # Optimiser states
        opt_states = {}
        for name, opt in self.aux_optimizers.items():
            opt_states[name] = opt.state_dict()
        checkpoint["aux_optimizers"] = opt_states

        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        try:
            torch.save(checkpoint, path)
            return path
        except (IOError, OSError, RuntimeError) as exc:
            import tempfile

            fallback = os.path.join(
                tempfile.gettempdir(),
                f"empo_ppo_checkpoint_{os.path.basename(path)}",
            )
            logger.warning(
                "Cannot save to %s: %s — falling back to %s", path, exc, fallback
            )
            torch.save(checkpoint, fallback)
            return fallback

    def load_checkpoint(self, path: str) -> None:
        """Restore trainer state from a checkpoint saved by ``save_checkpoint``.

        Loads auxiliary network weights, targets, optimiser states, and
        training counters.  The ``actor_critic`` weights are loaded as
        well so that PPO can resume from the saved policy.

        .. warning::

           ``weights_only=False`` is used because optimiser state dicts
           contain non-tensor types (step counts, betas).  Only load
           checkpoints from trusted sources.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.actor_critic.load_state_dict(checkpoint["actor_critic"])
        nets = self.auxiliary_networks
        nets.v_h_e.load_state_dict(checkpoint["v_h_e"])
        if "x_h" in checkpoint and nets.x_h is not None:
            nets.x_h.load_state_dict(checkpoint["x_h"])
        if "u_r" in checkpoint and nets.u_r is not None:
            nets.u_r.load_state_dict(checkpoint["u_r"])
        # Targets: recreate from online nets if missing but present in checkpoint
        if "v_h_e_target" in checkpoint:
            if nets.v_h_e_target is None and nets.v_h_e is not None:
                v_h_e_target = copy.deepcopy(nets.v_h_e)
                v_h_e_target.eval()
                for p in v_h_e_target.parameters():
                    p.requires_grad_(False)
                v_h_e_target.to(self.device)
                nets.v_h_e_target = v_h_e_target
            if nets.v_h_e_target is not None:
                nets.v_h_e_target.load_state_dict(checkpoint["v_h_e_target"])
        if "x_h_target" in checkpoint:
            if nets.x_h_target is None and nets.x_h is not None:
                x_h_target = copy.deepcopy(nets.x_h)
                x_h_target.eval()
                for p in x_h_target.parameters():
                    p.requires_grad_(False)
                x_h_target.to(self.device)
                nets.x_h_target = x_h_target
            if nets.x_h_target is not None:
                nets.x_h_target.load_state_dict(checkpoint["x_h_target"])
        if "u_r_target" in checkpoint:
            if nets.u_r_target is None and nets.u_r is not None:
                u_r_target = copy.deepcopy(nets.u_r)
                u_r_target.eval()
                for p in u_r_target.parameters():
                    p.requires_grad_(False)
                u_r_target.to(self.device)
                nets.u_r_target = u_r_target
            if nets.u_r_target is not None:
                nets.u_r_target.load_state_dict(checkpoint["u_r_target"])
        # Optimisers
        opt_states = checkpoint.get("aux_optimizers", {})
        for name, state in opt_states.items():
            if name in self.aux_optimizers:
                self.aux_optimizers[name].load_state_dict(state)
        # Counters
        self.global_env_step = checkpoint.get("global_env_step", 0)
        self.ppo_iteration = checkpoint.get("ppo_iteration", 0)
        self.aux_training_step = checkpoint.get("aux_training_step", 0)

    # ------------------------------------------------------------------
    # Reward-scale calibration
    # ------------------------------------------------------------------

    def calibrate_reward_scale(
        self,
        env_creator: Callable[[], Any],
        n_episodes: int = 20,
    ) -> float:
        """Estimate ``u_r_scale`` by running random episodes.

        Creates a temporary environment, runs *n_episodes* random
        rollouts, and records the largest ``|U_r|`` observed.  The
        result is written to ``self.config.u_r_scale`` so that all
        environments created afterwards use the calibrated scale.

        Call this **before** ``train()`` (or after warm-up completes
        for neural auxiliary networks) so that vectorised envs pick up
        the calibrated value at creation time.

        Returns the calibrated scale factor.
        """
        env = env_creator()
        # Mirror train()'s injection of auxiliary_networks so that
        # _compute_u_r() has the networks it needs to produce non-zero rewards.
        if (
            getattr(env, "auxiliary_networks", None) is None
            and self.auxiliary_networks is not None
        ):
            env.auxiliary_networks = self.auxiliary_networks
        try:
            max_abs: float = 0.0
            for _ in range(n_episodes):
                env.reset()
                for _ in range(self.config.steps_per_episode):
                    # Sample a random action and take an env_step. We rely on the
                    # reward / info returned by env_step so that U_r is computed
                    # exactly once per env_step, avoiding duplicate forward passes
                    # through neural auxiliary networks.
                    action = env.action_space.sample()
                    _, reward, terminated, truncated, info = env.step(action)

                    u_r = None
                    if isinstance(info, dict) and "u_r" in info:
                        u_r = info["u_r"]
                    else:
                        # Fall back to the scalar reward returned by env_step.
                        u_r = reward
                    # Many wrappers expose a scaled u_r; rescale to the underlying
                    # U_r magnitude if _u_r_scale is present.
                    scale_attr = getattr(env, "_u_r_scale", 1.0)
                    try:
                        u_r = float(u_r) * float(scale_attr)
                    except (TypeError, ValueError):
                        # If casting fails, skip this sample.
                        continue
                    max_abs = max(max_abs, abs(float(u_r)))
                    if terminated or truncated:
                        break
        finally:
            env.close()
        scale = max_abs if max_abs > 1e-10 else 1.0
        self.config.u_r_scale = scale
        logger.info("Calibrated u_r_scale = %.6f (from %d episodes)", scale, n_episodes)
        return scale

    # ------------------------------------------------------------------
    # Full training loop  (PufferLib-backed)
    # ------------------------------------------------------------------

    def train(
        self,
        env_creator: Callable[[], Any],
        num_iterations: Optional[int] = None,
    ) -> List[Dict[str, float]]:
        """Run the full PufferLib PPO Phase 2 training loop.

        The loop has two stages:

        1. **Warm-up stage** — auxiliary networks are trained with random
           robot actions (no PPO).  Networks are enabled progressively
           according to ``config.get_active_aux_networks()``.
        2. **PPO stage** — PufferLib drives the PPO loop; auxiliary
           networks continue to be trained alongside.

        Parameters
        ----------
        env_creator : callable
            A zero-argument callable that returns a new Gymnasium-compatible
            :class:`EMPOWorldModelEnv` instance.  The trainer wraps each
            instance with ``pufferlib.emulation.GymnasiumPufferEnv`` and
            passes the wrapped creator to ``pufferlib.vector.make()`` for
            vectorised execution.
        num_iterations : int or None
            Override ``config.num_ppo_iterations``.

        Returns
        -------
        metrics : list[dict]
            Per-iteration loss / metric dictionaries.
        """
        cfg = self.config
        if num_iterations is None:
            n_iters = cfg.num_ppo_iterations
        else:
            n_iters = num_iterations

        # ── TensorBoard ─────────────────────────────────────────────────
        self._init_tensorboard()

        # ── Warm-up phase ────────────────────────────────────────────────
        warmup_metrics = self._run_warmup(env_creator)

        # ── Auto-calibrate U_r reward scale after warm-up ────────────────
        # If no explicit u_r_scale was configured, run a short calibration
        # phase to determine the empirical max|U_r|.  This prevents the
        # conservative theoretical bound (e.g. ~2000 for xi=1, eta=1.1)
        # from crushing actual U_r values (typically 1-10) to near-zero and
        # destroying the PPO reward signal via PufferLib's clamp(r, -1, 1).
        if cfg.u_r_scale is None and cfg.get_total_warmup_steps() > 0:
            calibrated = self.calibrate_reward_scale(env_creator, n_episodes=20)
            logger.info(
                "Auto-calibrated u_r_scale = %.6f after warmup", calibrated
            )

        # ── PufferLib PPO phase ──────────────────────────────────────────
        # Build PufferLib config dict (PuffeRL expects a flat dict).
        # Override device/seed from the trainer to avoid mismatch between
        # the config fields and the actual trainer state.
        puffer_config = cfg.to_pufferlib_config()
        puffer_config["device"] = str(self.device)
        puffer_config["seed"] = cfg.seed
        total_timesteps = n_iters * puffer_config["batch_size"]
        puffer_config["total_timesteps"] = total_timesteps

        # Wrap the Gymnasium env creator with PufferLib's emulation layer.
        # Ensure every env instance is wired to this trainer's auxiliary
        # networks so that U_r rewards and freeze/sync behave correctly.
        aux_nets = self.auxiliary_networks

        def puffer_env_creator(buf=None, seed=0):
            def _create():
                env = env_creator()
                if getattr(env, "auxiliary_networks", None) is None:
                    if aux_nets is None:
                        raise RuntimeError(
                            "PPOPhase2Trainer.train(): environment was created "
                            "without 'auxiliary_networks', and the trainer's "
                            "'auxiliary_networks' attribute is None. Either "
                            "pass auxiliary_networks into your env_creator or "
                            "initialise PPOPhase2Trainer with non-None "
                            "auxiliary networks."
                        )
                    env.auxiliary_networks = aux_nets
                return env

            return pufferlib.emulation.GymnasiumPufferEnv(
                env_creator=_create, buf=buf, seed=seed
            )

        # Create vectorised environments via PufferLib
        vecenv = pufferlib.vector.make(
            puffer_env_creator,
            num_envs=cfg.num_envs,
            backend="Serial",
        )

        # Initialise PuffeRL training driver
        pufferl = pufferlib.pufferl.PuffeRL(puffer_config, vecenv, self.actor_critic)

        # Initial freeze of auxiliary networks and sync into envs
        self.freeze_auxiliary_networks()
        self._sync_aux_nets_to_envs(vecenv)

        all_metrics: List[Dict[str, float]] = warmup_metrics
        iteration = 0
        # Use a driver env for auxiliary forward passes (world_model access).
        # Bounded unwrapping to reach underlying EMPOWorldModelEnv, matching
        # the pattern in _collect_aux_data_from_rollout().
        wm = self._unwrap_world_model(vecenv.driver_env)

        prev_stage = cfg.get_warmup_stage(self.aux_training_step)

        while pufferl.global_step < total_timesteps and iteration < n_iters:
            self.ppo_iteration = iteration

            # --- PufferLib: collect rollout + PPO update ---
            pufferl.evaluate()

            # --- Extract auxiliary data from rollout info dicts ---
            self._collect_aux_data_from_rollout(pufferl, vecenv)

            pufferl.train()

            # --- EMPO-specific: train auxiliary networks ---
            aux_losses: Dict[str, float] = {}
            aux_loss_counts: Dict[str, int] = {}
            active = cfg.get_active_aux_networks(self.aux_training_step)
            for _ in range(cfg.aux_training_steps_per_iteration):
                step_losses = self.train_auxiliary_step(
                    world_model=wm, active_networks=active
                )
                for k, v in step_losses.items():
                    aux_losses[k] = aux_losses.get(k, 0.0) + v
                    aux_loss_counts[k] = aux_loss_counts.get(k, 0) + 1
                # Only advance aux_training_step when a gradient update actually ran.
                if step_losses:
                    self.aux_training_step += 1
            # Average over the number of gradient steps that actually ran,
            # so reported losses are comparable across different
            # aux_training_steps_per_iteration settings.
            for k in aux_losses:
                c = aux_loss_counts.get(k, 1)
                if c > 1:
                    aux_losses[k] /= c

            # Re-freeze auxiliary networks periodically and sync to envs
            if (iteration + 1) % cfg.reward_freeze_interval == 0:
                self.freeze_auxiliary_networks()
                self._sync_aux_nets_to_envs(vecenv)

            metrics = {
                **pufferl.losses,
                **aux_losses,
                "iteration": iteration,
                "global_step": pufferl.global_step,
            }
            all_metrics.append(metrics)
            self.global_env_step = pufferl.global_step
            iteration += 1

            # ── Logging ──────────────────────────────────────────────────
            if iteration % cfg.log_interval == 0:
                step = self.aux_training_step
                for k, v in aux_losses.items():
                    self._log_scalar(f"Loss/{k}", v, step)
                self._log_scalar(
                    "Loss/policy_loss",
                    pufferl.losses.get("policy_loss", 0),
                    step,
                )
                self._log_scalar(
                    "Loss/value_loss",
                    pufferl.losses.get("value_loss", 0),
                    step,
                )
                self._log_scalar("PPO/iteration", iteration, step)
                self._log_scalar("PPO/global_env_step", pufferl.global_step, step)
                self._log_scalar("PPO/entropy_coef", cfg.get_entropy_coef(step), step)
                cur_stage = cfg.get_warmup_stage(step)
                self._log_scalar("Warmup/stage", cur_stage, step)
                if cur_stage != prev_stage:
                    self._log_text(
                        "Warmup/transitions",
                        f"Step {step}: stage {prev_stage} → {cur_stage} "
                        f"({cfg.get_warmup_stage_name(step)})",
                        step,
                    )
                    prev_stage = cur_stage

            # ── Plasticity diagnostics ───────────────────────────────────
            if (
                cfg.plasticity_diagnostics_interval > 0
                and iteration % cfg.plasticity_diagnostics_interval == 0
            ):
                self._log_plasticity_diagnostics(vecenv, self.aux_training_step)

            if iteration % 100 == 0:
                logger.info(
                    "PPO iter %d (step %d): policy_loss=%.4f value_loss=%.4f",
                    iteration,
                    pufferl.global_step,
                    pufferl.losses.get("policy_loss", 0),
                    pufferl.losses.get("value_loss", 0),
                )

            # ── Checkpointing ────────────────────────────────────────────
            if cfg.checkpoint_interval > 0 and iteration % cfg.checkpoint_interval == 0:
                ckpt_dir = cfg.checkpoint_dir or "checkpoints"
                ckpt_path = os.path.join(ckpt_dir, f"ppo_phase2_iter{iteration}.pt")
                self.save_checkpoint(ckpt_path)
                logger.info("Saved checkpoint to %s", ckpt_path)

        pufferl.close()

        if self.writer is not None:
            self.writer.close()
        return all_metrics

    # ------------------------------------------------------------------
    # Warm-up implementation
    # ------------------------------------------------------------------

    def _run_warmup(self, env_creator: Callable[[], Any]) -> List[Dict[str, float]]:
        """Execute the auxiliary-only warm-up phase.

        During warm-up, the robot acts with a **uniform random policy**
        (equivalent to β_r = 0).  Auxiliary networks are trained
        progressively (V_h^e → X_h → U_r) until
        ``config.get_total_warmup_steps()`` gradient updates have been
        performed.

        Returns per-step metric dicts (one per gradient step).
        """
        cfg = self.config
        total_warmup = cfg.get_total_warmup_steps()
        if total_warmup <= 0 or self.aux_training_step >= total_warmup:
            return []

        logger.info(
            "Starting warm-up: %d aux training steps (current=%d)",
            total_warmup,
            self.aux_training_step,
        )

        # Create a single warm-up environment (no PufferLib needed).
        env = env_creator()
        aux_nets = self.auxiliary_networks
        if getattr(env, "auxiliary_networks", None) is None:
            env.auxiliary_networks = aux_nets

        # Freeze targets so reward computation during rollouts is stable.
        self.freeze_auxiliary_networks()

        wm = getattr(env, "world_model", None)

        warmup_metrics: List[Dict[str, float]] = []
        obs, info = env.reset()
        prev_stage = cfg.get_warmup_stage(self.aux_training_step)

        while self.aux_training_step < total_warmup:
            # Detect and log stage transitions
            cur_stage = cfg.get_warmup_stage(self.aux_training_step)
            if cur_stage != prev_stage:
                logger.info(
                    "Warm-up stage %d → %d (%s) at aux step %d",
                    prev_stage,
                    cur_stage,
                    cfg.get_warmup_stage_name(self.aux_training_step),
                    self.aux_training_step,
                )
                self._log_text(
                    "Warmup/transitions",
                    f"Step {self.aux_training_step}: stage {prev_stage} → "
                    f"{cur_stage} ({cfg.get_warmup_stage_name(self.aux_training_step)})",
                    self.aux_training_step,
                )
                # Re-freeze at stage transitions so newly-enabled networks
                # see the latest V_h^e / X_h targets.
                self.freeze_auxiliary_networks()
                prev_stage = cur_stage

            # Step the environment with a uniformly random action.
            action = env.action_space.sample()
            obs, _reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, info = env.reset()

            # Collect aux transition data from the env's _aux_buffer.
            buf = getattr(env, "_aux_buffer", None)
            if buf:
                while buf:
                    td = buf.popleft()
                    robot_action = td.get("robot_action", (action,))
                    if not isinstance(robot_action, tuple):
                        robot_action = (robot_action,)
                    self.push_transition_to_aux_buffer(
                        state=td.get("state"),
                        next_state=td.get("next_state"),
                        robot_action=robot_action,
                        goals=td.get("goals", {}),
                        goal_weights=td.get("goal_weights", {}),
                        human_actions=td.get("human_actions", []),
                        transition_probs=td.get("transition_probs"),
                        terminal=td.get("terminal", False),
                    )

            # Train auxiliary networks (respecting active set).
            active = cfg.get_active_aux_networks(self.aux_training_step)
            losses = self.train_auxiliary_step(world_model=wm, active_networks=active)

            # Only count as a warm-up step when a gradient update ran
            # (replay buffer may still be below batch_size early on).
            if losses:
                self.aux_training_step += 1
                losses["warmup_stage"] = float(cur_stage)
                warmup_metrics.append(losses)

                # TensorBoard logging during warm-up.
                step = self.aux_training_step
                for k, v in losses.items():
                    self._log_scalar(f"Warmup/{k}", v, step)
                self._log_scalar("Warmup/stage", cur_stage, step)

            # Re-freeze periodically during warm-up.
            if self.aux_training_step % max(1, cfg.reward_freeze_interval) == 0:
                self.freeze_auxiliary_networks()

        logger.info(
            "Warm-up complete after %d aux training steps", self.aux_training_step
        )
        env.close()

        # Keep warmup data in the replay buffer.  V_h^e and X_h are
        # state-value functions whose TD targets use the *current* target
        # network regardless of the behaviour policy that generated the
        # transitions.  Retaining warmup data provides the PPO phase with
        # a pre-filled buffer of diverse state coverage, avoiding a cold
        # start where the buffer is too small for auxiliary training.

        return warmup_metrics

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _unwrap_world_model(env: Any) -> Any:
        """Unwrap nested env wrappers to find a ``world_model`` attribute."""
        current = env
        for _ in range(20):
            if current is None:
                return None
            wm = getattr(current, "world_model", None)
            if wm is not None:
                return wm
            if hasattr(current, "env"):
                current = current.env
            else:
                return None
        return None
