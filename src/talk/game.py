"""LanguageGame — the main interface for Talk to GRN.

Supports two modes:
  - Direct mode (env_name provided): single environment, no routing.
  - Router mode (env_name=None): per-prompt environment selection via RouterAgent.

When output_dir is provided, the designed state is set in the actual RL environment,
the observation is derived from it (matching training), and frames are rendered.

Value signal: each chat round draws a random baseline observation, evaluates the
critic on it (V_init) and on the LLM-designed perturbation (V_designed), and
reports the change ΔV = V_designed - V_init. Differences are interpretable
across environments in a way absolute values are not (PPO trains under
VecNormalize, so absolute V lives on a per-env normalized return scale).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.nn as nn

from talk.policy import _DEFAULT_AGENTS_ROOT, list_envs, load_model
from talk.router import RouterAgent, TRAINED_ENVS
from talk.translators import Agent2HumanTranslator, HumanToAgentTranslator


@dataclass
class ChatResult:
    """Full result of a single chat turn."""
    reply: str
    env_name: str
    goal: str
    state: list[float]
    action: list[float]
    delta_v: float
    v_init: float
    v_designed: float
    reasoning: str = ""
    before_img: str | None = None
    after_img: str | None = None


class LanguageGame:
    def __init__(
        self,
        client,
        system: str = "lorenzsystem_gradient",
        env_name: str | None = None,
        verbose: bool = False,
        seed: int = 0,
        agents_root: str = _DEFAULT_AGENTS_ROOT,
        model: str = "gpt-4o",
        device: str = "cpu",
    ):
        self.client = client
        self.system = system
        self.env_name = env_name
        self.verbose = verbose
        self.seed = seed
        self.agents_root = agents_root
        self.llm_model = model
        self.device = device

        self._model_cache: dict[str, nn.Module] = {}
        self._translator_cache: dict[str, tuple[HumanToAgentTranslator, Agent2HumanTranslator]] = {}
        # Seeded RNG for the random baseline observation used in ΔV computation.
        # Advanced once per chat() call so successive prompts share a reproducible
        # but non-degenerate sequence of baselines.
        self._baseline_rng = torch.Generator().manual_seed(seed)

        if env_name is not None:
            policy = load_model(env_name, system, seed, agents_root, device)
            obs_dim = policy.pi_features_extractor.encoder.in_features
            self._model_cache[env_name] = (policy, obs_dim)
            self._translator_cache[env_name] = (
                HumanToAgentTranslator(client, env_name, model),
                Agent2HumanTranslator(client, env_name, model),
            )
            self.router = None
        else:
            available = set(list_envs(system, agents_root)) & set(TRAINED_ENVS)
            available_envs = sorted(available)
            if not available_envs:
                raise ValueError(
                    f"No trained environments found for system '{system}' in '{agents_root}'."
                )
            self.router = RouterAgent(client, available_envs, model)
            if verbose:
                print(f"  [Router] {len(available_envs)} environments available: {available_envs}")

    def log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def _get_model(self, env_name: str) -> tuple[nn.Module, int]:
        if env_name not in self._model_cache:
            policy = load_model(
                env_name, self.system, self.seed, self.agents_root, self.device,
            )
            obs_dim = policy.pi_features_extractor.encoder.in_features
            self._model_cache[env_name] = (policy, obs_dim)
        return self._model_cache[env_name]

    def _get_translators(self, env_name: str) -> tuple[HumanToAgentTranslator, Agent2HumanTranslator]:
        if env_name not in self._translator_cache:
            self._translator_cache[env_name] = (
                HumanToAgentTranslator(self.client, env_name, self.llm_model),
                Agent2HumanTranslator(self.client, env_name, self.llm_model),
            )
        return self._translator_cache[env_name]

    @torch.no_grad()
    def chat(self, prompt: str, output_dir: str | None = None) -> ChatResult:
        # Step 0: Route
        reasoning = ""
        if self.router is not None:
            choice = self.router.route(prompt)
            env_name = choice.env_name
            reasoning = choice.reasoning
            self.log(f"  [Router] → {env_name} ({reasoning})")
        else:
            env_name = self.env_name

        h2a, a2h = self._get_translators(env_name)
        policy, obs_dim = self._get_model(env_name)

        # Step 0.5: random baseline observation. The critic value at this
        # baseline serves as the reference point for ΔV.
        baseline_obs = torch.randn(
            obs_dim, generator=self._baseline_rng, dtype=torch.float32
        ).reshape(1, -1)
        v_init = policy.get_value(baseline_obs).item()

        # Step 1: LLM designs internal state
        goal = h2a.prompt2goal(prompt)
        self.log(f"  [Goal] {goal}")
        state_obj = h2a.goal2state(goal)
        state_list = state_obj.state
        self.log(f"  [Designed State] ({len(state_list)}d) {state_list}")

        before_img = None
        after_img = None

        if output_dir is not None:
            # Grounded path: set internal state in env → get real obs → render before
            # (after frame will be rendered once we have the action)
            from talk.render import render_state
            import numpy as np

            # First pass: get obs from env (render_state with action=None)
            _, _, obs_from_env = render_state(
                env_name, state_list, action=None, output_dir=output_dir,
            )
            obs_arr = obs_from_env
            if len(obs_arr) < obs_dim:
                obs_arr = np.concatenate([obs_arr, np.zeros(obs_dim - len(obs_arr))])
            elif len(obs_arr) > obs_dim:
                obs_arr = obs_arr[:obs_dim]
            state_tensor = torch.tensor(obs_arr, dtype=torch.float32).reshape(1, -1)
            self.log(f"  [Env Obs] ({len(obs_arr)}d)")
        else:
            # Lightweight path: use LLM-designed state directly (no gym env needed)
            if len(state_list) < obs_dim:
                state_list = state_list + [0.0] * (obs_dim - len(state_list))
            elif len(state_list) > obs_dim:
                state_list = state_list[:obs_dim]
            state_tensor = torch.tensor(state_list, dtype=torch.float32).reshape(1, -1)

        # Step 2: policy inference
        action = policy.get_action(state_tensor)
        v_designed_t = policy.get_value(state_tensor)
        action_list = action.squeeze().tolist()
        if not isinstance(action_list, list):
            action_list = [action_list]
        v_designed = v_designed_t.item()
        delta_v = v_designed - v_init
        self.log(f"  [Action] {action_list}")
        self.log(f"  [V_init] {v_init:.4f}  [V_designed] {v_designed:.4f}  [ΔV] {delta_v:+.4f}")

        if output_dir is not None:
            # Second pass: render both before and after with the actual action
            before_img, after_img, _ = render_state(
                env_name, state_list, action=action_list, output_dir=output_dir,
            )

        # Step 3: translate back to human language
        reply = a2h(state_list, action_list, delta_v)

        # Write markdown report
        if output_dir is not None:
            self._write_report(output_dir, prompt, env_name, reasoning, goal,
                               state_obj.state, action_list, delta_v, reply)

        return ChatResult(
            reply=reply,
            env_name=env_name,
            goal=goal,
            state=state_list,
            action=action_list,
            delta_v=delta_v,
            v_init=v_init,
            v_designed=v_designed,
            reasoning=reasoning,
            before_img=before_img,
            after_img=after_img,
        )

    def _write_report(self, output_dir, prompt, env_name, reasoning,
                      goal, state, action, delta_v, reply):
        lines = [
            f"# Talk to GRN: {self.system}",
            "",
            f"**Prompt:** {prompt}",
            "",
            f"**Environment:** {env_name}",
        ]
        if reasoning:
            lines.append(f"**Routing Reason:** {reasoning}")
        lines += [
            "",
            f"**Goal:** {goal}",
            "",
            f"**Designed State:** `{state}`",
            "",
            f"**Action:** `{action}`",
            "",
            f"**ΔV:** {delta_v:+.4f}",
            "",
            "## Before (designed state)",
            "![before](before.png)",
            "",
            "## After (one step)",
            "![after](after.png)",
            "",
            "## Reply",
            reply,
            "",
        ]
        path = os.path.join(output_dir, "report.md")
        with open(path, "w") as f:
            f.write("\n".join(lines))
