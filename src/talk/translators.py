"""LLM translators for the Talk to GRN pipeline.

Pipeline:
  Human prompt
    -> Prompt2Goal: infer goal action
    -> Goal2State: design environment state
    -> (policy inference happens in game.py)
    -> Agent2HumanTranslator: translate (state, action, value) back to language
"""

from __future__ import annotations

import os
from typing import List

from pydantic import BaseModel


# ─────────────────────────────────────────────────────────────────────────────
# Environment descriptions (loaded from doc/games/*.md)
# ─────────────────────────────────────────────────────────────────────────────

_GAMES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "doc", "games")


def load_env_desc(folder: str | None = None) -> dict:
    folder = folder or _GAMES_DIR
    env_desc = {}
    for file in os.listdir(folder):
        if file.endswith(".md") and file != "README.md":
            with open(os.path.join(folder, file), "r") as f:
                env_desc[file.replace(".md", "")] = f.read()
    return env_desc


env_desc = load_env_desc()


# ─────────────────────────────────────────────────────────────────────────────
# LLM prompt templates
# ─────────────────────────────────────────────────────────────────────────────


class InferGoalSystemPrompt:
    template = """You are translate human natural language into a goal action for a rational agent in the {env_name} RL environment.

# Environment Description:
{env_desc}

# Requirement and Instruction

You will translate human natural language into a formal goal action for a rational agent in the {env_name} RL environment.

For example, in the CartPole-v1 environment, "Keep stable" can be translated into "Keep the pole upright".\
And "Move fast" can be translated into "Make the cart continuously move to left or right".\

# Format

Directly output the translated goal state or action, without any other text.
"""
    env_desc = env_desc

    def get(self, env_name: str, desc: str = None) -> str:
        if desc is None:
            desc = self.env_desc[env_name]
        return self.template.format(env_desc=desc, env_name=env_name)


class State(BaseModel):
    state: List[float]


class DesignEnvSystemPrompt:
    template = """Given {env_name} RL environment and a goal action, you design a environment state that lead a rational agent to take the goal action.
# Environment Description:
{env_desc}

# Requirement and Instruction

You will design a environment state that lead a rational agent to take the goal action.

For example, in the CartPole-v1 environment, if the goal action is "Move to left", you can design the environment state as:
(assume the vector represents [cart_position, cart_velocity, pole_angle, pole_velocity])
{{
    "state": [0.1, 0.0, -0.2, 0.0],
}}

# Format

Directly output the environment state in JSON format, with the key "state" and the value is a list of numbers.
"""
    env_desc = env_desc

    def get(self, env_name: str, desc: str = None) -> str:
        if desc is None:
            desc = self.env_desc[env_name]
        return self.template.format(env_desc=desc, env_name=env_name)


class ReplySystemPrompt:
    template = """You are an agent in the {env_name} RL environment. The user \
will provide you the current state of the environment, your action, and a value-change \
signal Delta V (the PPO critic's value at the current state minus its value at a \
random baseline state). You will translate your action into natural language.

# Environment Description:
{env_desc}

# Instruction

Given the current state of the environment and the action, first infer the \
short-term goal of the action. Then use Delta V as an emotional signal: a clearly \
positive Delta V means the current state looks more promising than the baseline \
(confident, hopeful tone); a clearly negative Delta V means it looks worse than the \
baseline (uncertain, stressed tone); a near-zero Delta V means roughly indifferent \
(neutral, cautious tone). Finally, draft a reply to the user.

# Example (with CartPole-v1)

When the cart's pole is tilted to the left, the action is "Move to left", and Delta V \
is clearly negative, you can reply: "I'm not feeling good, and I'm trying to make the \
pole upright."

Note: Keep your reply concise and short in plain text. Do not directly talk about the numbers.

# Format

Directly output your reply in natural language, without any other text.
"""
    env_desc = env_desc

    def get(self, env_name: str, desc: str = None) -> str:
        if desc is None:
            desc = self.env_desc[env_name]
        return self.template.format(env_desc=desc, env_name=env_name)


# ─────────────────────────────────────────────────────────────────────────────
# LLM pipeline classes
# ─────────────────────────────────────────────────────────────────────────────


class Prompt2Goal:
    def __init__(self, client, env_name: str, model: str = "gpt-4o", env_desc: str = None):
        self.env_name = env_name
        self.client = client
        self.model = model
        self.env_desc = env_desc
        self.system_prompt = InferGoalSystemPrompt().get(env_name)

    def translate(self, prompt: str) -> str:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        return completion.choices[0].message.content

    def __call__(self, prompt: str) -> str:
        return self.translate(prompt)


class Goal2State:
    def __init__(self, client, env_name: str, model: str = "gpt-4o", env_desc: str = None):
        self.env_name = env_name
        self.client = client
        self.model = model
        self.env_desc = env_desc
        self.system_prompt = DesignEnvSystemPrompt().get(env_name)

    def translate(self, goal: str) -> State:
        completion = self.client.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": goal},
            ],
            response_format=State,
        )
        return completion.choices[0].message.parsed

    def __call__(self, goal: str) -> State:
        return self.translate(goal)


class HumanToAgentTranslator:
    def __init__(self, client, env_name: str, model: str = "gpt-4o", env_desc: str = None):
        self.env_name = env_name
        self.model = model
        self.env_desc = env_desc
        self.prompt2goal = Prompt2Goal(client, env_name, model, env_desc)
        self.goal2state = Goal2State(client, env_name, model, env_desc)

    def __call__(self, prompt: str, verbose: bool = False) -> List[float]:
        goal = self.prompt2goal(prompt)
        if verbose:
            print(f"  [Goal] {goal}")
        state_obj = self.goal2state(goal)
        if verbose:
            print(f"  [State] {state_obj.state}")
        return state_obj.state


class Agent2HumanTranslator:
    def __init__(self, client, env_name: str, model: str = "gpt-4o", env_desc: str = None):
        self.env_name = env_name
        self.client = client
        self.model = model
        self.env_desc = env_desc
        self.system_prompt = ReplySystemPrompt().get(env_name)

    def translate(self, state: list, action: list, delta_v: float) -> str:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": f"State: {state}, Action: {action}, Delta V: {delta_v:+.4f}",
                },
            ],
        )
        return completion.choices[0].message.content

    def __call__(self, state: list, action: list, delta_v: float) -> str:
        return self.translate(state, action, delta_v)
