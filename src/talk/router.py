"""Router Agent: selects the best RL environment for a given user prompt.

Uses an LLM to match the user's conversational intent against short
descriptions of the 16 trained environments, returning a structured choice.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, create_model


# ─────────────────────────────────────────────────────────────────────────────
# Canonical list of trained environments
# ─────────────────────────────────────────────────────────────────────────────

TRAINED_ENVS = [
    "Acrobot-v1",
    "BankHeist-ram",
    "CartPole-v1",
    "CrazyClimber-ram",
    "finger-spin",
    "HalfCheetah-v4",
    "Hopper-v4",
    "HumanoidStandup-v4",
    "Kangaroo-ram",
    "KungFuMaster-ram",
    "MountainCarContinuous-v0",
    "Pendulum-v1",
    "PointMaze",
    "Pusher-v4",
    "Reacher-v4",
    "Swimmer-v4",
]

# ─────────────────────────────────────────────────────────────────────────────
# Short environment descriptions for the Router prompt
# ─────────────────────────────────────────────────────────────────────────────

ENV_SHORT_DESCRIPTIONS: dict[str, str] = {
    "Acrobot-v1": (
        "A two-link pendulum hangs downward from a fixed point. "
        "Apply torque at the middle joint to swing the free end above a target height. "
        "Requires energy pumping and momentum management."
    ),
    "BankHeist-ram": (
        "Navigate a maze of underground vaults as a robber, collecting money "
        "while avoiding police. Plant dynamite to create escape routes. "
        "Involves spatial navigation, risk-reward trade-offs, and escape planning."
    ),
    "CartPole-v1": (
        "Balance an upright pole on a cart by pushing the cart left or right "
        "along a frictionless track. A classic stabilization and balance task."
    ),
    "CrazyClimber-ram": (
        "Climb the outside of a tall building, dodging falling objects and "
        "hostile residents dropping hazards. Requires upward spatial movement, "
        "obstacle avoidance, and persistence."
    ),
    "finger-spin": (
        "A planar finger with two joints must continuously spin a free body "
        "attached at its tip. Requires precise rotational coordination and "
        "sustained periodic motion."
    ),
    "HalfCheetah-v4": (
        "A 2D cheetah robot with six joints must run forward as fast as possible "
        "by coordinating leg torques. A high-DOF locomotion task emphasizing "
        "speed and rhythmic gait."
    ),
    "Hopper-v4": (
        "A one-legged robot must hop forward as fast as possible while "
        "maintaining upright balance. Combines locomotion with dynamic "
        "stability on a single leg."
    ),
    "HumanoidStandup-v4": (
        "A 3D humanoid with 17 joints must stand up from a prone position "
        "on the ground. Requires whole-body coordination, strength, "
        "and sequential posture planning."
    ),
    "Kangaroo-ram": (
        "Control a mother kangaroo to rescue baby kangaroos by climbing "
        "platforms, punching enemies, and ringing bells at the top. "
        "Involves vertical platforming, combat, and rescue objectives."
    ),
    "KungFuMaster-ram": (
        "Fight through waves of enemies using punches and kicks while "
        "advancing through a martial arts temple. Emphasizes combat, "
        "timing, and forward progression."
    ),
    "MountainCarContinuous-v0": (
        "A car stuck in a valley must build momentum by rocking back and "
        "forth to reach the hilltop goal. Requires energy accumulation "
        "and delayed gratification."
    ),
    "Pendulum-v1": (
        "Apply continuous torque to swing an inverted pendulum from a random "
        "starting angle up to the vertical upright position and keep it balanced. "
        "A classic control and stabilization task."
    ),
    "PointMaze": (
        "A point mass navigates through a U-shaped maze to reach a goal "
        "position by applying directional forces. Requires spatial navigation "
        "and path planning."
    ),
    "Pusher-v4": (
        "A 7-DOF robotic arm must push a cylinder across a table to a target "
        "location. Involves multi-joint manipulation, contact physics, "
        "and precise object interaction."
    ),
    "Reacher-v4": (
        "A 2-link planar robot arm must move its fingertip to reach a randomly "
        "placed target. A target-reaching task requiring spatial accuracy "
        "and joint coordination."
    ),
    "Swimmer-v4": (
        "A 3-link swimming robot undulates its body joints to move forward "
        "through a 2D viscous fluid. Requires rhythmic, wave-like coordination."
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Structured output model
# ─────────────────────────────────────────────────────────────────────────────


class RouterChoice:
    """Routing result returned to callers (plain strings, no enum)."""

    __slots__ = ("reasoning", "env_name")

    def __init__(self, reasoning: str, env_name: str):
        self.reasoning = reasoning
        self.env_name = env_name


def _build_response_format(env_names: list[str]) -> type[BaseModel]:
    """Build a Pydantic model with env_name constrained to an Enum of valid names.

    OpenAI structured output enforces the enum constraint during generation,
    so the model can only output one of the valid environment names.
    """
    members = {name.replace("-", "_").replace(".", "_"): name for name in env_names}
    env_enum = Enum("EnvName", members, type=str)
    return create_model(
        "_RouterResponse",
        reasoning=(str, ...),
        env_name=(env_enum, ...),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Router Agent
# ─────────────────────────────────────────────────────────────────────────────


class RouterAgent:
    """Select the most appropriate RL environment for a user prompt."""

    def __init__(self, client, env_names: list[str] | None = None, model: str = "gpt-4o"):
        self.client = client
        self.model = model
        self.env_names = env_names or TRAINED_ENVS
        self._response_format = _build_response_format(self.env_names)

        env_list = "\n".join(
            f"- {name}: {ENV_SHORT_DESCRIPTIONS[name]}"
            for name in self.env_names
        )
        self.system_prompt = (
            "You select the most appropriate RL environment to serve as the "
            "metaphorical frame for interpreting a user's conversational message.\n\n"
            "Each environment defines a distinct vocabulary of states and actions. "
            "The chosen environment provides the context within which the user's "
            "intent is translated into the agent's behavior.\n\n"
            "Available environments:\n"
            f"{env_list}\n\n"
            "Given the user's message, choose the environment whose dynamics, goals, "
            "or metaphorical structure best match the user's intent, emotional state, "
            "or described situation. Return the exact env_name from the list above.\n\n"
            "Think step by step: first identify the core concept in the user's message "
            "(e.g., balance, speed, fighting, navigation, reaching), then match it "
            "to the most fitting environment."
        )

    def route(self, prompt: str) -> RouterChoice:
        """Select the best environment and return the full choice with reasoning."""
        completion = self.client.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            response_format=self._response_format,
        )
        parsed = completion.choices[0].message.parsed
        return RouterChoice(
            reasoning=parsed.reasoning,
            env_name=parsed.env_name.value,
        )

    def __call__(self, prompt: str) -> str:
        """Select the best environment and return just the env_name."""
        return self.route(prompt).env_name
