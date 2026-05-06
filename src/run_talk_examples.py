"""Batch-run Talk-to-GRN communication examples across 4 GRN reservoirs.

Each of the four GRNs handles four distinct prompts, drawn so that the full
table exercises the same 16 tasks used in training but partitions them
across biologically heterogeneous reservoirs. The router (LLM) is GRN
independent, so for each prompt the environment selection is identical
across groups; any variation in action, ΔV, or natural-language reply is
attributable to the GRN reservoir.
"""

from __future__ import annotations

import json
import os

import dotenv
from openai import OpenAI

from talk import LanguageGame

GROUPS = [
    {
        "grn": "tyson1999circlelock_gradient",
        "label": "Tyson1999 circadian clock",
        "prompts": [
            "Stay balanced and don't fall over.",
            "Swing up and reach the top.",
            "Stabilize and hold your position.",
            "Get over the hill to the other side.",
        ],
    },
    {
        "grn": "gerard2010cellcycle_gradient",
        "label": "Gerard2010 cell cycle",
        "prompts": [
            "Run forward as fast as you can.",
            "Hop forward and keep your balance.",
            "Stand up from the ground.",
            "Spin it and keep it going.",
        ],
    },
    {
        "grn": "chickarmane2006stemcellswitch_gradient",
        "label": "Chickarmane2006 stem-cell switch",
        "prompts": [
            "Reach out and touch the target.",
            "Move it over to the right spot.",
            "Find the exit and get through.",
            "Swim forward through the water.",
        ],
    },
    {
        "grn": "kholodenko2000mapkcascade_gradient",
        "label": "Kholodenko2000 MAPK cascade",
        "prompts": [
            "Grab the money and get out.",
            "Climb higher and don't look down.",
            "Rescue the baby before it's too late.",
            "Kick the enemy and move on.",
        ],
    },
]


def main():
    dotenv.load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    output_root = "results/talk_examples"
    os.makedirs(output_root, exist_ok=True)

    results = []
    total = sum(len(g["prompts"]) for g in GROUPS)
    counter = 0

    for group in GROUPS:
        grn = group["grn"]
        label = group["label"]
        print(f"\n{'#'*70}\n# {label}  ({grn})\n{'#'*70}")

        game = LanguageGame(
            client,
            system=grn,
            env_name=None,  # Router mode
            verbose=True,
            seed=0,
            model="gpt-4o",
        )

        for prompt in group["prompts"]:
            counter += 1
            print(f"\n{'='*60}")
            print(f"[{counter}/{total}] GRN: {label}")
            print(f"         Prompt: {prompt}")
            print(f"{'='*60}")

            result = game.chat(prompt)

            row = {
                "group_index": GROUPS.index(group),
                "grn": grn,
                "grn_label": label,
                "prompt": prompt,
                "routed_env": result.env_name,
                "reasoning": result.reasoning,
                "goal": result.goal,
                "state": result.state,
                "action": result.action,
                "v_init": result.v_init,
                "v_designed": result.v_designed,
                "delta_v": result.delta_v,
                "reply": result.reply,
            }
            results.append(row)

            print(f"  Routed to: {result.env_name}")
            print(f"  Goal: {result.goal}")
            print(f"  ΔV: {result.delta_v:+.4f}  "
                  f"(V_init={result.v_init:+.4f} → V_designed={result.v_designed:+.4f})")
            print(f"  Reply: {result.reply}")

    json_path = os.path.join(output_root, "talk_examples.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {json_path}")

    print(f"\n{'='*80}\nSUMMARY\n{'='*80}")
    for r in results:
        print(f"  [{r['grn_label']:<40}] \"{r['prompt']}\" → {r['routed_env']}  ΔV={r['delta_v']:+.3f}")


if __name__ == "__main__":
    main()
