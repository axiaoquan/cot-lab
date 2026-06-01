"""Unified Gradio demo: enter a math question, see all 5 strategies solve it side-by-side.

Run:
    python -m app.gradio_app
"""
from __future__ import annotations

from pathlib import Path

import gradio as gr
import yaml

from common.models import load_model, generate
from common.prompts import DIRECT_TEMPLATE, ZERO_SHOT_COT_TEMPLATE, build_prompt
from common.extract_answer import extract_answer


CFG = yaml.safe_load((Path(__file__).parent.parent / "configs/default.yaml").read_text())
MODEL = None  # lazy


def _model():
    global MODEL
    if MODEL is None:
        MODEL = load_model(
            CFG["model"]["name"],
            backend="transformers",
            device=CFG["model"]["device"],
            torch_dtype=CFG["model"]["torch_dtype"],
        )
    return MODEL


def run_direct(q: str):
    out = generate(_model(), build_prompt(DIRECT_TEMPLATE, q),
                   max_new_tokens=64, temperature=0.0, do_sample=False)[0]
    return out, extract_answer(out)


def run_cot(q: str):
    out = generate(_model(), build_prompt(ZERO_SHOT_COT_TEMPLATE, q),
                   max_new_tokens=512, temperature=0.0, do_sample=False)[0]
    return out, extract_answer(out)


def run_self_consistency(q: str, n: int = 8):
    outs = generate(_model(), build_prompt(ZERO_SHOT_COT_TEMPLATE, q),
                    max_new_tokens=512, temperature=0.8, n=n)
    answers = [extract_answer(o) for o in outs]
    valid = [a for a in answers if a is not None]
    if not valid:
        return "\n---\n".join(outs), None
    from collections import Counter
    voted = Counter(valid).most_common(1)[0][0]
    return "\n---\n".join(outs), voted


def solve_all(q: str):
    rows = []
    direct_text, direct_ans = run_direct(q)
    rows.append(("Direct", direct_ans, direct_text))

    cot_text, cot_ans = run_cot(q)
    rows.append(("Zero-shot CoT", cot_ans, cot_text))

    sc_text, sc_ans = run_self_consistency(q, n=8)
    rows.append(("Self-Consistency (N=8)", sc_ans, sc_text))

    # ToT / BoN-verifier / GRPO LoRA: TODO — wire in once those modules ship results.

    md = "## Predictions\n\n| Method | Answer |\n|---|---|\n"
    for name, ans, _ in rows:
        md += f"| {name} | {ans} |\n"

    full_outputs = "\n\n---\n\n".join(f"### {n}\n```\n{t}\n```" for n, _, t in rows)
    return md, full_outputs


def main():
    with gr.Blocks(title="CoT-Lab Demo") as demo:
        gr.Markdown("# 🧠 CoT-Lab · Live Demo\n\nType a math problem and watch each strategy solve it.")
        q_in = gr.Textbox(label="Question", lines=3,
                          value="If 3 pens cost 12 dollars and 2 notebooks cost 14 dollars, how much do 5 pens and 3 notebooks cost?")
        btn = gr.Button("Solve")
        summary = gr.Markdown()
        details = gr.Markdown()
        btn.click(solve_all, inputs=q_in, outputs=[summary, details])

    demo.launch()


if __name__ == "__main__":
    main()
