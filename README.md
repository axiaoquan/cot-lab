# 🧠 CoT-Lab

```
   ____      _____   _          _
  / ___|___ |_   _| | |    __ _| |__
 | |   / _ \  | |   | |   / _` | '_ \
 | |__| (_) | | |   | |__| (_| | |_) |
  \____\___/  |_|   |_____\__,_|_.__/
                                  on Apple Silicon
```

> **Implement, evaluate, and compare the full Chain-of-Thought family — from prompting tricks to GRPO training — on a single Mac.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org)
[![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-MPS-purple.svg)](https://developer.apple.com/metal/pytorch/)
[![Tests](https://img.shields.io/badge/tests-pytest-success.svg)](#-tests)
[![Status: WIP](https://img.shields.io/badge/status-WIP-orange.svg)](#)

---

## 🎯 What this is

A **hands-on lab** that implements five canonical Chain-of-Thought reasoning strategies on the **same small model** (Qwen2.5-1.5B-Instruct), evaluates them on the **same dataset** (GSM8K), and produces a **head-to-head comparison**. The whole thing runs on a single Apple Silicon Mac (32GB+ recommended).

You walk away with hands-on understanding of the full CoT research line, from 2022 prompting papers to 2025 R1-style RL training.

---

## ⚡ 30-second Quickstart

```bash
git clone https://github.com/<you>/cot-lab && cd cot-lab
make setup        # creates .venv and installs requirements
make smoke        # downloads model and runs one example
make run-01       # first real experiment: Direct vs CoT
```

Or one-liner without Make:
```bash
pip install -r requirements.txt && python scripts/smoke_test.py
```

---

## 🗺 Concept Map

```
                          ┌─────────────────────────────────┐
                          │   Question:  3 pens + 2 books   │
                          └───────────────┬─────────────────┘
                                          │
        ┌─────────────────────────────────┼─────────────────────────────────┐
        ▼                                 ▼                                 ▼
   [Module 01]                       [Module 02]                       [Module 03]
   Direct vs CoT                  Self-Consistency                  BoN + Verifier
   ┌──────────┐                    ┌────────────┐                    ┌────────────┐
   │  prompt  │                    │ sample N   │                    │ sample N + │
   │  trick   │                    │ + vote     │                    │ score with │
   │          │                    │            │                    │ R1-7B      │
   └────┬─────┘                    └─────┬──────┘                    └─────┬──────┘
        │                                │                                 │
        ▼                                ▼                                 ▼
        ┌──────────────────────────────────────────────────────────────────┐
        │                        Master Comparison Table                   │
        │             (accuracy, latency, Mac-friendliness)                │
        └──────────────────────────────────────────────────────────────────┘
                                  ▲                ▲
                          [Module 04]          [Module 05]
                       Tree of Thoughts        GRPO LoRA
                         (24-game)            (R1-Zero mini)
                          ┌───────┐            ┌─────────┐
                          │ beam  │            │ rule-RM │
                          │ search│            │ + PPO   │
                          └───────┘            └─────────┘
```

| Module | Method | Paper | Status |
|---|---|---|---|
| [01](modules/01_zero_shot_cot/) | Zero-shot CoT (Direct vs CoT) | Wei 2022 / Kojima 2022 | 🟡 framework |
| [02](modules/02_self_consistency/) | Self-Consistency | Wang ICLR 2023 | 🟡 framework |
| [03](modules/03_bon_verifier/) | Best-of-N + Verifier | Lightman 2024 | 🟡 framework |
| [04](modules/04_tree_of_thoughts/) | Tree of Thoughts (24-game) | Yao NeurIPS 2023 | 🟡 framework |
| [05](modules/05_grpo_lora/) | GRPO LoRA (R1-Zero mini) | DeepSeek 2025 | 🟡 framework |

> 🟢 results filled  ·  🟡 framework only, results pending  ·  ⏳ not started

---

## 📊 Master Comparison Table

> 📌 _Numbers are placeholders. Each module fills its own row after experiments complete._

| Method | GSM8K Acc. | Avg Latency | Trainable? | Mac-friendly? |
|---|---|---|---|---|
| Direct (no CoT) | XX.X% | 1× | — | ✅ |
| Zero-shot CoT | XX.X% | 3× | — | ✅ |
| Self-Consistency (N=32) | XX.X% | 32× | — | ✅ |
| Best-of-N + Verifier (N=16) | XX.X% | 16× | RM only | ✅ |
| Tree of Thoughts (24-game) | XX.X% | ~100× | — | ✅ |
| **GRPO LoRA (ours)** | **XX.X%** | 5× | yes (1×GPU) | ✅ |
| DeepSeek-R1-Distill-7B (ref) | XX.X% | 8× | — | ✅ via ollama |

After running modules, regenerate:
```bash
make report       # writes results/master_table.png
```

---

## 🚀 Common workflows

```bash
make help              # show every available target
make smoke             # 1-shot env + model sanity check
make run-01            # Module 01: Direct vs CoT (200 examples)
make run-02            # Module 02: Self-Consistency (sweep N)
make run-04            # Module 04: Tree of Thoughts (24-game)
make demo              # launch Gradio UI on :7860
make report            # rebuild master comparison table
make test              # run unit tests
make clean             # remove caches and venv
```

---

## 🗂 Project structure

```
cot-lab/
├── README.md
├── Makefile
├── requirements.txt
├── configs/default.yaml
│
├── common/                       ← shared utilities
│   ├── models.py                 ← unified loader (transformers + ollama)
│   ├── datasets.py               ← GSM8K, 24-game
│   ├── prompts.py                ← prompt templates
│   ├── extract_answer.py         ← robust answer extraction (4-tier fallback)
│   └── eval.py                   ← accuracy + latency
│
├── modules/                      ← one folder per CoT method
│   ├── 01_zero_shot_cot/
│   ├── 02_self_consistency/
│   ├── 03_bon_verifier/
│   ├── 04_tree_of_thoughts/
│   └── 05_grpo_lora/             ← GRPO trainer + reward fns + emergence eval
│
├── app/gradio_app.py             ← unified interactive demo
├── scripts/                      ← smoke test + master plotter
├── tests/                        ← pytest unit tests
├── docs/                         ← paper notes, blog draft, troubleshooting
└── results/                      ← all runs dump here (JSON + PNG)
```

---

## 🍎 Why Apple Silicon?

Most CoT / GRPO repos target NVIDIA GPUs. This one is built around the realities of an Apple Silicon Mac:

- ✅ Runs on **MPS** (Metal Performance Shaders)
- ✅ **bfloat16** by default — no `bitsandbytes` (broken on MPS)
- ✅ No `vLLM` / `flash-attn` (unsupported on MPS) — plain `transformers`
- ✅ Module 05 GRPO uses small group sizes + grad accumulation to fit 48 GB
- ✅ Module 03 verifier can run via local **Ollama** (free + private)

If you have CUDA, override `device` in `configs/default.yaml`.

---

## 🧪 Tests

```bash
make test
# or
pytest tests/ -v
```

We unit-test the core utilities (answer extraction, dataset loaders, reward components).

---

## 📚 Reading list

Read the canonical paper before building each module. Notes in [`docs/PAPER_NOTES.md`](docs/PAPER_NOTES.md).

1. **CoT Prompting** — Wei et al., NeurIPS 2022
2. **Zero-shot CoT** — Kojima et al., NeurIPS 2022
3. **Self-Consistency** — Wang et al., ICLR 2023
4. **Process Reward Models** — Lightman et al., 2024
5. **Tree of Thoughts** — Yao et al., NeurIPS 2023
6. **DeepSeekMath (GRPO)** — Shao et al., 2024
7. **DeepSeek-R1** — DeepSeek, 2025 ⭐

---

## 🤝 Contributing

Personal learning project — issues / PRs welcome.

## 📄 License

[MIT](LICENSE)

---

> Maintainer: [@axiaoquan](https://github.com/axiaoquan) · Personal site: <https://axiaoquan.github.io/>
