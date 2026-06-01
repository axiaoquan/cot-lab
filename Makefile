# CoT-Lab Makefile · all common entry points
# Usage: make <target>   |   make help

PY      := python
VENV    := .venv
ACT     := source $(VENV)/bin/activate &&

# Default config
CFG     := configs/default.yaml

.PHONY: help setup smoke test \
        run-01 run-02 run-03 run-04 \
        train-05 prepare-05 eval-05 \
        demo report clean

help:  ## Show this help
	@echo ""
	@echo "CoT-Lab targets:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'
	@echo ""

setup:  ## Create venv and install requirements
	$(PY) -m venv $(VENV)
	$(ACT) pip install --upgrade pip
	$(ACT) pip install -r requirements.txt
	@echo "✅ Setup complete. Activate with: source $(VENV)/bin/activate"

smoke:  ## Quick env + model sanity check
	$(ACT) $(PY) scripts/smoke_test.py

test:  ## Run unit tests
	$(ACT) pytest tests/ -v

run-01:  ## Module 01: Direct vs Zero-shot CoT (200 examples)
	$(ACT) $(PY) -m modules.01_zero_shot_cot.run --n 200

run-02:  ## Module 02: Self-Consistency (sweep N)
	$(ACT) $(PY) -m modules.02_self_consistency.run --n 100 --samples 1,4,8,16,32

run-03:  ## Module 03: BoN + R1 verifier (needs ollama)
	$(ACT) $(PY) -m modules.03_bon_verifier.run --n 50 --bon 8

run-04:  ## Module 04: Tree of Thoughts on 24-game
	$(ACT) $(PY) -m modules.04_tree_of_thoughts.run --n 5 --beam 5 --depth 3

prepare-05:  ## Build SFT data for GRPO
	$(ACT) $(PY) -m modules.05_grpo_lora.prepare_data --n 500

train-05:  ## Train GRPO LoRA (long; ~8-12h on M5 Pro for 500 steps)
	$(ACT) $(PY) -m modules.05_grpo_lora.train --steps 500

eval-05:  ## Evaluate trained GRPO LoRA + emergence stats
	$(ACT) $(PY) -m modules.05_grpo_lora.eval --ckpt checkpoints/grpo --n 200

demo:  ## Launch Gradio interactive demo
	$(ACT) $(PY) -m app.gradio_app

report:  ## Aggregate all module results into the master table
	$(ACT) $(PY) scripts/plot_master_table.py

clean:  ## Remove caches and venv
	rm -rf $(VENV) **/__pycache__ **/*.pyc .pytest_cache
	@echo "🧹 Clean."
