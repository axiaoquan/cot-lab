# Troubleshooting

Living document of pitfalls encountered while developing on Apple Silicon.

## MPS / PyTorch

**`RuntimeError: MPS backend out of memory`**
Common when `num_generations` × `max_completion_length` is too large in Module 05.
Fix: drop `num_generations` to 4, `max_completion_length` to 512, or run swap.

**`bf16` warning at start**
Some ops (e.g., certain norms) silently fall back to fp32 on MPS. Usually fine.

**Generation produces only the prompt repeated**
Almost always a tokenizer chat-template mismatch. Confirm:
```python
print(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
```
should end with the assistant role marker.

## Ollama

**Connection refused on `localhost:11434`**
Start the server in another terminal: `ollama serve`.

**Slow first response**
Cold model load. Subsequent requests reuse the in-memory model.

## Datasets

**`OSError: gsm8k not found`**
The HuggingFace dataset `openai/gsm8k` may need authentication. Use:
```python
from datasets import load_dataset
ds = load_dataset("gsm8k", "main", split="test")
```

## trl + GRPO

**`use_vllm=True` errors on Mac**
Always set `use_vllm=False` in `GRPOConfig` on Apple Silicon.

**Reward stays at 0 forever**
Likely the reward function isn't extracting answers. Print a few `(completion, pred, gt)` triples
and check `extract_answer` regex hits. Common cause: the model never produces `<answer>...</answer>`
because the system prompt was stripped by the chat template.

**Loss is NaN**
Lower the learning rate (try 5e-7), increase `beta` to 0.1, or shorten `max_completion_length`.
