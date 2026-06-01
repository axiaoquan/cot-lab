# Module 03 · Best-of-N + Verifier

> **Question**: Instead of voting, what if we _score_ each candidate with a verifier (a stronger model) and pick the best?

## Paper

- Lightman et al., _Let's Verify Step by Step_, OpenAI 2023.
- Cobbe et al., _Training Verifiers to Solve Math Word Problems_ (GSM8K), 2021.

## How to run

```bash
# Make sure ollama is up and has deepseek-r1:7b pulled:
ollama pull deepseek-r1:7b

python -m modules.03_bon_verifier.run --n 100 --bon 16
```

## Verifier choices

| Verifier | Setup |
|---|---|
| Ollama (recommended) | `ollama pull deepseek-r1:7b` |
| HF model | uses `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` (slower on Mac) |
| Trained PRM | TODO: train a small PRM on GSM8K solutions |

## Expected pattern

BoN with a strong verifier should match or beat majority voting at the same N,
**especially when N is small** (8-16). At very large N, voting often catches up.
