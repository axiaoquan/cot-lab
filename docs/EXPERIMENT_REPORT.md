# CoT-Lab 实验报告

> 在一台 Apple Silicon Mac 上，用同一个小模型（Qwen2.5-1.5B-Instruct）、同一个数据集（GSM8K），实现并对比了五种经典的思维链（Chain-of-Thought）推理策略——从 2022 年的提示词技巧，一路到 2025 年 R1 风格的强化学习训练。


**硬件**：Apple Silicon Mac（MPS 后端，bfloat16）
**策略模型**：`Qwen/Qwen2.5-1.5B-Instruct`
**裁判模型**（仅 Module 03）：`deepseek-r1:7b`（Ollama，Q4_K_M 量化）

---

## 1. 总览：五种策略横向对比

| 排名 | 方法 | GSM8K 准确率 | 实验设置 | 可训练 |
|---|---|---|---|---|
| 🥇 | **Self-Consistency (N=32)** | **0.780** | T=0.8, n=100 | 否 |
| 🥈 | Zero-shot CoT | 0.660 | 贪心, n=200 | 否 |
| 🥉 | GRPO LoRA (500 步) | 0.615 | ~6.5h, n=200 | 是 (LoRA 0.28%) |
| 4 | Best-of-N + Verifier (N=8) | 0.540 | R1-7B 裁判, n=50 | 仅奖励模型 |
| 5 | Direct (无 CoT) | 0.210 | 贪心, n=200 | 否 |
| — | Tree of Thoughts (24 点) | 0/5 solved | beam=5, depth=3 | 否 |

> 注：各实验的 `n`（题量）和采样温度不同，准确率之间不是严格同条件可比，但整体趋势清晰可靠。

---

## 2. 逐模块结果与分析

### Module 01 · Direct vs Zero-shot CoT

| 方法 | 准确率 | 平均延迟 (s) |
|---|---|---|
| Direct | 0.210 | 1.20 |
| Zero-shot CoT | 0.660 | 6.05 |

**结论**：仅在提示里加一句"let's think step by step"，准确率从 21% 跃升到 66%（3 倍），代价是 5 倍的推理延迟。这是整套实验里**性价比最高**的一招——零训练、零额外模型，只改提示词。复现了 Kojima et al. 2022 的核心发现。

### Module 02 · Self-Consistency

| 采样数 N | 准确率 |
|---|---|
| 1 | 0.590 |
| 4 | 0.640 |
| 8 | 0.720 |
| 16 | 0.750 |
| 32 | 0.780 |

**结论**：标准的"单调上升 + 边际递减"曲线，完美复现 Wang et al. 2023。

- **前 8 次采样贡献了 19 个点提升里的 13 个**——甜区在 N=1→8。
- N=16→32 算力翻倍只换 3 个点——性价比拐点在 N=8~16。
- 多数投票把单次采样的随机错误"洗"掉，且**不依赖任何外部裁判**，这是它最稳的原因。

### Module 03 · Best-of-N + Verifier ⚠️

| 设置 | 准确率 |
|---|---|
| n=10, BoN=4（小跑） | 0.700 |
| **n=50, BoN=8（正式）** | **0.540** |

**反常结果**：BoN=8 从 8 个候选里挑"最优"，理论上应 ≥ 单次 CoT（0.66），但实测只有 0.54，**反而更低**。

**根因分析**：
- 裁判用的是 R1（reasoning 模型），打分前会输出一大段 `<think>...</think>`。
- `max_new_tokens=64` 把生成截断在模型输出 `Score: X` **之前**。
- `parse_score` 的兜底逻辑是"抓文本里最后一个数字"，于是抓到的是 R1 思考过程里的**无关数字**（算式、中间结果），而非真正的评分。
- 打分一乱，"选最优"退化成近乎随机选择，于是把正确候选筛掉了。

**洞察**：**verifier 的质量是 BoN 的天花板**。一个不稳定的裁判会让 Best-of-N 掉到比单次推理还差。这与 2024 年以来关于 process/outcome reward model 可靠性的讨论一致。

**修复方向**（已诊断，未实施）：增大 `max_new_tokens`、收紧 `parse_score`（只认 `Score:` 后的数字，抓不到则丢弃该候选而非兜底 5.0）。

### Module 04 · Tree of Thoughts（24 点游戏）⚠️

**结果**：Solved 0/5，所有题目 trace 全空。

**根因分析**（两个真实 bug）：
1. **早停检查位置错误**：`is_solved` 只在 beam 截断后的 top-5 里检查。若正确终局状态因评分偏低没进 top-5，就永远检测不到——哪怕它已经被算出来了。
2. **评分器几乎不产生有效信号**：`llm_score` 要求 1.5B 模型只输出 `sure/likely/impossible`，但带 chat template 的小模型几乎不会严格遵守，三个关键词都不命中 → 全部落到默认值 0.3 → beam search 因为所有候选同分而退化成随机砍枝。
3. **浮点容差过严**：除法产生循环小数，累积误差使本应等于 24.0 的结果差出 `1e-6` 以上，判不出 solved。

**洞察**：ToT 的效果**高度依赖评估器质量**。用小模型当状态评估器时，整个搜索框架很容易"空转"。这本身是一个有价值的负面结论——ToT 不是免费午餐。

**修复方向**（已诊断，未实施）：在生成候选时就检查 is_solved（而非截断后）、放宽浮点容差到 `1e-3` 并对接近整数的值取整、给评分解析加兜底。

### Module 05 · GRPO LoRA（R1-Zero mini）

**训练**：500 步，耗时 **6 小时 32 分**（约 46s/step）。

| 训练指标 | 值 | 含义 |
|---|---|---|
| trainable% | 0.28% (435 万 / 15.5 亿) | LoRA 正确挂载 |
| reward mean / std | 0.444 / 0.463 | 奖励函数有非零信号，组内有差异 |
| KL | 0.0012 | 策略稳定，没跑飞 |
| clipped_ratio | **0.975** | 97.5% 生成被 512 token 截断 ⚠️ |

**评估**（n=200）：

| 指标 | 值 |
|---|---|
| 准确率 | 0.615 |
| 反思率 (reflection rate) | 0.030 |
| 平均回答长度 | 179.6 tokens |

**结果**：训练后准确率（0.615）**低于**基座 CoT（0.66），反思率几乎为 0，**没有观察到 R1-Zero 式的"自我修正"涌现行为**。

**根因分析**：
1. **步数太少**：500 步对 GRPO 远远不够，R1-Zero 级别的涌现通常需要数千步。
2. **截断污染**：97.5% 的训练样本被 512 token 截断，模型学到了大量"半截答案"，倾向于输出更短、更草率的回答（评估时平均仅 179.6 token）。
3. **规模不足**：1.5B 模型 + 仅 0.28% 的 LoRA 可训练参数，容量不足以涌现复杂的自我修正能力。

**洞察**：**小规模 RL 不仅可能无收益，甚至会略微损害性能**。强化学习的"推理涌现"对模型规模、训练步数、生成长度都有硬性要求。

---

## 3. 核心洞察

这套实验最有价值的，不是某个漂亮的准确率数字，而是三个**反直觉**的结论：

1. **🏆 最朴素的方法赢了**。Self-consistency 不靠裁判、不靠训练，纯采样 + 投票，反而拿到最高的 0.78。简单且鲁棒。

2. **⚠️ "更高级"不等于"更好"**。Best-of-N 用强模型当裁判，本应优于单次推理，却因裁判打分解析脆弱而掉到 0.54——**低于单次 CoT**。劣质裁判是负资产。

3. **⚠️ 推理增强不是免费午餐**。GRPO 训练了 6.5 小时，准确率不升反降。RL 涌现对规模和算力有硬门槛，小作坊配置容易"赔了夫人又折兵"。

> 一句话：**在小模型 + 单机的现实约束下，"花算力多采样投票"比"花算力请裁判 / 做 RL"更划算、更可靠。**

---

## 4. 复现指南 & 踩坑记录

### 环境

- **不要用 `make setup`**：它内部用 shell 默认的 `python`（本机是 conda base 3.13），装 torch/trl 易失败。改用独立的 **Python 3.11** 建 venv：
  ```bash
  /path/to/python3.11 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  ```
- **数据集 bug**（已修）：`common/datasets.py` 中 `load_dataset("gsm8k", ...)` 是旧写法，新版 `datasets` 库报 `HfUriError`。已改为 `"openai/gsm8k"`。
- **HF 下载慢**：`export HF_ENDPOINT=https://hf-mirror.com` 切镜像。

### Ollama（仅 Module 03 需要）

- **关键坑**：`brew install ollama` 装出的包（0.30.7）**缺 `llama-server` 二进制**，`/api/tags` 正常但 `/api/generate` 报 500。
- **解决**：卸载 brew 版，改装[官方 .app 版](https://ollama.com/download/mac)，自带完整二进制并自动起服务。
- **验证推理**用 `ollama run deepseek-r1:7b "say hi"`（比 `curl /api/tags` 更可靠——tags 通不代表推理通）。

### 运行顺序

```bash
# 轻量，几分钟到十几分钟
python -m modules.01_zero_shot_cot.run --n 200
python -m modules.02_self_consistency.run --n 100 --samples 1,4,8,16,32
python -m modules.04_tree_of_thoughts.run --n 5 --beam 5 --depth 3

# 需 Ollama 起着
python -m modules.03_bon_verifier.run --n 50 --bon 8

# 重头戏，约 6.5h
python -m modules.05_grpo_lora.prepare_data --n 500
python -m modules.05_grpo_lora.train --steps 500
python -m modules.05_grpo_lora.eval --ckpt checkpoints/grpo --n 200
```
