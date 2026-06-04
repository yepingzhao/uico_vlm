# Prompt Migration: 全量实验重跑计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 因 Prompt A/B/C/Fewshot 全部重写，重跑 VLM baseline 实验计划中所有推理+评估

**Architecture:** 先修代码（移除 ZH、更新训练 prompt、加 Qwen2VL LoRA 配置），再按 Phase 2→6 顺序重跑推理+评估。Prompt B/C 仅用 ref-free 指标评估。

**Tech Stack:** Python, PyTorch, HuggingFace Transformers, PEFT/QLoRA, pycocoevalcap, CLIP

**关联文档:**
- 实验计划: `docs/research-notes/2026-06-04-vlm-baseline-plan.md`
- Prompt 分析: `docs/research-notes/2026-06-04-prompt-gt-alignment-analysis.md`

---

## Prompt 变更摘要

| Prompt | 旧 | 新 | 影响范围 |
|--------|----|----|---------|
| A | "Describe any urban incivility or civic norm violations..." | "In one sentence, describe any violation of urban order... State what/where" | ZS全量 + LoRA推理 |
| B | 3-part analysis (1)(2)(3) | Violation/Location 结构化格式 | Sensitivity (ref-free only) |
| C | Urban management inspector persona | Content ablation: 加 "why" | Sensitivity (ref-free only) |
| ZH | 中文 prompt | **已删除** | — |
| Fewshot | "describe any urban incivility..." | 对齐新 Prompt A | FS k=1,3 |

---

## 前置代码修改

### Task 1: 清理残留的 ZH prompt 引用

**Files:**
- Modify: `scripts/run_eval.py:116-120`
- Modify: `scripts/run_inference.py:14,17`
- Modify: `config/__init__.py:41-53` (stale PROMPTS dict)

- [ ] **Step 1: run_eval.py — 移除 ZH combo**

```python
# scripts/run_eval.py:110-120, 将:
        # Chinese prompt: Qwen only
        combos.append(("qwen2vl", "ZH"))
# 改为: 删除这两行

# 同时将 B/C 标记为 ref_free_only — 在 --all 中跳过 B/C 的 ref-based 指标。
# 策略：compute_all_metrics 增加 skip_ref_based 参数，--all 时对 B/C 传 skip_ref_based=True
```

实际改动：在 `run_eval.py` 的 `--all` 分支中：
1. 删除 ZH 行
2. B/C combo 改为 `compute_all_metrics(name, pk, args.device, skip_ref_based=True)`

- [ ] **Step 2: run_inference.py — 更新 docstring**

```python
# Line 14: 删除
#     python scripts/run_inference.py --models qwen2vl --prompt ZH

# Line 53: docstring 中 "ZH" 删除
```

- [ ] **Step 3: config/__init__.py — 同步或删除 stale PROMPTS dict**

`config/__init__.py` 中的 `PROMPTS` dict 没有任何脚本引用（所有脚本从 `prompts.templates` 导入）。两个选择：
- 删除（推荐：避免混淆）
- 同步为新 prompt 文本

删除 `PROMPTS` dict。保留 `SENSITIVITY_MODELS`。

- [ ] **Step 4: 验证 — 确认无 ZH 引用**

```bash
grep -rn "ZH\|prompt_zh\|PROMPT_ZH" --include="*.py" | grep -v __pycache__ | grep -v ".pyc"
```
预期：仅在已存在的 predictions/metrics 文件名中出现。

- [ ] **Step 5: Commit**

```bash
git add scripts/run_eval.py scripts/run_inference.py config/__init__.py
git commit -m "chore: remove ZH prompt references, mark B/C as ref-free-only in eval --all"
```

### Task 2: 更新训练 prompt 默认值

**Files:**
- Modify: `data/training_dataset.py:28`

- [ ] **Step 1: 更新 user_prompt 默认值**

```python
# data/training_dataset.py:28, 将:
        user_prompt: str = "Describe this urban scene in one sentence.",
# 改为:
        user_prompt: str = (
            "In one sentence, describe any violation of urban order visible in "
            "this image. State what the problem is and where it is located."
        ),
```

原因：训练时的 instruction prompt 应与 zero-shot 推理的 Prompt A 一致，否则 LoRA 模型学到的是错误的 prompt distribution。

- [ ] **Step 2: Commit**

```bash
git add data/training_dataset.py
git commit -m "fix: align training user_prompt with new Prompt A text"
```

### Task 3: 添加 Qwen2VL LoRA 训练配置

**Files:**
- Modify: `config/training.py:52-65`

- [ ] **Step 1: 添加 qwen2vl entry 到 MODEL_LORA_CONFIGS**

```python
# 在 config/training.py 的 MODEL_LORA_CONFIGS 中添加:
    "qwen2vl": {
        "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "model_class_name": "Qwen2_5_VLForConditionalGeneration",
        "processor_class_name": "AutoProcessor",
        "target_modules": ("q_proj", "k_proj", "v_proj", "o_proj"),
    },
```

注意：Qwen2.5-VL 的 attention 模块名是否与 LLaVA 一致需要验证。如果 Qwen2.5-VL 使用不同的 projection 命名（如 `attn.q_proj`），需要调整 `target_modules`。

- [ ] **Step 2: 验证 target_modules 命名**

```python
# 在 Python 中加载模型，打印 attention 层参数名:
from transformers import Qwen2_5_VLForConditionalGeneration
import torch
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype=torch.float16
)
# 检查一层 attention 的参数名
for name, _ in model.model.language_model.layers[0].self_attn.named_parameters():
    print(name)
```

预期输出类似 `q_proj.weight`, `k_proj.weight`, `v_proj.weight`, `o_proj.weight` — 与 LLaVA 一致。

- [ ] **Step 3: 确认 training dataset 兼容性**

`UICOInstructionDataset` 使用 `processor.apply_chat_template()` — Qwen2.5-VL 的 processor 支持此 API。需确认 `collate_fn` 中 `pixel_values` 的 shape 处理（Qwen2.5-VL 可能有不同的 image processing 输出格式）。

- [ ] **Step 4: Commit**

```bash
git add config/training.py
git commit -m "feat: add Qwen2.5-VL QLoRA training config"
```

---

## 实验重跑

### 约定

- 所有命令在 `conda activate uico_vlm` 环境下执行
- 确保 `vlm_eval` symlink 存在
- 输出文件会自动覆盖同路径的旧文件（checkpoint/resume 基于 image_id 去重，旧 prompt 生成的预测与新 prompt 不兼容，需手动删除旧文件或使用 `--no-resume` 机制）

**重要**: 现有的 `.jsonl` 预测文件包含旧 prompt 的输出。推理脚本的 checkpoint/resume 机制按 `image_id` 去重——如果旧文件存在，所有 image_id 都已被标记为已处理，新推理会「跳过」全部。**必须在重跑前删除旧预测文件，或脚本需要支持 `--no-resume` / `--overwrite` 标志。**

### Task 4: 推理脚本增加 overwrite 支持

**Files:**
- Modify: `scripts/run_inference.py`
- Modify: `scripts/run_fewshot.py`
- Modify: `scripts/inference_lora.py`

- [ ] **Step 1: 为 run_inference.py 添加 `--overwrite` 参数**

在 `argparse` 中添加 `--overwrite` flag。当设置时，在开始推理前删除已存在的预测文件。

```python
parser.add_argument("--overwrite", action="store_true",
                    help="Delete existing predictions file before starting.")
```

在 `run_inference()` 函数中，生成 `output_file` 路径后：
```python
if overwrite and os.path.exists(output_file):
    os.remove(output_file)
    print(f"[Overwrite] Removed existing {output_file}")
```

- [ ] **Step 2: 为 run_fewshot.py 添加相同的 `--overwrite`**

同样的改动。

- [ ] **Step 3: 检查 inference_lora.py 是否需要**

`inference_lora.py` 无需改动 — LoRA 是新增/重训，输出目录可能不存在或需要手动管理。

- [ ] **Step 4: Commit**

```bash
git add scripts/run_inference.py scripts/run_fewshot.py
git commit -m "feat: add --overwrite flag to inference scripts for prompt migration"
```

---

### Phase 2: Zero-Shot Prompt A (全量 3500)

**模型**: instructblip, llava, qwen2vl, internvl2

> 注意：实验计划中 BLIP-2 出现在 Table I 草稿但不在 4 模型名单中。如需要 BLIP-2 ZS 作为额外 baseline，需单独确认。

- [ ] **Step 1: 删除旧预测文件**

```bash
rm -f outputs/instructblip/predictions_prompt_a.jsonl
rm -f outputs/llava/predictions_prompt_a.jsonl
rm -f outputs/qwen2vl/predictions_prompt_a.jsonl
rm -f outputs/internvl2/predictions_prompt_a.jsonl
```

- [ ] **Step 2: 跑 instructblip ZS**

```bash
python scripts/run_inference.py --models instructblip --prompt A
```
预估时间：~2-3h（3500 images × Vicuna-7B）

- [ ] **Step 3: 跑 llava ZS**

```bash
python scripts/run_inference.py --models llava --prompt A
```
预估时间：~2-3h

- [ ] **Step 4: 跑 qwen2vl ZS**

```bash
python scripts/run_inference.py --models qwen2vl --prompt A
```
预估时间：~3-4h（Qwen2.5-VL 较慢）

- [ ] **Step 5: 跑 internvl2 ZS**

```bash
python scripts/run_inference.py --models internvl2 --prompt A
```
预估时间：~2-3h

- [ ] **Step 6: 验证行数**

```bash
wc -l outputs/instructblip/predictions_prompt_a.jsonl \
      outputs/llava/predictions_prompt_a.jsonl \
      outputs/qwen2vl/predictions_prompt_a.jsonl \
      outputs/internvl2/predictions_prompt_a.jsonl
```
预期：每个文件 3500 行。

---

### Phase 3: Few-Shot (k=1,3, 全量 3500)

**模型**: llava, qwen2vl

- [ ] **Step 1: 删除旧 fewshot 预测**

```bash
rm -f outputs/llava/predictions_fewshot_k1.jsonl \
      outputs/llava/predictions_fewshot_k3.jsonl \
      outputs/qwen2vl/predictions_fewshot_k1.jsonl \
      outputs/qwen2vl/predictions_fewshot_k3.jsonl
```

- [ ] **Step 2: 跑 llava + qwen2vl fewshot (k=1,3)**

```bash
python scripts/run_fewshot.py --models llava qwen2vl --k 1 3
```
预估时间：~6-8h（k=1 约 1.5x ZS 时间，k=3 约 2.5x ZS 时间）

- [ ] **Step 3: 验证**

```bash
wc -l outputs/llava/predictions_fewshot_k*.jsonl \
      outputs/qwen2vl/predictions_fewshot_k*.jsonl
```

---

### Phase 4: Prompt Sensitivity (Prompt B/C, subsample=500)

**模型**: llava, qwen2vl
**评估**: ref-free only (CLIPScore + RefCLIPScore)

- [ ] **Step 1: 删除旧 sensitivity 预测**

```bash
rm -f outputs/llava/predictions_prompt_b.jsonl \
      outputs/llava/predictions_prompt_c.jsonl \
      outputs/qwen2vl/predictions_prompt_b.jsonl \
      outputs/qwen2vl/predictions_prompt_c.jsonl
```

- [ ] **Step 2: 跑 Prompt B (llava + qwen2vl, subsample=500)**

```bash
python scripts/run_inference.py --models llava qwen2vl --prompt B --subsample 500
```

- [ ] **Step 3: 跑 Prompt C (llava + qwen2vl, subsample=500)**

```bash
python scripts/run_inference.py --models llava qwen2vl --prompt C --subsample 500
```

- [ ] **Step 4: 验证**

```bash
wc -l outputs/llava/predictions_prompt_b.jsonl \
      outputs/llava/predictions_prompt_c.jsonl \
      outputs/qwen2vl/predictions_prompt_b.jsonl \
      outputs/qwen2vl/predictions_prompt_c.jsonl
```
预期：每个 500 行。

- [ ] **Step 5: B 格式遵循率检查**

写一个简单脚本，检查 Prompt B 输出中是否同时包含 "Violation:" 和 "Location:" 字段：

```python
# scripts/check_format_compliance.py
import json, sys

def check(filepath):
    total = 0
    compliant = 0
    with open(filepath) as f:
        for line in f:
            rec = json.loads(line)
            caption = rec["caption"]
            total += 1
            if "Violation:" in caption and "Location:" in caption:
                compliant += 1
    rate = compliant / total * 100 if total > 0 else 0
    print(f"{filepath}: {compliant}/{total} ({rate:.1f}%) compliant")

for fp in sys.argv[1:]:
    check(fp)
```

```bash
python scripts/check_format_compliance.py \
    outputs/llava/predictions_prompt_b.jsonl \
    outputs/qwen2vl/predictions_prompt_b.jsonl
```

预期：格式遵循率 > 80%。将此数字写入论文 Appendix。

---

### Phase 5: LoRA

#### 5a: LLaVA-LoRA 重训

- [ ] **Step 1: 删除旧 LoRA weights + 预测**

```bash
rm -rf outputs/llava-lora/checkpoint-* outputs/llava-lora/adapter_* outputs/llava-lora/*.safetensors
rm -f outputs/llava-lora/predictions_prompt_a.jsonl
rm -rf outputs/llava-lora-merged/
```

- [ ] **Step 2: 训练**

```bash
python scripts/train_lora.py --model llava
```
预估时间：~4-6h（取决于训练集大小和 GPU）

- [ ] **Step 3: 推理**

```bash
python scripts/inference_lora.py --model llava
```

- [ ] **Step 4: 验证**

```bash
wc -l outputs/llava-lora/predictions_prompt_a.jsonl
```
预期：3500。

#### 5b: Qwen2VL-LoRA (新开发)

- [ ] **Step 1: 确认 Qwen2.5-VL tokenizer 兼容性**

`UICOInstructionDataset` 中 `labels[input_ids == 32000] = -100` 假设 image placeholder token id 为 32000（对应 LLaVA 的 processor）。Qwen2.5-VL 的 image token id 可能不同，需验证：

```python
from transformers import AutoProcessor
p = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
# 检查 conversation template 中 image placeholder 的 token id
```

如果不同，需要在 `training_dataset.py` 中将 32000 参数化，或为 Qwen2VL 单独处理。

- [ ] **Step 2: 验证 pixel_values shape 兼容性**

`collate_fn` 中 `torch.stack([item["pixel_values"]...)` 假设所有样本的 pixel_values shape 相同。Qwen2.5-VL 使用动态分辨率，可能产生不同 shape 的 pixel_values。需要验证或处理。

- [ ] **Step 3: 训练**

```bash
python scripts/train_lora.py --model qwen2vl
```

- [ ] **Step 4: 推理**

```bash
python scripts/inference_lora.py --model qwen2vl
```

- [ ] **Step 5: Commit LoRA 相关改动**

```bash
git add -A
git commit -m "feat: Qwen2.5-VL QLoRA training + inference support"
```

---

### Phase 6: 评估

#### 6a: Zero-Shot 评估

- [ ] **Step 1: Prompt A (ref-based + ref-free, 全量)**

```bash
# 逐个模型评估
python scripts/run_eval.py --model instructblip --prompt A
python scripts/run_eval.py --model llava --prompt A
python scripts/run_eval.py --model qwen2vl --prompt A
python scripts/run_eval.py --model internvl2 --prompt A
```

- [ ] **Step 2: Prompt B/C (ref-free only, llava + qwen2vl)**

```bash
python scripts/run_eval.py --model llava --prompt B --ref_free_only
python scripts/run_eval.py --model llava --prompt C --ref_free_only
python scripts/run_eval.py --model qwen2vl --prompt B --ref_free_only
python scripts/run_eval.py --model qwen2vl --prompt C --ref_free_only
```

- [ ] **Step 3: 生成 all_metrics.json**

```bash
python scripts/run_eval.py --all
```
注意：Task 1 应已修改 `--all` 分支，移除 ZH、B/C 用 ref-free only。

#### 6b: Few-Shot 评估

- [ ] **Step 1: 评估**

```bash
python scripts/eval_fewshot.py --all
```

或逐个：
```bash
python scripts/eval_fewshot.py --model llava --k 1 3
python scripts/eval_fewshot.py --model qwen2vl --k 1 3
```

#### 6c: LoRA 评估

- [ ] **Step 1: LLaVA-LoRA**

```bash
# 手动评估（inference_lora.py 的输出路径是 outputs/llava-lora/predictions_prompt_a.jsonl）
# 需要 run_eval.py 能读取此路径，或手动指定
# 检查 inference_lora.py 输出路径是否与 run_eval.py 期望路径一致
```

`run_eval.py` 期望路径：`outputs/{model_name}/predictions_prompt_{key}.jsonl`
`inference_lora.py` 输出路径：`outputs/{model_name}-lora/predictions_prompt_a.jsonl`

**路径不匹配！** 需要：
- 方案1：复制文件到期望路径
- 方案2：添加 `--predictions_dir` 参数到 `run_eval.py`

推荐方案1（简单，不改代码）：
```bash
mkdir -p outputs/llava-lora-eval
cp outputs/llava-lora/predictions_prompt_a.jsonl outputs/llava-lora-eval/predictions_prompt_a.jsonl
# run_eval.py 不识别 "llava-lora-eval" 作为 model name
```

实际上 `run_eval.py` 用 `model_name` 直接拼路径。`--model llava-lora` 会找 `outputs/llava-lora/predictions_prompt_a.jsonl` — 正好是 `inference_lora.py` 的输出路径。

- [ ] **Step 2: 评估 LLaVA-LoRA**

```bash
python scripts/run_eval.py --model llava-lora --prompt A
```

- [ ] **Step 3: 评估 Qwen2VL-LoRA**

```bash
python scripts/run_eval.py --model qwen2vl-lora --prompt A
```

---

### Phase 7: 论文表格更新

在 `make_table.py` 或手动更新 `docs/tcsvt2026.tex`：
- Table I: 填入新的 ZS 指标
- Table II: 填入 FS + LoRA scaling 指标
- Appendix: 填入 B/C sensitivity (ref-free only)

---

## 验证检查清单

```bash
# 1. 确认所有预测文件存在且行数正确
find outputs -name "predictions_prompt_a.jsonl" | while read f; do
    echo "$f: $(wc -l < "$f") lines"
done

# 2. 确认无 ZH 残留
find outputs -name "*prompt_zh*"

# 3. 确认 metrics 都已生成
find outputs -name "metrics_*.json" | sort

# 4. 确认 all_metrics.json 包含所有预期条目
python -c "
import json
with open('outputs/all_metrics.json') as f:
    data = json.load(f)
print('Entries:', len(data))
for k in sorted(data):
    print(f'  {k}')
"
```

---

## 风险

| 风险 | 可能性 | 缓解 |
|------|--------|------|
| Qwen2VL LoRA 训练 OOM | 中 | 已有 fewshot low-res processor 先例；降低 max_pixels 或 batch_size |
| Qwen2.5-VL tokenizer image token id ≠ 32000 | 高 | Task 5b Step 1 先验证再训练 |
| Qwen2.5-VL 动态分辨率导致 pixel_values shape 不一致 | 中 | collate_fn 可能需要改为 per-sample forward |
| 推理时间超预期 | 中 | 可分批跑，或使用 vLLM 后端加速 |
| 新 prompt 效果不如旧 prompt | 低 | GT alignment 分析表明新 prompt 更匹配；如果指标下降，回查典型输出 |
| InternVL2 对 Prompt C 空输出 | 中 | 旧 Prompt C (persona) 对 InternVL2 全空。新 Prompt C (content ablation) 更接近 Prompt A 格式，风险降低但不排除 |

---

## 预估总时间

| Phase | 内容 | 预估 GPU 时间 |
|-------|------|-------------|
| Phase 2 | 4 模型 ZS × 3500 | ~10-12h |
| Phase 3 | 2 模型 FS k=1,3 × 3500 | ~15-20h |
| Phase 4 | 2 模型 B/C × 500 | ~1-2h |
| Phase 5a | LLaVA LoRA 重训+推理 | ~5-7h |
| Phase 5b | Qwen2VL LoRA 开发+训练+推理 | ~6-8h + 开发 |
| Phase 6 | 全量评估 | ~3-4h |
| **合计** | | **~40-53h** |

可并行：Phase 2 的 4 个模型和 Phase 4 可以同时跑（不同 GPU）。
