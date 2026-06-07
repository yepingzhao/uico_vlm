# VLM 实验设置（更新版，可直接执行）

**日期**: 2026-06-07
**关联文档**:
- [[2026-06-04-vlm-baseline-plan]] — 原始实验计划
- [[2026-06-06-vlm-baseline-currency-analysis]] — VLM baseline 时效性调研
- [[2026-06-07-vlm-experiment-supplement-strategy]] — 实验补充策略（完整分析）

---

## 1. 模型名单与配置

| Short Name | HF Model ID | 参数量 | 定位 | ZS | FS | LoRA |
|------------|-------------|--------|------|:--:|:--:|:---:|
| `llava` | `llava-hf/llava-1.5-7b-hf` | 7B | 经典 baseline | ✅ | ✅ | ✅ |
| `instructblip` | `Salesforce/instructblip-vicuna-7b` | 7B | 弱对照组 | ✅ | — | — |
| `qwen3vl` | `Qwen/Qwen3-VL-8B-Instruct` | 8B | 最新 Qwen 代 | ✅ | ✅ | ✅ |
| `internvl3` | `OpenGVLab/InternVL3-8B` | 8B | 最新 InternVL 代 | ✅ | — | — |

> 已从原计划移除: `qwen2vl`, `internvl2`。旧模型结果如已有，保留在 supplementary 中作为代际进步分析。

---

## 2. 推理设置

```
MAX_NEW_TOKENS  = 128
DO_SAMPLE       = False
RANDOM_SEED     = 42
TEST_IMAGES     = 3500 (全量) / 500 (prompt sensitivity) / 100 (dev mode)
```

### Qwen3-VL Processor 分辨率

| 场景 | min_pixels | max_pixels |
|------|-----------|------------|
| Zero-Shot | `256*28*28` | `1280*28*28` |
| Few-Shot | `128*28*28` | `256*28*28` |
| LoRA Training | `128*28*28` | `256*28*28` |

### InternVL3 Processor

```
CLIPImageProcessor(size=448, crop_size=448)  -- 复用 _internvl_base.py
```

---

## 3. Prompt 配置

| Prompt | 用途 | 评估 |
|--------|------|------|
| **A** | 主实验（所有模型） | ref-based + ref-free |
| **B** | Sensitivity (llava + qwen3vl only) | ref-free only |
| **C** | Sensitivity (llava + qwen3vl only) | ref-free only |

Prompt 内容见 `prompts/templates.py`，所有模型统一英文 Prompt A。

---

## 4. Few-Shot 设置

```
k             ∈ {1, 3}
examples       = 全测试集共用（预采样，固定 seed=42）
采样策略       = 从训练集随机采样 k 个 image-caption pairs
cache 路径     = outputs/fewshot_cache/fewshot_examples_k{k}_seed42.json
embed_images   = qwen3vl: True (inline PIL), llava: False (placeholder)
```

---

## 5. LoRA 训练设置

```
precision      = bfloat16 (compute dtype)
quantization   = 4-bit NF4 + double quantization
lora_r         = 16
lora_alpha     = 32
lora_dropout   = 0.05
target_modules = attention-only (q/k/v/o, 4 modules) -- 两个模型统一
batch_size     = 1 × 8 grad_accum
learning_rate  = 2e-4 (llava) / 1e-4 (qwen3vl, 保守)
warmup_ratio   = 0.1
epochs         = 2
max_grad_norm  = 1.0
vision_encoder = frozen
```

> **Qwen3-VL 特殊处理**: target_modules 仅 attention（4 modules），不碰 MLP 的 gate/up/down。DeepStack 在 frozen vision encoder 侧，不引入额外风险。bf16 解决 fp16 溢出问题（参 [[2026-06-05-lora-nan-analysis]]）。

---

## 6. 评估指标

| 类型 | 指标 | 工具 |
|------|------|------|
| Ref-based | BLEU-1/4, METEOR, ROUGE-L, CIDEr-D, SPICE, S_m | `eval/ref_based.py` |
| Ref-free | CLIPScore, RefCLIPScore | `eval/ref_free.py` |
| Human | Accuracy / Completeness / Normative Precision (3-way blind, 200 samples) | 人工 |

> Prompt B/C 仅用 ref-free 评估（格式差异会机械性压低 n-gram 指标）。

---

## 7. 执行命令

### Phase 0 — 补齐工程

```bash
# 1. 创建 InternVL3 wrapper（继承 InternVLBase）
#    文件: models/internvl3.py
#    在 models/__init__.py 注册 "internvl3"
#    在 config/__init__.py MODEL_REGISTRY 添加 internvl3 entry

# 2. 验证新模型下载
python download_models.py --model qwen3vl --dry-run

# 3. 快速验证两个新模型推理
python scripts/run_inference.py --models qwen3vl --subsample 5 --prompt A
python scripts/run_inference.py --models internvl3 --subsample 5 --prompt A
```

### Phase 1 — 速查

```bash
python scripts/run_inference.py --models llava --subsample 100 --prompt A
python scripts/run_eval.py --model llava --prompt A
```

### Phase 2 — Zero-Shot 全覆盖

```bash
python scripts/run_inference.py \
  --models instructblip llava qwen3vl internvl3 \
  --prompt A --subsample 3500

python scripts/run_eval.py --all --prompt A
```

### Phase 3 — Few-Shot

```bash
python scripts/run_fewshot.py --models llava qwen3vl --k 1 3 --subsample 3500
python scripts/eval_fewshot.py --all
```

### Phase 4 — Prompt Sensitivity

```bash
python scripts/run_inference.py --models llava qwen3vl --prompt B --subsample 500
python scripts/run_inference.py --models llava qwen3vl --prompt C --subsample 500

python scripts/run_eval.py --model llava --prompt B --ref_free_only
python scripts/run_eval.py --model llava --prompt C --ref_free_only
python scripts/run_eval.py --model qwen3vl --prompt B --ref_free_only
python scripts/run_eval.py --model qwen3vl --prompt C --ref_free_only
```

### Phase 5 — LoRA

```bash
# LLaVA（bf16 重跑）
python scripts/train_lora.py --model llava --epochs 2 --lr 2e-4

# Qwen3-VL（首次训练，保守 lr）
python scripts/train_lora.py --model qwen3vl --epochs 2 --lr 1e-4

# 推理
python scripts/inference_lora.py --model llava
python scripts/inference_lora.py --model qwen3vl

# 评估
python scripts/run_eval.py --model llava-lora --prompt A
python scripts/run_eval.py --model qwen3vl-lora --prompt A
```

### Phase 6 — 评估汇总

```bash
python scripts/run_eval.py --all
# 生成 outputs/zeroshot_all_metrics.json 和 outputs/fewshot_all_metrics.json
```

---

## 8. 输出结构

```
outputs/
  llava/
    predictions_prompt_a.jsonl
    predictions_prompt_b.jsonl, predictions_prompt_c.jsonl
    predictions_fewshot_k1.jsonl, predictions_fewshot_k3.jsonl
    metrics_prompt_a.json, metrics_prompt_b.json, metrics_prompt_c.json
    metrics_fewshot_k1.json, metrics_fewshot_k3.json
  instructblip/
    predictions_prompt_a.jsonl
    metrics_prompt_a.json
  qwen3vl/
    predictions_prompt_a.jsonl
    predictions_prompt_b.jsonl, predictions_prompt_c.jsonl
    predictions_fewshot_k1.jsonl, predictions_fewshot_k3.jsonl
    metrics_prompt_a.json, ...
  internvl3/
    predictions_prompt_a.jsonl
    metrics_prompt_a.json
  llava-lora/
    metrics_prompt_a.json
  qwen-3vl-lora/
    training.log
    metrics_prompt_a.json
  zeroshot_all_metrics.json
  fewshot_all_metrics.json
  fewshot_cache/
```

---

## 9. 结果表格草稿

**Table I: Comparison with VLM Baselines (Zero-Shot)**

```
Methods            | Type       | B@4 | M    | R    | C    | S    | S_m  | CLIP-S
InstructBLIP       | VLM (ZS)   |  —  |  —   |  —   |  —   |  —   |  —   |  —
LLaVA-1.5-7B       | VLM (ZS)   |  —  |  —   |  —   |  —   |  —   |  —   |  —
Qwen3-VL-8B        | VLM (ZS)   |  —  |  —   |  —   |  —   |  —   |  —   |  —
InternVL3-8B       | VLM (ZS)   |  —  |  —   |  —   |  —   |  —   |  —   |  —
--- (existing specialist rows above) ---
UIFormer           | Specialist | 32.9| 24.99| 46.6 | 126.6| 20.6 | 50.3 |  —
```

**Table II: VLM Scaling Behavior**

```
Model              | Setting    | B@4 | M    | R    | C    | S    | CLIP-S
LLaVA-1.5-7B       | ZS         |  —  |  —   |  —   |  —   |  —   |  —
                   | FS (k=1)   |  —  |  —   |  —   |  —   |  —   |  —
                   | FS (k=3)   |  —  |  —   |  —   |  —   |  —   |  —
                   | LoRA       |  —  |  —   |  —   |  —   |  —   |  —
Qwen3-VL-8B        | ZS         |  —  |  —   |  —   |  —   |  —   |  —
                   | FS (k=1)   |  —  |  —   |  —   |  —   |  —   |  —
                   | FS (k=3)   |  —  |  —   |  —   |  —   |  —   |  —
                   | LoRA       |  —  |  —   |  —   |  —   |  —   |  —
```

**Table III (Supplementary): Generational Improvement Gap**

```
Model              | Year | MMMU (general) | UICO S_m (domain) | Δ
LLaVA-1.5-7B       | 2024 | ~36            |  —                | —
Qwen2.5-VL-7B      | 2025 | 55.0           |  —                | —
Qwen3-VL-8B        | 2025 | ~60+           |  —                | Qwen3 − Qwen2.5
InternVL2-8B       | 2024 | ~50            |  —                | —
InternVL3-8B       | 2025 | 62.7           |  —                | InternVL3 − InternVL2
```

---

## 10. 实施检查清单

```
□ Phase 0: 补齐工程
  □ 创建 models/internvl3.py wrapper（继承 InternVLBase）
  □ 在 models/__init__.py 注册 "internvl3"
  □ 在 config/__init__.py MODEL_REGISTRY 添加 internvl3
  □ 确认 Qwen3-VL-8B 和 InternVL3-8B 模型已下载
  □ 验证两个新模型 ZS 推理（--subsample 5）

□ Phase 1: LLaVA ZS dev mode (--subsample 100)
  □ 推理 + 评估 → 确认 VLM 能力基线

□ Phase 2: ZS 全覆盖
  □ instructblip + llava + qwen3vl + internvl3 × Prompt A × 3500
  □ 评估 ref_based + ref_free

□ Phase 3: Few-Shot
  □ llava + qwen3vl × k=1,3 × 3500
  □ 评估

□ Phase 4: Prompt Sensitivity
  □ llava + qwen3vl × B/C × 500
  □ 评估 ref_free_only

□ Phase 5: LoRA
  □ llava-lora bf16 重跑（确保公平对比）
  □ qwen3vl-lora 首次训练
  □ 推理 + 评估

□ Phase 6: 汇总
  □ 生成 zeroshot_all_metrics.json, fewshot_all_metrics.json
  □ 人工评估（200 samples, 3 evaluators, 3 dimensions）
```
