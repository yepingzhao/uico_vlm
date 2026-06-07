# InternVL QLoRA 修复记录与 IMG_CONTEXT 架构稳定性分析

**日期**: 2026-06-07
**触发**: fix-internvl session — 修复 InternVL3/3.5 QLoRA 训练
**关联文档**:
- [[2026-06-07-vlm-experiment-supplement-strategy]] — 实验补充策略（InternVL3 升级决策）
- [[2026-06-05-lora-nan-analysis]] — bf16 NaN 修复
- [[2026-06-04-prompt-gt-alignment-analysis]] — prompt 设计

---

## 1. 背景

前次会话中 InternVL 家族的 QLoRA 训练全部失败：
- **InternVL2-8B** (InternLM2 backbone): baseline rep_ratio=0.37，训练崩溃
- **InternVL3-8B** (Qwen2 backbone): lr=1e-4 在 step 100 崩溃到 rep=0.398
- **InternVL3.5-8B** (Qwen3 backbone): processor `start_image_token` 不兼容，无法启动

本次会话目标是修复 InternVL3 和 InternVL3.5 的训练。

---

## 2. InternVL3.5 修复过程

### 2.1 修复清单

| # | 问题 | 根因 | 修复 |
|---|------|------|------|
| 1 | `AttributeError: 'TokenizersBackend' has no attribute 'start_image_token'` | processor_config.json 指定 InternVLProcessor，但 tokenizer 是 Rust TokenizersBackend，缺少 image token 属性 | `train_lora.py`: internvl35 使用 `AutoTokenizer` 直接加载，绕过 InternVLProcessor |
| 2 | `ImportError: cannot import name 'Qwen3Config'` | transformers 4.49.0 不包含 Qwen3 模型 | 升级 transformers 4.49.0 → 4.51.3 |
| 3 | `libnvJitLink.so.13: cannot open shared object file` | cu13 CUDA 库路径不在 LD_LIBRARY_PATH | 训练时设置 `LD_LIBRARY_PATH` 包含 `nvidia/cu13/lib` |
| 4 | Monkey-patch 在 4.49 找不到 `get_keys_to_not_convert` | 4.x 中该函数在 `integrations.bitsandbytes`，5.x 中在 `quantizers.base` | `models/lora.py`: importlib 尝试两个位置 |
| 5 | 训练稳定性未知 | IMG_CONTEXT QLoRA 历史无成功案例 | 保守参数: lr=5e-5, r=8, max_patches=1 |

### 2.2 代码变更

**`scripts/train_lora.py`**:
```python
# InternVL3.5 processor loading bypass
if args.model == "internvl35":
    from transformers import AutoTokenizer as _Tok
    processor = _Tok.from_pretrained(
        config.model_id,
        trust_remote_code=model_cfg.get("trust_remote_code", False),
        **_internvl_extra,  # includes local_files_only
    )
# Force single-patch mode for training stability
if args.model == "internvl35":
    image_processor.max_patches = 1
```

**`models/lora.py`** — `_patch_bitsandbytes_compat()` 重写:
```python
for module_name in (
    "transformers.quantizers.base",       # 5.x API
    "transformers.integrations.bitsandbytes",  # 4.x API
):
    try:
        _mod = importlib.import_module(module_name)
        _orig = _mod.get_keys_to_not_convert
        break
    except (ImportError, AttributeError):
        continue
```

**`scripts/train_lora.py`** — `local_files_only` 修复:
```python
# InternVL2/3/3.5 的 processor 和 image_processor 加载也需要 local_files_only
_internvl_extra = {}
if model_cfg.get("model_kwargs", {}).get("local_files_only"):
    _internvl_extra["local_files_only"] = True
```

### 2.3 训练结果

**全量训练 (3854 步, lr=5e-5, r=8, max_patches=1)**:

| Step | rep_ratio | avg_len | 状态 |
|------|-----------|---------|------|
| 0 | 0.870 | 24.4 | baseline |
| 100 | 0.856 | 26.3 | ✅ 稳定 |
| 400 | 0.957 | 9.8 | ✅ 最佳 (avg_len 收敛到 GT 中位数) |
| 2000 | 0.914 | 12.0 | ✅ 稳定 |
| 3000 | 0.917 | 12.5 | ✅ 稳定 |
| 3800 | 0.920 | 12.8 | ✅ 最终 |

**全程 rep_ratio 范围: 0.856–0.957，无崩溃。**

---

## 3. InternVL3 修复尝试（失败）

### 3.1 尝试

| 尝试 | lr | 结果 |
|------|-----|------|
| v1 (前次会话) | 1e-4 | step 100 崩溃 (rep=0.398) |
| v2 (本次) | 5e-5 | step 200+ 退化，step 500+ 崩溃 |

### 3.2 退化序列

```
step=  0 rep=0.8991 avg_len=22.4  ← baseline healthy
step=100 rep=0.8206 avg_len=45.4  ← 开始压力
step=200 rep=0.6933 avg_len=34.7  ← 仍可接受
step=400 rep=0.5166 avg_len=42.5  ← 临界
step=500 rep=0.3934 avg_len=66.9  ← 崩溃
step=600 rep=0.1733 avg_len=91.1  ← 完全崩溃
```

lr 从 1e-4 降到 5e-5 仅将崩溃延迟了约 400 步，未能防止。

### 3.3 额外修复

训练加载时发现 processor/image_processor 缺少 `local_files_only=True`，导致静默网络挂起（进程状态 `do_select`）。已在上节 `_internvl_extra` 中修复。

---

## 4. InternVL generate() 输出格式差异

### 4.1 问题

推理时所有 caption 为空字符串。

### 4.2 根因

`InternVLChatModel.generate()` 内部调用 `language_model.generate(inputs_embeds=...)`，由于不给 `input_ids`，返回的 tensor 只包含**新生成的 token**，不含输入前缀。

标准 HuggingFace 模型的 `generate()` 返回 `[input_ids | generated_tokens]` 拼接结果。

### 4.3 修复

`scripts/inference_lora.py`:

```python
# ❌ 错误（标准模型路径，对 InternVL 返回空 tensor）:
generated = output_ids[:, input_len:]

# ✅ 正确（InternVL 路径）:
generated = output_ids  # 输出只含生成 token，无需 slice
```

### 4.4 训练代码比对

`scripts/train_lora.py` 的 `run_validation()` 已正确处理（line 428-434）:

```python
# InternVL2.generate() passes inputs_embeds to the language model,
# so the returned output_ids contain ONLY generated tokens (no input
# prepended). Standard models return input_ids + generated tokens.
if is_internvl2:
    generated = output_ids
else:
    generated = output_ids[:, input_len:]
```

推理脚本缺少这段逻辑，现已补充。

---

## 5. IMG_CONTEXT QLoRA 稳定性分析

### 5.1 对比数据

| 模型 | Backbone | Connector | lr | 崩溃步数 | 结论 |
|------|----------|-----------|-----|----------|------|
| InternVL2-8B | InternLM2 | IMG_CONTEXT | 1e-4 | 450 (baseline 已退化) | ❌ 根本不可行 |
| InternVL3-8B | Qwen2 | IMG_CONTEXT | 1e-4 | 100 | ❌ 快速崩溃 |
| InternVL3-8B | Qwen2 | IMG_CONTEXT | 5e-5 | 500 | ❌ 仅延迟崩溃 |
| **InternVL3.5-8B** | **Qwen3** | IMG_CONTEXT | 5e-5 | **无 (3854步)** | ✅ 稳定 |

### 5.2 根因假设

Qwen3 attention 使用 `attention_bias` + `head_dim`，提供了更好的梯度流动。IMG_CONTEXT tokens（每张图 256 个相同 `<IMG_CONTEXT>` token）产生极端重复模式，4-bit 量化 + LoRA 下 Qwen2 的 attention 无法补偿。

### 5.3 工程建议

| Backbone | QLoRA 策略 |
|----------|------------|
| Qwen3 (native) | 安全。lr=1e-4, r=16 均可。 |
| Qwen3 (IMG_CONTEXT) | 安全但需保守。lr≤5e-5, r≤8, max_patches=1。 |
| Qwen2 (IMG_CONTEXT) | ZS only。不要尝试 LoRA。 |
| InternLM2 (IMG_CONTEXT) | 完全避免。ZS baseline 已退化。 |

---

## 6. 推理修复

### 6.1 inference_lora.py InternVL 路径

为 `inference_lora.py` 添加了完整的 InternVL 推理路径，镜像训练代码中的 `run_validation()`：

1. 检测 `is_internvl = args.model in ("internvl2", "internvl3", "internvl35")`
2. InternVL3.5: 使用 `AutoTokenizer` 而非 `AutoProcessor`
3. 加载独立的 `AutoImageProcessor`，设置 `max_patches=1`
4. 展开 `<image>` → `<img><IMG_CONTEXT>×256</img>`
5. 使用 plain-string chat template（非结构化 content）
6. `model.generate(pixel_values=..., input_ids=..., attention_mask=...)`
7. 输出: `generated = output_ids`（无 slice）

### 6.2 评估结果

| 模型 | S_m | BLEU-4 | CIDEr |
|------|-----|--------|-------|
| internvl35-lora | **18.92** | 7.50 | 41.01 |
| qwen3vl-lora | 21.15 | 8.03 | 47.66 |
| qwen2vl-lora | 9.00 | 6.72 | 42.11 |
| llava-lora | 4.18 | 0.68 | 1.62 |
| internvl2 (ZS) | 4.06 | 0.81 | 1.61 |

---

## 7. 文件变更摘要

| 文件 | 变更 |
|------|------|
| `scripts/train_lora.py` | internvl35 processor bypass, local_files_only 补充, max_patches=1 |
| `scripts/inference_lora.py` | 新增 InternVL 推理路径, generate() 输出格式修复 |
| `models/lora.py` | _patch_bitsandbytes_compat 跨 4.x/5.x 兼容 |
| `config/training.py` | 无变更（配置已存在） |
| `transformers` | 4.49.0 → 4.51.3 (Qwen3 support) |

---

## 8. 经验教训

1. **IMG_CONTEXT QLoRA 稳定性由 backbone 决定，不是超参数。** 降 lr 只能延迟崩溃。
2. **InternVL generate() API 与标准 HF 模型不同。** 返回值只含生成 token。
3. **离线环境中所有 `from_pretrained()` 都需要 `local_files_only=True`。** 包括 processor、image_processor、tokenizer。
4. **Transformers 版本升级是连锁反应。** 新模型→版本→CUDA lib→monkey-patch。
5. **InternVL3 QLoRA 不可行不代表 InternVL3 ZS 不可用。** ZS 推理走 wrapper 管线，不受 QLoRA 稳定性影响。

---

## 相关文件

- [[2026-06-07-vlm-experiment-supplement-strategy]] — 实验补充策略
- [[2026-06-05-lora-nan-analysis]] — bf16 NaN 分析
- [[2026-06-04-vlm-baseline-plan]] — 原始实验计划
- `models/lora.py` — QLoRA 加载工具
- `scripts/train_lora.py` — 训练脚本
- `scripts/inference_lora.py` — 推理脚本
- `config/training.py` — 训练配置
- `data/training_dataset.py` — 训练数据集
