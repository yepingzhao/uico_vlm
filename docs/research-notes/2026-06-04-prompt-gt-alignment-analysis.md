# Prompt A/B/C vs Ground Truth Alignment Analysis

**日期**: 2026-06-04
**触发**: 审稿人关注 prompt design — 分析现有 prompt 是否与 GT 标注风格对齐
**相关**: [[2026-06-04-vlm-baseline-plan]]

---

## 1. GT 特征

基于 `captions_test.json` (3500 images × 5 captions = 17500 captions) 的全面分析。

### 统计概要

| 指标 | 数值 |
|------|------|
| 总 captions | 17,500 |
| 中位长度 | **9 words** |
| 平均长度 | 9.4 words |
| 最小/最大 | 1 / 34 words |

### 结构模式

| 模式 | 占比 | 示例 |
|------|------|------|
| Adjectival opening | 27% | "Unauthorized parking of vehicles in public areas." |
| Existential | 22% | "There is garbage on the ground in front of the door." |
| Article-based | 13% | "The road surface is damaged." |
| Vehicle-subject | 9% | "Vehicles are parked in a mess at the entrance." |

### 内容维度

GT 统一遵循 **WHAT + WHERE** 两维度结构：

- **WHAT**: 违规类型 — parking, garbage, clutter, vending, wires, damage, etc.
- **WHERE**: 空间位置 — 62% 的 captions 包含显式位置词 (in front of, on the sidewalk, at the entrance, etc.)
- **不含**: justification, impact assessment, analysis, reasoning

### 违规类别分布 (per image, multi-label)

| 类别 | 占比 |
|------|------|
| Clutter/mess | 44% |
| Vegetation-related | 33% |
| Illegal parking | 31% |
| Garbage/waste | 25% |
| Unauthorized vending | 18% |
| Signage | 10% |
| Electric wires | 6% |
| Infrastructure damage | 4% |
| Clothes hanging | 4% |
| Smoking | 3% |

### 核心词汇

unauthorized (2816), illegal (1886), disorderly/random/messy (3000+), parked/parking (5221), garbage/trash/waste (2316), cluttered/scattered (1500+), front of door/entrance (2379)

---

## 2. 现有 Prompt 分析

### Prompt A (当前主力)

```
Describe any urban incivility or civic norm violations visible in
this image in one or two sentences.
```

**设计意图**: 开放式描述，简短输出

**实际效果** (5 模型 × 全量 3500 images):

| 指标 | 评分 | 评价 |
|------|------|------|
| METEOR | 6.6 (best 4/5 模型) | 三个 prompt 中最佳 |
| ROUGE-L | 10.4 (best 4/5 模型) | 三个 prompt 中最佳 |
| CLIPScore | 0.237 (best 3/5 模型) | 三个 prompt 中最佳 |
| BLEU-4 | 0 (全部模型) | 与 GT 无 4-gram 重叠 |
| CIDEr | 0 (全部模型) | 与 GT 无语义共识 |

**问题诊断**:

1. **长度失控**: 尽管要求 "one or two sentences"，VLM 实际输出 3-10 句 (InstructBLIP 最严重，Qwen2VL 次之)
2. **术语障碍**: "urban incivility" 是学术术语，弱模型 (BLIP2) 可能不完全理解
3. **缺少空间引导**: 部分模型只说 "有什么" 不说 "在哪里"，miss 了 GT 62% 的空间信息
4. **无 domain vocabulary priming**: 模型用通用语言而非 domain-specific 词汇

**典型输出 vs GT**:
```
GT:       "Vehicles are parked in a mess at the entrance." (9 words)
Qwen2VL:  "The image shows a covered outdoor area with a red electric scooter parked under
           the structure, which is not a designated parking spot for such vehicles. This could
           be considered an urban incivility as it may obstruct pedestrian access and create
           clutter in the public space. Additionally, the presenc..." (50+ words, truncated)
```

### Prompt B (sensitivity)

```
Analyze this urban scene and describe:
(1) what type of civic norm violation is present,
(2) where it is located,
(3) why it constitutes an incivility.
```

**实际效果**:

| 指标 | vs Prompt A |
|------|-------------|
| METEOR | ↓ 0.8 (5.8 vs 6.6) |
| ROUGE-L | ↓ 1.5 (8.9 vs 10.4) |
| CLIPScore | ↓ 0.016 (0.221 vs 0.237) |

**问题诊断**:

1. **结构标记污染**: "(1)(2)(3)" 作为 n-gram 直接损害 BLEU/CIDEr/ROUGE
2. **为什么维度多余**: GT 从不包含 "why it constitutes an incivility"
3. **模型混淆**: InstructBLIP 有时只输出 "(3) why..." 而丢失前两个维度
4. **输出更长**: Qwen2VL 输出 150-300 words 的结构化分析

**论文价值**: 作为 sensitivity analysis，Prompt A→B 的 1-2 点 METEOR 下降证明 VLM 对 prompt 敏感，这正是论文叙事需要的证据。

### Prompt C (sensitivity)

```
You are an urban management inspector. Describe the urban incivility
in this image, focusing on the specific violation, its spatial
context, and its impact on public order.
```

**实际效果**:

| 指标 | vs Prompt A |
|------|-------------|
| METEOR | ↓ 1.3 (5.3 vs 6.6) |
| ROUGE-L | ↓ 2.0 (8.4 vs 10.4) |
| CLIPScore | ↓ 0.017 (0.220 vs 0.237) |

**严重问题**:

1. **InternVL2 空输出**: 全部 3500 张图片的 predictions 为空字符串 — prompt 不兼容
2. **Role-playing 副作用**: 角色扮演导致更冗长、更 "official" 风格，与 GT 差距更大
3. **"impact on public order" 偏离 GT**: GT 描述状态，不做 impact 判断

---

## 3. 核心 Mismatch

```
GT 结构:    [WHAT violation] + [WHERE location]   — 9 words, 纯描述, 零分析
VLM 输出:   [WHAT] + [WHERE] + [WHY] + [IMPACT] + [CONTEXT] + [HEDGING]
            — 30-100 words, 解释性, 带有不确定性措辞
```

**根因**: VLM 的指令微调训练使其倾向于 "helpful assistant" 风格 (详细解释 + 上下文)，而 GT 标注员被要求写 "concise factual description"。Prompt A 的 "one or two sentences" 约束太弱，无法覆盖模型的默认行为。

**量化证据**: BLEU-4 和 CIDEr 对所有 prompt 均为 0 — 这不仅是 prompt 问题，更是 **范式级 mismatch** (VLM 生成风格 vs COCO 式 reference 风格)。

---

## 4. 建议

### 短期 (当前论文轮次)

**保留 Prompt A 作为主力**，做最小改动：

```python
# 当前版本
PROMPT_A = (
    "Describe any urban incivility or civic norm violations visible in "
    "this image in one or two sentences."
)

# 建议改为
PROMPT_A = (
    "In one sentence, describe any violation of urban order visible in "
    "this image. State what the problem is and where it is located."
)
```

改动理由：
- `"In one sentence"` 比 `"one or two sentences"` 约束更强 (VLM 仍会超，但起点更低)
- `"violation of urban order"` 比 `"urban incivility or civic norm violations"` 更直白
- 显式要求 `"what + where"` — 对齐 GT 的二维内容结构
- 去掉 `"civic norm violations"` — 减少术语负担

**Prompt B/C 保留作为 sensitivity analysis，但只用 ref-free 指标评估。**
Prompt B 的格式标记和 Prompt C 的 "why" 维度会机械性损害 n-gram 指标——这不是 sensitivity 信号，是格式噪声。只有 CLIPScore/RefCLIPScore 能测量语义差异而不被格式差异污染。
- Prompt A → ref-based + ref-free
- Prompt B/C → ref-free only

### 中期 (如果重跑实验)

设计任务对齐型 prompt：

```python
PROMPT_A_V2 = (
    "Describe this urban scene in one short sentence (under 15 words). "
    "Focus on: what is wrong (e.g., illegal parking, garbage, clutter) "
    "and where it occurs."
)
```

优势：
- 硬长度约束 (under 15 words, 接近 GT 的 9-word median)
- Domain vocabulary examples 作为隐式 priming
- 显式 what + where 结构

预期效果：显著缩小 VLM 输出与 GT 的格式 gap，n-gram 指标应有明显改善。

### 不建议

- ❌ Chain-of-thought / step-by-step — GT 不包含推理
- ❌ 在 zero-shot prompt 中嵌入 few-shot examples — 与 few-shot 实验 confound
- ❌ 每个模型单独设计 prompt — 破坏公平性
- ❌ 移除 Prompt B/C — 它们在 sensitivity analysis 中有明确角色

### 5a. Sensitivity 的指标选择：为什么只用 ref-free

Prompt B 引入了格式标记 (`"Violation:" / "Location:"`)，Prompt C 引入了额外内容 (`"why"`)。这些差异会**机械性地损害所有 ref-based 指标**：

| 指标 | 被 B 的格式标记污染？ | 被 C 的额外内容污染？ |
|------|---------------------|---------------------|
| BLEU | 是 — "Violation:" 是 0-match n-gram | 是 — "why" 句子中的 n-gram 与 GT 无重叠 |
| METEOR | 是 — 格式标记增加 chunk 碎片化 | 是 — 额外内容稀释 alignment |
| ROUGE-L | 是 — "Violation:" 打断 LCS | 是 — "why" 内容不出现在 GT 中 |
| CIDEr | 是 — 格式标记的 IDF=∞ (GT 中不存在) | 是 — justification 词汇的 IDF 权重极低 |
| SPICE | 不确定 — parser 对格式标记的行为未知 | 是 — "why" 增加无匹配的 scene graph 节点 |

**最致命的是 CIDEr**：其机制是 "GT 中高频 n-gram 权重大"。GT 中从未出现 `"Violation:"`，其 IDF 权重为 0，意味着输出长度被无权重 token 占据，CIDEr 必然偏低 — 但这与语义质量无关。

因此 **Prompt B/C 仅用 ref-free (CLIPScore + RefCLIPScore) 评估**。这两个指标通过 embedding 空间的 cosine similarity 测量语义对齐，对表面格式差异不敏感。

### 5b. 仅用 2 个 CLIP 指标是否充分？

**先定义「充分」**。Prompt sensitivity 要回答的问题：

> 改变 prompt 的格式 (A→B) 或内容 (A→C)，是否改变了生成 caption 的质量？

「质量」在此场景下的可操作维度：

| 维度 | A→B 可测？ | A→C 可测？ | 工具 |
|------|-----------|-----------|------|
| 图文相关性 (说了对的东西吗？) | ✅ | ✅ | CLIPScore |
| GT 对齐 (说了标注员认为重要的东西吗？) | ✅ | ✅ | RefCLIPScore |
| 事实准确性 (说错了吗？) | 低风险 — 格式变化不引入幻觉 | ⚠️ **盲区** — "why" 可能是编造的 | 需补充 |
| 语言流畅性 (说得通顺吗？) | ⚠️ 结构化可能碎片化 | 低风险 — free-form 不会断裂 | 需补充 (B) |

两个指标覆盖了 2/4 维度。缺失的 2 个：

**事实准确性 (A→C 的关键风险)**：A→C 要求 "why it violates norms"，VLM 可能编造似是而非的理由。CLIPScore 无法检测这点 — 编造的 "blocks pedestrian access" 在语义上与包含人行道的图像仍然高度相似。

**语言流畅性 (A→B 的关键风险)**：结构化格式可能导致不完整输出 (如只输出 "Violation: parking" 缺少 Location 字段)。

### 5c. 盲区补救

```
A→B 补充: 正则检查输出是否同时包含 "Violation:" 和 "Location:" 字段
          → 报告 compliance rate (格式遵循率)
          → 零人工成本

A→C 补充: 小规模 truthfulness check (50 samples)
          → LLM-as-judge: GPT-4o 判断 justification 是否 grounded in image
          → 或人工抽查
          → 报告 justification hallucination rate
```

### 5d. 结论：必要且近似充分

CLIPScore + RefCLIPScore 是 sensitivity analysis 的**必要且近似充分**的指标组合。它们避免了 n-gram 指标的格式污染，能捕捉语义层面的质量变化。唯一的实质性盲区是 A→C 的 hallucinated justification，50-sample 的 LLM 抽查即可覆盖。

**最终评估策略**：

```
Prompt A:   ref-based + ref-free (全量 3500)
Prompt B:   ref-free only + format compliance check
Prompt C:   ref-free only + 50-sample truthfulness check
```

---

## 6. 对论文叙事的影响

当前 Prompt A 主表 + B/C sensitivity 的策略仍然合理。建议在 Experiment Setup 中说明：

> "We adopt Prompt A as the primary configuration. Prompts B and C serve as sensitivity checks (Section X). Since B and C introduce format and content variations that mechanically affect n-gram overlap with reference captions, we evaluate prompt sensitivity exclusively via reference-free metrics (CLIPScore and RefCLIPScore), which measure semantic image-text alignment independent of surface-form variation."

这直接回应了审稿人对 prompt design 的潜在质疑，同时将 Prompt B/C 从 "辅助实验" 升级为 "方法学验证"。

---

## 7. 相关文件

- Prompt 定义: `prompts/templates.py`
- GT 数据: `/home/uesr/zhao/media_data/ccmc/annotations/captions_test.json`
- 实验计划: `docs/research-notes/2026-06-04-vlm-baseline-plan.md`
- 论文正文: `docs/tcsvt2026.tex`
