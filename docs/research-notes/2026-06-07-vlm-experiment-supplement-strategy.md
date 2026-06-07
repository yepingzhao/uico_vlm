# 论文 VLM 实验补充策略

**日期**: 2026-06-07
**触发**: VLM baseline 时效性调研结论 → 升级实验模型名单
**关联文档**:
- [[2026-06-04-vlm-baseline-plan]] — 原始实验计划（12 个关键决策）
- [[2026-06-06-vlm-baseline-currency-analysis]] — 时效性调研（本次升级的依据）
- [[2026-06-05-lora-nan-analysis]] — bf16 修复（Qwen3-VL LoRA 的前置条件）

---

## 1. 背景：为什么需要调整

### 1.1 原始计划的模型名单（2026-06-04）

| 模型 | 发布时间 | 定位 |
|------|----------|------|
| LLaVA-1.5-7B | 2024年1月 | 经典 baseline |
| Qwen2.5-VL-7B | 2025年1月 | 中文 VLM 代表 |
| InstructBLIP-7B | 2023年 | 弱对照组 |
| InternVL2-8B | 2024年 | 强 VLM 参比 |

### 1.2 时效性调研结论（2026-06-06）

| 模型 | 判断 | 问题 |
|------|------|------|
| LLaVA-1.5-7B | ✅ 仍合格 | VLM 领域的 ResNet-50，NeurIPS/WACV 2026 仍广泛使用 |
| Qwen2.5-VL-7B | ❌ 已过时 | 被 Qwen3-VL (2025.9) + Qwen3-VL-Flash (2026.1) 两代取代 |
| InternVL2-8B | ❌❌ 严重过时 | 被 InternVL3 (2025.4) + InternVL3.5 (2025.8) 两代取代，代际提升 +11.5 推理分 |
| InstructBLIP-7B | ⚠️ 旧但可用 | 其「弱」是叙事需要的对照组 |

### 1.3 审稿人风险矩阵

如果在 2026 年提交的论文中使用 2024 年初的模型作为「modern VLM baselines」：

- **InternVL2-8B**：审稿人最高概率质疑点。InternVL3.5 基于 InternVL3 架构（2025年4月 blog），后者在基准测试表中直接比较了 Qwen2.5-VL 和 InternVL2.5，展示代际差距。审稿人会问 "why not InternVL3.5?"
- **Qwen2.5-VL-7B**：次高风险。Qwen3-VL 在 Vision Arena 排名开源第一、全球第二，且已有 Qwen3-VL-Flash 小模型超越 Qwen2.5-VL-72B 的证据
- **LLaVA-1.5-7B**：低风险，甚至正面 — 审稿人期望看到它

---

## 2. 更新后的模型名单

### 2.1 推荐方案：升级 2/4 模型

| 模型 | 角色 | FEW-SHOT | LoRA | 工程状态 |
|------|------|----------|------|----------|
| **LLaVA-1.5-7B** | 经典 baseline（保留） | ✅ 已有 | ✅ 已有 | 全管线完备 |
| **Qwen3-VL-8B** | 升级替代 Qwen2.5-VL | ✅ 已有 wrapper | ⚠️ LoRA 配置已有，待验证 | wrapper 已存在 |
| **InstructBLIP-7B** | 弱对照组（保留） | — | — | 全管线完备 |
| **InternVL3.5-8B** | 升级替代 InternVL2-8B | ✅ | ✅ | ✅ wrapper 已创建 |

### 2.2 为什么是 InternVL3.5-8B

| 维度 | InternVL3.5-8B | InternVL3.5-8B |
|------|-------------|----------------|
| 发布时间 | 2025年4月 | 2025年8月 |
| 架构 | Qwen2.5-7B backbone + InternViT | Qwen3-8B backbone + InternViT |
| 推理代码 | 基于 InternVLChatModel，chat() API | 相同的 chat() API |
| 代际优势 | — | Cascade RL 后训练，推理性能 +16%，推理速度 4× |
| 审稿安全性 | 2025年4月，合格 | 2025年8月，更新更强 |

**选择 InternVL3.5-8B 的理由**：在已验证 chat() API 兼容的前提下，InternVL3.5 是更新、更强的选择。其 Cascade RL 后训练带来了显著的推理性能提升，且模型文件已下载、训练配置已存在（config/training.py:internvl35），工程成本为零。2025年8月的发布日期进一步强化了"最新代 VLM 也无法胜任域特定任务"的叙事。

### 2.3 保留 InstructBLIP 的理由

InstructBLIP 在原始计划中的定位是「预期最弱、作为下界」。这个角色对叙事 A（专家模型以小博大）至关重要：
- 它证明不是所有 VLM 都能不经微调直接用于这个 domain
- 它设置在 LLaVA-1.5 之下，建立了一个清晰的能力梯度
- 审稿人不会质疑 InstructBLIP 的时效性，因为没人期望它是 SOTA

---

## 3. 工程差距分析

### 3.1 当前代码库状态

```text
已有 wrapper (models/):
  llava.py            ✅ LLaVA-1.5 — ZS + FS + LoRA 全管线完备
  qwen3vl.py          ✅ Qwen3-VL-8B — ZS 可用，FS 有 _build_fewshot_inputs
  qwen2vl.py          ✅ Qwen2.5-VL — 保留作为 fallback
  instructblip.py     ✅ InstructBLIP — ZS 可用
  internvl2.py        ✅ InternVL2-8B — ZS 可用 (via _internvl_base.py)
  internvl25.py       ✅ InternVL2.5-8B — ZS 可用
  internvl35.py       ✅ InternVL3.5-8B — ZS + FS 可用，LoRA 已验证
  phi4_multimodal.py  ✅ Phi-4-mm — ZS 可用（新增 baseline，不在核心四模型内）

已有 LoRA 配置 (config/training.py):
  llava               ✅ 7 modules, r=16
  qwen2vl             ✅ 4 modules, r=16 (Qwen2.5-VL)
  qwen3vl             ✅ 4 modules, r=16 (Qwen3-VL) — 配置已存在
  internvl35          ✅ 4 modules, r=16 — wrapper 已存在，配置完备
  internvl2           ✅ 5 modules, r=16 — 已有但 InternLM2 backbone 命名不同

已有 MODEL_REGISTRY (config/__init__.py):
  qwen3vl             ✅ 已注册
  internvl35          ✅ 已注册
```

### 3.2 需要新增的工作量

| 任务 | 复杂度 | 预计时间 | 依赖 |
|------|--------|----------|------|
| `models/internvl35.py` wrapper | 低 | 0.5h | `_internvl_base.py` 可直接复用 |
| `config/__init__.py` 添加 internvl35 | 低 | 5min | wrapper 就绪 |
| 验证 InternVL3.5 ZS 推理 | 低 | 0.5h | wrapper + HF 模型已下载 |
| 验证 Qwen3-VL ZS 推理 | 低 | 0.5h | wrapper 已有 |
| 验证 Qwen3-VL few-shot 推理 | 中 | 1h | few-shot content builder 兼容性 |
| 验证 Qwen3-VL LoRA 训练 | 高 | 2-4h | NaN 风险、target_modules 验证 |
| 添加 InternVL3.5 few-shot 支持 | 中 | 1h | chat() history API |

### 3.3 InternVL3.5 Wrapper 设计

InternVL3.5 使用与 InternVL2 相同的 `chat()` API（`InternVLChatModel`），架构继承关系为：

```text
InternVL2 → InternVL2.5 → InternVL3 → InternVL3.5
  chat()     chat()       chat()      chat()
```

因此 `_internvl_base.py` 可以直接复用。InternVL3.5 wrapper 仅需：

```python
class InternVL35Wrapper(InternVLBase):
    model_id = "OpenGVLab/InternVL3_5-8B"

    @property
    def model_name(self) -> str:
        return "internvl35"
```

与 InternVL2 的关键差异：
- InternVL3.5 的 LLM backbone 为 Qwen3（不同于 InternVL2 的 InternLM2-7B）
- Tokenizer 为 Qwen2Tokenizer（使用 AutoTokenizer 加载，不再需要 tokenization_internlm2 模块）
- 因此 `_load_tokenizer()` 必须 override，使用 `AutoTokenizer.from_pretrained()`


**需要验证的关键点：chat() 函数签名（已验证兼容，pixel_values + question + generation_config）。

### 3.4 Qwen3-VL LoRA 风险

基于 [[2026-06-05-lora-nan-analysis]] 的经验：

- **已修复**：bf16 compute dtype 解决了 Qwen 系列 gated MLP 的大 intermediate_size 导致的 fp16 溢出
- **新风险**：Qwen3-VL 的 DeepStack 模块融合了 ViT 多层特征，可能引入新的需要特殊处理的层
- **Mitigation**：training.py 已配置为 4 modules（attention-only），不触碰 MLP 的 gate/up/down。DeepStack 位于 vision encoder 侧，vision encoder 在 QLoRA 中是 frozen 的，所以 DeepStack 不应引入新问题
- **验证策略**：先跑 1 epoch × 100 samples 快速测试，确认 loss 曲线在 bf16 下无 NaN

---

## 4. 更新后的实验阶段

```
Phase 0 — 补齐工程（新增）
  ├─ 创建 InternVL3.5 wrapper
  ├─ 注册 internvl35 到 MODEL_REGISTRY
  ├─ 确认 Qwen3-VL-8B 和 InternVL3.5-8B 模型文件已下载
  └─ 快速验证两个新模型的 ZS 推理（--subsample 5）

Phase 1 — 速查（同原计划）
  └─ LLaVA ZS dev mode (--subsample 100) → 快速评估 VLM 能力基线

Phase 2 — Zero-Shot 全覆盖（模型名单更新）
  └─ instructblip + llava + qwen3vl + internvl35 × Prompt A × 3500 images
     （如果原 internvl2 的 ZS 结果已经跑完，可保留作为
      supplementary table 中的比较数据，证明「代际提升」）

Phase 3 — Few-Shot（模型名单更新）
  └─ llava + qwen3vl + internvl35 × k=1,3 × 3500 images
  └─ llava + qwen3vl + internvl35 × k=1,3 × 3500 images
     （internvl35 通过 chat() history API 实现 few-shot）

Phase 4 — Prompt Sensitivity (Appendix)（不变）
  └─ llava + qwen3vl × Prompt B/C × subsample=500

Phase 5 — LoRA（模型名单更新）
  ├─ llava-lora: 已有，直接跑（bf16 重跑确保与 Qwen3-VL 公平对比）
  ├─ qwen3vl-lora: 需验证（Qwen2.5-VL LoRA 结果如已有，可在 supp 中保留）
  └─ internvl35-lora: 已跑通，training.py 配置完备
  ├─ llava-lora: 已有，直接跑（bf16 重跑确保与 Qwen3-VL 公平对比）
  └─ qwen3vl-lora: 需验证（Qwen2.5-VL LoRA 结果如已有，可在 supp 中保留）

Phase 6 — 评估（不变）
  ├─ ref_based.py (BLEU/METEOR/ROUGE/CIDEr/SPICE)
  ├─ ref_free.py (CLIPScore + RefCLIPScore)
  └─ 人工评估 (200 samples × 3 evaluators × 3 dimensions)

Phase 7 — 论文修改（不变）
  ├─ Experiment Section 重写
  ├─ Table I 新增 VLM ZS 行
  ├─ Table II 新增 VLM scaling 表
  ├─ Efficiency table (params + VRAM + latency)
  └─ Human eval subsection
```

---

## 5. 叙事策略更新

### 5.1 主叙事不变

**叙事 A：「专家模型以小博大」** 的主体逻辑不变：
- VLMs 是通用模型，在 urban incivility 这个 specialized domain 上 zero-shot/few-shot 表现不佳
- UIFormer (~200M) 在精度和效率上不可替代
- LoRA 微调后 VLM 才能部分追上

### 5.2 新增叙事层：VLM 代际进步

升级模型名单后，可以增加一个次要叙事维度：

> "Even the latest generation of VLMs (InternVL3.5, Qwen3-VL) — which substantially outperform their predecessors on general benchmarks — struggle on this specialized domain, reinforcing the need for domain-specific approaches."

这反而**强化**了原始叙事：不仅是「老 VLM 不行」，而是「连最新最强的 VLM 也不行」。

### 5.3 旧模型结果的处理

**保留作为 supplementary material**：
- Qwen2.5-VL 的 ZS/FS/LoRA 结果 → 证明 Qwen2.5-VL → Qwen3-VL 的代际进步在通用 benchmark 上显著但在 captioning domain 上可能有限
- InternVL2-8B 的 ZS 结果 → 证明 InternVL2 → InternVL3.5 的代际进步

**审稿人 value**：如果你能展示 "InternVL3.5 在 MMMU 上比 InternVL2 高 30+ 分，但在 UICO captioning 上只高了 X 分"，这本身就是一个 contribution — 说明通用 benchmark 的提升不一定迁移到 specialized domain。

---

## 6. 风险评估与应对

### 6.1 风险矩阵

| 风险 | 概率 | 影响 | 应对 |
|------|------|------|------|
| InternVL3.5 chat() API 不兼容 | 低 | 中 | fallback 到 InternVL2.5（已有 wrapper）或手动构建 processor 管线 |
| Qwen3-VL LoRA NaN | 中 | 高 | bf16 应解决根因；如仍 NaN，4 modules 已避开 MLP；如仍 NaN，降 lr 或转 2 modules |
| Qwen3-VL few-shot OOM | 中 | 中 | 已有 Qwen2.5-VL 低分辨率 processor 配置可复用 |
| InternVL3.5 8B 显存不足 | 低 | 高 | InternVL3.5-8B 参数量与 InternVL2-8B 接近，InternVL2-8B 已在 24GB 上通过 |
| 训练/推理时间窗口不够 | 中 | 高 | 优先 Phase 0+2（ZS 全覆盖），Phase 5（LoRA）可降优先级 |
| 审稿人认为 Qwen3-VL 也不是最新 | 低 | 低 | Qwen3-VL 2025.9 发布，距离 TCSVT 审稿周期可接受；2026 年 Qwen3.5 尚未普及 |

### 6.2 Fallback 层级

```
Tier 1 (目标): LLaVA-1.5 + Qwen3-VL-8B + InternVL3.5-8B + InstructBLIP
Tier 2 (InternVL 降级): LLaVA-1.5 + Qwen3-VL-8B + InternVL3.5-8B + InstructBLIP
                       （InternVL3.5 不兼容时的降级方案）
Tier 3 (Qwen 降级):    LLaVA-1.5 + Qwen2.5-VL-7B + InternVL3.5-8B + InstructBLIP
                       （Qwen3-VL LoRA 无法收敛时的降级方案）
Tier 4 (最小改动):     LLaVA-1.5 + Qwen2.5-VL-7B + InternVL2-8B + InstructBLIP
                       （回到原始计划，接受审稿风险）
```

---

## 7. 预期结果表格草稿（更新版）

- 实验列表
  - Zero-Shot
    - BLIP-2
    - InstructBLIP
    - LLaVA-1.5-7B
    - Qwen3-VL-8B
    - InternVL3.5-8B
  - Few-Shot
    - LLaVA-1.5-7B
    - Qwen3-VL-8B
    - InternVL3.5-8B
  - Lora
    - LLaVA-1.5-7B
    - Qwen3-VL-8B
    - InternVL3.5-8B

**Table I: Comparison with VLM Baselines (Zero-Shot)**

```
Methods            | Type         | B@4  | M    | R    | C    | S    | S_m  | CLIP-S
BLIP-2             | VLM (ZS)     | tbd  | tbd  | tbd  | tbd  | tbd  | tbd  | tbd
InstructBLIP       | VLM (ZS)     | tbd  | tbd  | tbd  | tbd  | tbd  | tbd  | tbd
LLaVA-1.5-7B       | VLM (ZS)     | tbd  | tbd  | tbd  | tbd  | tbd  | tbd  | tbd
Qwen3-VL-8B        | VLM (ZS)     | tbd  | tbd  | tbd  | tbd  | tbd  | tbd  | tbd
InternVL3.5-8B       | VLM (ZS)     | tbd  | tbd  | tbd  | tbd  | tbd  | tbd  | tbd
--- (existing specialist rows above) ---
UIFormer           | Specialist   | 32.9 | 24.99| 46.6 | 126.6| 20.6 | 50.3 | tbd
```

**Table II: VLM Scaling Behavior**

```
Model              | Setting      | B@4  | M    | R    | C    | S    | CLIP-S
LLaVA-1.5-7B       | ZS           | tbd  | tbd  | tbd  | tbd  | tbd  | tbd
                   | FS (k=1)     | tbd  | tbd  | tbd  | tbd  | tbd  | tbd
                   | FS (k=3)     | tbd  | tbd  | tbd  | tbd  | tbd  | tbd
                   | LoRA         | tbd  | tbd  | tbd  | tbd  | tbd  | tbd
Qwen3-VL-8B        | ZS           | tbd  | tbd  | tbd  | tbd  | tbd  | tbd
                   | FS (k=1)     | tbd  | tbd  | tbd  | tbd  | tbd  | tbd
                   | FS (k=3)     | tbd  | tbd  | tbd  | tbd  | tbd  | tbd
                   | LoRA         | tbd  | tbd  | tbd  | tbd  | tbd  | tbd
InternVL3.5-8B     | ZS           | tbd  | tbd  | tbd  | tbd  | tbd  | tbd
                   | FS (k=1)     | tbd  | tbd  | tbd  | tbd  | tbd  | tbd
                   | FS (k=3)     | tbd  | tbd  | tbd  | tbd  | tbd  | tbd
                   | LoRA         | tbd  | tbd  | tbd  | tbd  | tbd  | tbd
```

**Supplementary Table: Generational Improvement on UICO Captioning**

```
Model              | Year  | MMMU (general) | UICO S_m (domain) | Δ
LLaVA-1.5-7B       | 2024  | ~36            | tbd               | —
Qwen2.5-VL-7B      | 2025  | 55.0           | tbd               | —
Qwen3-VL-8B        | 2025  | ~60+           | tbd               | Qwen3-VL − Qwen2.5-VL
InternVL2-8B       | 2024  | ~50            | tbd               | —
InternVL3.5-8B     | 2025  | ~70            | tbd               | InternVL3.5 − InternVL2
```

这张 supplementary table 直接回应了「为什么升级模型」的问题，同时展示了通用 benchmark 提升与 domain-specific 提升之间的不对等关系 — 如果 domain 提升远小于通用提升（很可能），这将强化「domain 特殊性」论证。

---

## 8. 实施优先级

按最小风险、最大收益排序：

```
Priority 1 — InternVL3.5 Wrapper + ZS 验证 ✅ 已完成
  └─ ZS + FS + LoRA 全管线已验证通过
  └─ 时间: ~1h | 风险: 低

Priority 2 — LLaVA ZS dev mode 速查（验证叙事可行性）
  └─ 同原计划 Phase 1
  └─ 时间: ~1h | 风险: 低

Priority 3 — 四模型 ZS 全量推理 + 评估（最大实验产出）
  └─ instructblip + llava + qwen3vl + internvl35 × Prompt A × 3500
  └─ 时间: ~4-6h GPU | 风险: 低

Priority 4 — Few-Shot 推理 + 评估
  └─ llava + qwen3vl × k=1,3 × 3500
  └─ 时间: ~3-5h GPU | 风险: 中（OOM）

Priority 5 — Qwen3-VL LoRA 验证 + 训练
  └─ 快速验证（100 samples × 1 epoch）→ 全量训练
  └─ 时间: ~6-12h GPU | 风险: 高（NaN/OOM）

Priority 6 — LLaVA LoRA bf16 重跑（公平对比）
  └─ 如果之前的 LLaVA LoRA 是用 fp16 跑的
  └─ 时间: ~3-5h GPU | 风险: 低

Priority 7 — Prompt Sensitivity + 人工评估
  └─ 同原计划 Phase 4 + Phase 6 人工评估部分
```

---

## 9. 论文修改要点

### 9.1 Introduction/Related Work

- 新增一段描述 VLM 发展的代际进步（LLaVA-1.5 → Qwen3-VL/InternVL3.5）
- 强调 "we select models spanning three generations of VLM development to provide a comprehensive baseline"
- 引用 InternVL3 blog (2025.4) 和 InternVL3.5 相关公告，加上 Qwen3-VL technical report (arXiv:2511.21631)，证明选择的模型代表了当前技术水平

### 9.2 Experiment Setup

- 明确每个模型的发布年份和选择理由：
  - LLaVA-1.5: "widely recognized standard baseline" (NeurIPS 2025 原话)
  - Qwen3-VL: "state-of-the-art open-source VLM (Vision Arena #1 open-source)"
  - InternVL3.5: "latest generation with Cascade RL post-training for strong reasoning (MMMU ~70)"
  - InstructBLIP: "classic zero-shot baseline for domain difficulty calibration"

### 9.3 Supplementary Material

- Generational improvement analysis（第7节中的 supplementary table）
- 如果实验结果显示 domain gap 问题，可以作为 future work 方向的论据

---

## 10. 与原计划的 diff 摘要

| 决策点 | 原计划 (2026-06-04) | 更新后 (2026-06-07) | 变更原因 |
|--------|---------------------|---------------------|----------|
| Qwen 模型 | Qwen2.5-VL-7B | **Qwen3-VL-8B** | 时效性：已被两代取代 |
| InternVL 模型 | InternVL2-8B | **InternVL3.5-8B** | 时效性：最新代，Cascade RL 后训练推理提升显著 |
| 新增工程 | Qwen2VL LoRA | **InternVL3.5 wrapper** + Qwen3VL LoRA 验证 | Qwen3VL wrapper 已有，InternVL3.5 wrapper 已创建 |
| 叙事 | 专家模型以小博大 | 保持不变 + 新增「代际进步也不够」层次 | 升级模型反而强化叙事 |
| 旧模型处理 | — | 保留在 supplementary 中 | 证明 domain gap 不受通用 benchmark 进步影响 |
| 实验阶段 | 7 phases | 8 phases (新增 Phase 0 补齐工程) | 新模型需要新增 wrapper |
| 风险 | 主要是 Qwen2VL LoRA NaN | Qwen3VL LoRA NaN（bf16 修复已降低风险） | bf16 修复解决了根因 |

---

## 相关文件

- 时效性调研: [[2026-06-06-vlm-baseline-currency-analysis]]
- 原始实验计划: [[2026-06-04-vlm-baseline-plan]]
- LoRA NaN 分析: [[2026-06-05-lora-nan-analysis]]
- 论文正文: `docs/tcsvt2026.tex`
- 审稿意见: `docs/reviewer-comments.md`
- 模型注册: `config/__init__.py`（已添加 internvl35）
- Training 配置: `config/training.py`（internvl35 配置已存在，已验证）
- Wrapper 基类: `models/_internvl_base.py`（InternVL3.5 可复用）
- 已有 wrapper: `models/internvl2.py`, `models/internvl35.py`, `models/qwen3vl.py`, `models/qwen2vl.py`
