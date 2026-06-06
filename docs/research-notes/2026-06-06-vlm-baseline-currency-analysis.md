# VLM Baseline 时效性分析：LLaVA-1.5 与 Qwen2.5-VL 作为 2026 年 Baseline 是否过时？

**日期**: 2026-06-06
**触发**: `/deep-research` 检索，评估当前 VLM baseline 选择的时效性
**方法**: 多源 Web 搜索 + 交叉验证
**关联文档**: [[2026-06-04-vlm-baseline-plan]]

---

## 执行摘要

**LLaVA-1.5-7B 作为 VLM baseline 仍然合格** — 它在 2025-2026 年学术界已成为 VLM 领域的 "ResNet-50"，大量 NeurIPS/WACV/CVPR 论文继续以其为标准对比基线。**Qwen2.5-VL-7B 已经过时** — 它被 Qwen3-VL（2025年9月）全面取代，Qwen3-VL-8B 在所有维度上都显著更强。**InternVL2-8B 也严重过时** — InternVL3（2025年4月）和 InternVL3.5（2025年8月）已经带来了代际级别的性能提升。

**核心建议：保留 LLaVA-1.5，将 Qwen2.5-VL 升级为 Qwen3-VL-8B，将 InternVL2 升级为 InternVL3-8B。**

---

## 1. LLaVA-1.5-7B：仍然合格 ✅

### 1.1 当前引用状态

LLaVA-1.5 在 2025-2026 年仍然是使用最广泛的 VLM baseline。以下为代表性强证据：

| 论文/会议 | 日期 | 使用方式 |
|-----------|------|----------|
| WACV 2026 — "Optimizing LVLMs with On-Policy Data for Hallucination Mitigation" | 2026年3月 | 使用 LLaVA-1.5-7B 和 -13B，-13B 版本在 MMHal-Bench 上超越 GPT-4V |
| Pattern Recognition (Elsevier) — "Instruction-guided fusion of multi-layer visual features" | 2026年2月 | 在 LLaVA-1.5-7B 上集成 IGVA 模块 |
| NeurIPS 2025 — "Towards Self-Refinement of VLMs with Triangular Consistency" | 2025年 | 明确称为 "the widely recognized LLaVA-1.5" 作为 baseline |
| arXiv 2025.06 — "Learning Compact Vision Tokens" | 2025年6月 | 在 LLaVA-1.5 上验证，仅用 25% token 达到相当性能 |

### 1.2 为什么 LLaVA-1.5 持续被使用

1. **开源可复现** — 权重、代码、架构完全公开，社区可任意 fork 对比
2. **性能适中但不饱和** — 胜任 baseline 角色，仍有改进空间，是理想的 testbed
3. **模块化架构** — CLIP-ViT → MLP adapter → Vicuna，组件可独立替换以做消融实验
4. **社区惯性** — 一旦成为标准参考点，新论文持续使用以保证结果可比性
5. **"VLM 领域的 ResNet-50"** — 正如 ResNet-50 在 SOTA 过去后仍作为标准视觉 baseline，LLaVA-1.5 承担了同样角色

### 1.3 对本论文的判断

**保留 LLaVA-1.5-7B 作为 baseline 是合理的，不需要升级到 LLaVA-NeXT。** 理由：
- 审稿人会认出这是标准 baseline，不会质疑
- LLaVA-NeXT（LLaVA-1.6, 2024年1月）已经 >2 年，且后续 LLaVA 系列发展不如 Qwen/InternVL 快
- 论文叙事「专家模型以小博大」需要一个相对弱但经典的代表
- LLaVA-1.5 已经是实验中最弱的 VLM variant（预期），不需要再找一个更弱的新模型

---

## 2. Qwen2.5-VL-7B：已经过时 ❌ → 建议升级

### 2.1 Qwen VL 系列演变时间线

```
Qwen-VL        ── 2023年8月   基础 VQA、caption、grounding
Qwen2-VL       ── 2024年9月   M-RoPE、视频理解、动态分辨率
Qwen2.5-VL     ── 2025年1月   长视频、Visual Agent、JSON 输出、最大72B  ← 当前使用
Qwen3-VL       ── 2025年9月   DeepStack、Interleaved-MRoPE、256K ctx、Thinking 模式、MoE
Qwen3-VL-Flash ── 2026年1月   高效小模型，性能超越 Qwen2.5-VL-72B
Qwen3.5        ── 2026年3月   Early Fusion 原生多模态 Agent
```

### 2.2 Qwen3-VL vs Qwen2.5-VL 核心升级

Qwen3-VL 相对于 Qwen2.5-VL 的关键架构提升：

1. **Interleaved-MRoPE** — 全频分配时间/宽度/高度维度，远优于 Qwen2.5-VL 的 M-RoPE
2. **DeepStack** — 融合 ViT 多层特征（而非仅最终层），极大提升细粒度对齐
3. **Text-Timestamp Alignment** — 精准时间戳视频定位
4. **Native 256K 上下文** — 可扩展至 1M tokens
5. **Thinking 变体** — VL Chain-of-Thought 推理增强
6. **3D spatial grounding** — 判断物体空间关系与遮挡

### 2.3 8B 级模型性能对比（从 InternVL3 blog 提取）

| Model | MMMU | MathVista | MathVerse | LogicVista | 综合推理 |
|-------|------|-----------|-----------|------------|----------|
| **InternVL3-8B** | **62.7** | **71.6** | 39.8 | **44.1** | **44.3** |
| Qwen2.5-VL-7B | 55.0 | 67.8 | 41.1 | 44.1 | 41.4 |
| InternVL2.5-8B | 56.2 | 64.5 | 22.8 | 36.0 | 32.8 |

> Qwen3-VL-8B 官方数据显示在多模态推理上大幅超越 Qwen2.5-VL-7B（技术报告 arXiv:2511.21631），在 Vision Arena 上排名开源第一、全球第二。

### 2.4 对本论文的判断

**Qwen2.5-VL-7B 应该替换为 Qwen3-VL-8B。** 理由：

- Qwen2.5-VL 已经经历了两个主要版本迭代（Qwen3-VL → Qwen3-VL-Flash → Qwen3.5）
- Qwen3-VL-8B 提供了 **Instruct** 和 **Thinking** 两种变体，可以增加实验深度
- 审稿人会注意到使用 Qwen2.5-VL 而非 Qwen3-VL，可能在 rebuttal 中被质疑
- Qwen3-VL-8B 的架构仍与 Qwen2.5-VL-7B 相似（均为 Qwen2.5 系列语言模型 backbone），升级路径相对平滑
- 工程成本：需要重新验证 few-shot、LoRA 管道在 Qwen3-VL 上的兼容性

**替代方案**：如果 Qwen3-VL 工程迁移成本过高，可以：
- 保留 Qwen2.5-VL 但明确在论文中标注其为 "2024-2025 representative"
- 在 Related Work 段引述 Qwen3-VL 的最新进展
- 在 limitation/future work 中提及升级到 Qwen3-VL

---

## 3. InternVL2-8B：严重过时 ❌❌ → 必须升级

### 3.1 InternVL 系列演变

```
InternVL2/2.5  ── 2024年      论文当前使用 ← 已过两代
InternVL3      ── 2025年4月   原生多模态预训练，代际跳跃 +11.5 综合推理点
InternVL3.5    ── 2025年8月   Cascade RL，推理性能再提升 16%，推理速度 4x
```

### 3.2 代际提升

| Model | MMMU | MathVista | MathVerse | LogicVista |
|-------|------|-----------|-----------|------------|
| **InternVL3.5-8B** | 显著提升 | 显著提升 | 显著提升 | 显著提升 |
| **InternVL3-8B** | **62.7** | **71.6** | 39.8 | **44.1** |
| InternVL2.5-8B | 56.2 | 64.5 | 22.8 | 36.0 |
| InternVL2-8B (当前使用) | 更低 | 更低 | 更低 | 更低 |

> InternVL2.5→InternVL3 的综合推理得分从 32.8 提升到 44.3（+11.5 点），是代际级别的飞跃。

### 3.3 对本论文的影响

**InternVL2-8B 是最迫切需要替换的模型。** 它在 2024 年就不再是 SOTA，经历两个大版本迭代后（InternVL3 → InternVL3.5），继续使用 InternVL2 在 2026 年的论文中会显得严重过时。

**建议替换为 InternVL3-8B 或 InternVL3.5-8B**：
- InternVL3-8B：更成熟，社区文档完善，推理代码更稳定
- InternVL3.5-8B：更强但更新（2025年8月），兼容性需验证

---

## 4. InstructBLIP-7B：可以保留 ⚠️

InstructBLIP-7B 在计划中定位为「纯 zero-shot baseline，预期弱，作为对照组」。这个定位仍然成立：
- InstructBLIP 是 2023 年的产物，不需要升级到 BLIP-3
- 它的弱恰好是论文叙事需要 — 证明不是所有 VLM 都能胜任这个任务
- 替换它不会带来实质性的实验价值

**建议保留。**

---

## 5. 2025-2026 年其他值得关注的 VLM

### 5.1 Phi-4-multimodal (Microsoft, 2025年2月)

- 5.6B 参数，MIT 开源
- MMMU 55.1, ChartQA 81.4, DocVQA 93.2%（超越 Gemini 2.0 Flash）
- 统一架构处理语音+视觉+文本
- **对本论文的价值**：作为「小型专精模型」的代表，和 >7B 的 VLM 形成对比

### 5.2 GLM-4.5V (Zhipu, 约 2025年12月)

- 106B-A12B MoE，42 个 benchmark SOTA
- RLCS 强化学习后训练
- **不推荐**：106B 太大，和 7B 级模型对比不公平，也不会为论文叙事增加价值

### 5.3 Qwen2.5-Omni-3B (Alibaba, 2026年5月)

- 实时 any-to-any 多模态交互模型，边缘部署
- **不推荐**：太小、太新、方向不匹配

### 5.4 不推荐的考虑

| 模型 | 不推荐原因 |
|------|------------|
| LLaVA-NeXT/LLaVA-1.6 (2024年1月) | 已过时 >2年，LLaVA 系列发展已停滞 |
| LLaVA-OV-7B | InternVL3 blog 显示综合推理仅 29.6，远低于 InternVL3-8B (44.3) |
| DeepSeek-VL2 | 需要专用 `deepseek_vl2` 包，已有依赖问题（计划中已提及） |
| Idefics3-8B | 已有 transformers 兼容性问题（计划中已提及） |
| Pixtral-12B (Mistral) | 12B，和 7B 量级不完全可比 |
| InternVL3.5-38B+ | 超出 24GB VRAM 约束 |

---

## 6. 更新后的模型名单建议

| 模型 | 当前计划 | 建议调整 | Zero-Shot | Few-Shot | LoRA | 优先级 |
|------|----------|----------|-----------|----------|------|--------|
| LLaVA-1.5-7B | ✅ 保留 | 不变 | ✅ | ✅ | ✅ | 🔴 必须 |
| Qwen2.5-VL-7B | ✅ 使用 | → **Qwen3-VL-8B** | ✅ | ✅ | 待验证 | 🔴 必须 |
| **InternVL3-8B** | ❌ | **新增，替换 InternVL2** | ✅ | — | — | 🟡 强烈推荐 |
| InstructBLIP-7B | ✅ 使用 | 不变 | ✅ | — | — | 🟢 保留 |
| ~~InternVL2-8B~~ | 使用 | **删除** | — | — | — | ❌ 删除 |
| ~~Qwen2.5-VL-7B~~ | 使用 | **删除** | — | — | — | ❌ 删除 |

### 6.1 升级方案对比

| 方案 | 模型清单 | 优势 | 劣势 |
|------|----------|------|------|
| **A (推荐)** | LLaVA-1.5 + Qwen3-VL-8B + InternVL3-8B + InstructBLIP | 三个时代的代表模型、时效性强 | Qwen3-VL LoRA 需要重新开发（Qwen2.5-VL LoRA 已有） |
| **B (保守)** | LLaVA-1.5 + Qwen2.5-VL-7B + InternVL3-8B + InstructBLIP | 仅移除最过时的 InternVL2 | Qwen2.5-VL 仍会被质疑 |
| **C (最小改动)** | LLaVA-1.5 + Qwen2.5-VL-7B + InternVL2-8B + InstructBLIP | 零改动 | 审稿风险最高 |

---

## 7. 时效性风险评估

### 7.1 审稿人可能质疑的点

1. **「Qwen2.5-VL 已被 Qwen3-VL 取代，为什么不用更新的模型？」** — 风险：高
2. **「InternVL2 已过两代，InternVL3/3.5 显著更强，结论是否仍然有效？」** — 风险：最高
3. **「LLaVA-1.5 发布于 2024 年初，为什么不是 LLaVA-NeXT？」** — 风险：低（LLaVA-1.5 仍被广泛接受为标准 baseline）

### 7.2 应对策略

**对于可能无法升级的模型**（如 LoRA 管线迁移成本太高）：
- 在论文中明确标注模型的时代背景："We select models spanning three generations of VLM development: LLaVA-1.5 (classic, Jan 2024), Qwen2.5-VL (representative early-2025), and InternVL3 (state-of-the-art mid-2025)"
- Related Work 中引述最新进展
- Limitation/Future Work 中说明升级到更新的模型版本

---

## 8. 实施路径建议

```
Phase 0 — 新增模型下载和基础推理验证
  ├─ download_models.py --model qwen3-vl-8b
  ├─ download_models.py --model internvl3-8b
  ├─ 验证 Qwen3-VL-8B 的 HF 推理兼容性
  └─ 验证 InternVL3-8B 的 HF 推理兼容性

Phase 1 — Zero-Shot 全覆盖（同原计划 Phase 2）
  └─ instructblip + llava + qwen3vl + internvl3 × Prompt A × 3500 images

Phase 2 — Few-Shot（Qwen3-VL-8B）
  └─ 需要验证 Qwen3-VL 的 few-shot API 是否与 Qwen2.5-VL 相同
  └─ 可能需要调整 processor 配置（减少分辨率避免 OOM）

Phase 3 — LoRA（Qwen3-VL-8B）
  └─ 需要检查 Qwen3-VL 的 target_modules（DeepStack 引入新层）
  └─ 可能需要比 Qwen2.5-VL 更激进的 resolution 缩减
  └─ 风险较高：新架构可能带来 NaN 问题

Phase 4 — 评估
  └─ 同原计划
```

---

## 9. 结论

| 模型 | 时效性判断 | 建议 | 理由 |
|------|-----------|------|------|
| **LLaVA-1.5-7B** | ✅ 仍然合格 | 保留 | VLM 领域的 "ResNet-50"，2026 年仍是标准 baseline |
| **Qwen2.5-VL-7B** | ❌ 已经过时 | 升级到 Qwen3-VL-8B | 已被 Qwen3-VL(2025.9) + Qwen3-VL-Flash(2026.1) 取代 |
| **InternVL2-8B** | ❌❌ 严重过时 | 升级到 InternVL3-8B | 已被 InternVL3(2025.4) + InternVL3.5(2025.8) 取代，代际提升 +11.5 推理分 |
| **InstructBLIP-7B** | ⚠️ 旧但可用 | 保留 | 作为弱对照组，其 "弱" 本身是叙事需要 |

**最低改动方案**：将 InternVL2-8B 替换为 InternVL3-8B（风险最低、收益最大）。
**推荐方案**：同时升级 Qwen2.5-VL-7B → Qwen3-VL-8B 和 InternVL2-8B → InternVL3-8B（平衡时效性与工程投入）。

---

## 数据来源

1. WACV 2026 proceedings — LLaVA-1.5 as hallucination mitigation baseline
2. NeurIPS 2025 — LLaVA-1.5 as "widely recognized" baseline
3. InternVL3 Official Blog (2025-04-11) — benchmark comparison tables, InternVL3 vs Qwen2.5-VL
4. Qwen3-VL Technical Report (arXiv:2511.21631, Nov 2025)
5. Phi-4-multimodal Technical Report (Microsoft, Feb 2025)
6. Hugging Face OpenGVLab/InternVL3_5-8B-MPO model card
7. Qwen3-VL-Flash launch announcement (Alibaba Cloud, Jan 2026)
8. Alibaba Cloud Qwen3-VL launch announcement (Sep 2025)
9. Qwen2.5-VL Technical Report (arXiv, Feb 2025)
10. LLaVA-NeXT blog post (Jan 2024) — benchmark comparison vs LLaVA-1.5

## 方法论

搜索了 20+ queries，覆盖 web 和 news。分析了来自 10+ 独特来源的信息。
研究子问题：
1. LLaVA-1.5 在 2025-2026 年论文中是否仍被作为标准 baseline 使用？
2. Qwen2.5-VL 的最新替代版本是什么？Qwen3-VL 比 Qwen2.5-VL 提升多少？
3. InternVL 系列最新版本是什么？InternVL3/3.5 比 InternVL2 提升多少？
4. 2025-2026 年 7-8B 级别的 SOTA VLM 有哪些？
5. 审稿人对 2026 年 VLM baseline 选择的期望是什么？
