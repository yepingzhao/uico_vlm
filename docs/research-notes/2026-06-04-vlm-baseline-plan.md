# VLM Baseline 对比实验计划

**日期**: 2026-06-04
**触发**: TCSVT 审稿人意见 — "absence of modern VLM baselines is a particularly serious weakness"
**决策方法**: grill-me session，逐层追问至收敛

---

## 审稿人核心批评 & 对应策略

| # | 批评 | 策略 |
|---|------|------|
| 1 | 缺失 VLM baseline | 新增 4 VLM × ZS/FS/LoRA 全矩阵对比 |
| 2 | 技术新颖性有限 | 强化 domain-motivated design 论证 + error analysis |
| 3 | 实验验证不充分 | 补充 ref-free eval + 人工评估 + efficiency table |

---

## 核心叙事定位

**叙事 A：「专家模型以小博大」**

VLMs 是通用模型，在 urban incivility 这个 specialized domain 上 zero-shot/few-shot 表现不佳。UIFormer (~200M 参数) 在这些场景上优于 7B+ VLMs，仅在被 LoRA 微调后 VLMs 才能部分追上。结论：domain-specific 专家模型在效率和精度上不可替代。

---

## 实验决策汇总 (12 个关键选择)

### 1. 叙事框架
- **选择**: 叙事 A — 专家模型以小博大
- **理由**: 与 UIFormer 的 contribution 一致，差异化最清晰

### 2. 实验深度
- **选择**: 4 个代表性 VLM，深入实验 (ZS + FS + LoRA)
- **弃用**: 10+ VLM 仅 zero-shot (广度不如深度有说服力)

### 3. 模型名单
- LLaVA-1.5-7B — 最经典 VLM，代码已有全管线支持
- Qwen2.5-VL-7B — 中文 VLM 代表（UICO 是中国城市数据集）
- InstructBLIP-7B — 纯 zero-shot baseline，预期弱，作为对照组
- InternVL2-8B — 强 VLM，zero-shot only 参比

### 4. 工程投入范围
- **选择**: 路径 1 — 仅新增 Qwen2VL LoRA 支持
- **弃用**: 全补 InternVL2 few-shot/LoRA（工程量大，边际收益低）

### 5a. VLM Prompt 策略
- **选择**: Prompt A 主表 + Prompt B/C 作为 Appendix sensitivity
- **理由**: 单一主 prompt 公平可复现；B/C sensitivity 强化「VLMs 对 prompt 敏感」论证

### 5b. 中文 Prompt
- **选择**: 移除，所有模型统一英文 Prompt A

### 5c. LoRA 训练 Prompt 格式
- **选择**: 各用各的 native chat template
- **理由**: 最公平 — 每个模型用自己的预训练格式才能发挥最优

### 6. LoRA 管线验证
- **选择**: 不做 COCO 验证，直接跑 UICO
- **风险**: 如 LoRA 结果不理想，无法区分「训练问题」还是「domain 不适合」

### 7. Few-Shot 设计
- **选择**: 固定 examples (全测试集共用)，k ∈ [1, 3]
- **弃用**: 动态 CLIP 检索 (引入额外 confound)；k=5 (显存风险)
- **辩护**: "conservative lower bound, fully reproducible, no retrieval confound"

### 8. 实验结果表结构
- **选择**: 方案 B — 两张表
  - Table I: Specialist models + VLM zero-shot (全局对比)
  - Table II: VLM scaling behavior (ZS → FS(k=1) → FS(k=3) → LoRA)
- **弃用**: 单张大表（太拥挤）；ZS+LoRA only（丢失 scaling 信息）

### 9. 自动指标局限性
- **选择**: RefCLIPScore + 人工评估（方案 D）
- RefCLIPScore: 全量 3500 sample，零工程量，化解 "n-gram bias" 质疑
- 人工评估: 小样本深度验证，审稿人最无法反驳的证据

### 10. 人工评估设计
- 3-way: GT vs UIFormer vs LLaVA-LoRA (best VLM variant)
- 3-dimension: Accuracy / Completeness / Normative Precision
- 3 evaluators: 实验室同事盲测
- 200 samples
- Cohen's κ 证明 inter-annotator agreement

### 11. 效率对比
- Total params (7B + adapter), 注明 adapter-only trainable params
- 不单独突出 adapter 参数量优势（帮 VLM 说话）
- 报告 inference latency + VRAM，不报告训练成本

### 12. 风险管理
- **选择**: 先跑 LLaVA zero-shot dev mode 看效果，再决定是否调整策略
- 场景 1 (VLM ZS > UIFormer): 转叙事 B，强调 efficiency
- 场景 2 (VLM LoRA > UIFormer): 转 efficiency + deployability
- 场景 3 (VLM 极差): RefCLIPScore 排除「措辞不同」混淆

---

## 实验阶段

```
Phase 1 — 速查
  └─ LLaVA ZS dev mode (--subsample 100) → 快速评估 VLM 能力基线

Phase 2 — Zero-Shot 全覆盖
  └─ instructblip + llava + qwen2vl + internvl2 × Prompt A × 3500 images

Phase 3 — Few-Shot
  └─ llava + qwen2vl × k=1,3 × 3500 images

Phase 4 — Prompt Sensitivity (Appendix)
  └─ llava + qwen2vl × Prompt B/C × subsample=500

Phase 5 — LoRA
  ├─ llava-lora: 已有，直接跑
  └─ qwen2vl-lora: 需开发 Qwen2.5-VL QLoRA adapter

Phase 6 — 评估
  ├─ ref_based.py (BLEU/METEOR/ROUGE/CIDEr/SPICE)
  ├─ ref_free.py (CLIPScore + RefCLIPScore)
  └─ 人工评估 (200 samples × 3 evaluators × 3 dimensions)

Phase 7 — 论文修改
  ├─ Experiment Section 重写
  ├─ Table I 新增 VLM ZS 行
  ├─ Table II 新增 VLM scaling 表
  ├─ Efficiency table (params + VRAM + latency)
  └─ Human eval subsection
```

---

## 模型能力矩阵 (当前状态)

| Model | Zero-Shot | Few-Shot | LoRA |
|-------|-----------|----------|------|
| LLaVA-1.5-7B | ✅ 已有 | ✅ 已有 | ✅ 已有 |
| Qwen2.5-VL-7B | ✅ 已有 | ✅ 已有 | ❌ 待开发 |
| InstructBLIP-7B | ✅ 已有 | — | — |
| InternVL2-8B | ✅ 已有 | — | — |

---

## 预期结果表格草稿

**Table I (新增 VLM 行)**:
```
Methods            | Type         | B@4  | M    | R    | C    | S    | S_m
BLIP-2             | VLM (ZS)     | tbd  | tbd  | tbd  | tbd  | tbd  | tbd
InstructBLIP       | VLM (ZS)     | tbd  | tbd  | tbd  | tbd  | tbd  | tbd
LLaVA-1.5-7B       | VLM (ZS)     | tbd  | tbd  | tbd  | tbd  | tbd  | tbd
Qwen2.5-VL-7B      | VLM (ZS)     | tbd  | tbd  | tbd  | tbd  | tbd  | tbd
InternVL2-8B       | VLM (ZS)     | tbd  | tbd  | tbd  | tbd  | tbd  | tbd
--- (existing specialist rows above) ---
UIFormer           | Specialist   | 32.9 | 24.99| 46.6 | 126.6| 20.6 | 50.3
```

**Table II (VLM scaling behavior)**:
```
Model              | Setting      | B@4  | M    | R    | C    | S
LLaVA-1.5-7B       | ZS           | tbd  | tbd  | tbd  | tbd  | tbd
                   | FS (k=1)     | tbd  | tbd  | tbd  | tbd  | tbd
                   | FS (k=3)     | tbd  | tbd  | tbd  | tbd  | tbd
                   | LoRA         | tbd  | tbd  | tbd  | tbd  | tbd
Qwen2.5-VL-7B      | ZS           | tbd  | tbd  | tbd  | tbd  | tbd
                   | FS (k=1)     | tbd  | tbd  | tbd  | tbd  | tbd
                   | FS (k=3)     | tbd  | tbd  | tbd  | tbd  | tbd
                   | LoRA         | tbd  | tbd  | tbd  | tbd  | tbd
```

---

## 相关文件

- 论文正文: `docs/tcsvt2026.tex`
- 审稿意见: `docs/reviewer-comments.md`
- 模型注册: `config/__init__.py`, `models/__init__.py`
- LoRA 配置: `config/training.py`
- LoRA NaN 分析 (bf16 修复): [[2026-06-05-lora-nan-analysis]]
- Few-shot 管线: `fewshot/sampler.py`, `scripts/run_fewshot.py`
