# QLoRA Training NaN Root Cause Analysis

**日期**: 2026-06-05
**触发**: LLaVA & Qwen2.5-VL LoRA 训练中 loss=NaN
**结论**: 根因是 fp16 动态范围不足 + Qwen2.5-VL gated MLP 的高维度交互，修复为统一使用 bfloat16 compute dtype

---

## NaN 发生时间线

| 版本 | 配置 | Qwen2.5-VL | LLaVA-1.5-7B |
|------|------|:--:|:--:|
| v1 | r=8, α=16, ep=1, lr=2e-4, warmup=0.03, **4 modules** + loss mask bug | 无 NaN（权重无效） | 无 NaN（权重无效） |
| v2 | r=16, α=32, ep=2, lr=2e-4, warmup=0.03, **7 modules** | **NaN @ step 1850** | **NaN @ step 2950** |
| v3 | v2 + lr→1e-4, warmup→0.1 | **NaN @ step 2900** (越过旧 NaN 点 1850) | ✅ 稳定 |
| v4 | v3 + Qwen2.5-VL 缩减为 4 modules | 🔄 重启 | ✅ 继续稳定 |

**关键观察**:
- LLaVA: NaN 就是纯学习率问题。lr=2e-4 → NaN；lr→1e-4 → 稳定。7 modules 完全无问题。
- Qwen2.5-VL: lr 减半只**延迟**了 NaN（1850 → 2900），未能消除。lr=1e-4 时在 step 2900 仍然 NaN。

---

## 根因分析

### 两个模型的 MLP 代码完全一致

```python
# Qwen2MLP (Qwen2.5-VL) 和 LlamaMLP (LLaVA) 使用完全相同的 forward
def forward(self, x):
    return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
```

两者都用 SiLU 激活 + gated MLP。差异在于**维度规模**：

| 参数 | LLaVA | Qwen2.5-VL |
|------|------:|-----------:|
| hidden_size | 4096 | 3584 |
| intermediate_size | 11,008 | **18,944** |
| MLP 膨胀比 | 2.69× | **5.28×** |
| gate/up/down 各层参数 | ~45M | ~**68M** |
| LoRA B 矩阵行数 | 11,008 | **18,944** |

### NaN 的统计机制

1. `gate(x)` 是 intermediate_size 维向量，每维是 hidden_size 维点积（std ≈ 60）
2. LoRA 在 gate/up/down 上的扰动随训练步数增长
3. Qwen2.5-VL 的 intermediate_size 是 LLaVA 的 **1.72 倍** → 1.72× 更多的维度可以产生 LoRA 引起的异常值
4. 少数维度异常 → `SiLU(gate) × up` 乘法交互放大 → 超出 fp16 表示范围 (max 65,504)
5. 这解释了 lr 减半只延迟 NaN：LoRA 参数增长速率减半，但总会达到溢出阈值

### 为什么 4 modules 不 NaN

去除 gate/up/down 的 LoRA 后，MLP 的三层仅使用 4-bit 量化后的基础权重。基础权重在 fp16 下本来稳定（预训练时通常用 bf16），异常值来源消失。

### 为什么 LLaVA 7 modules 不 NaN

LLaVA 的 intermediate_size 较小（11,008 vs 18,944），LoRA 在 gate/up/down 上产生异常值的统计概率更低。在 lr=1e-4 的训练步数内（~7700 step），异常值增长尚未触及 fp16 溢出阈值。

---

## 修复方案

### 选定方案: 统一使用 bfloat16 compute dtype

**动机**:
- fp16 动态范围 (max 65,504) 是训练不稳定的根本原因
- bfloat16 与 fp16 内存占用相同，动态范围与 fp32 相同 (max ~3.4e38)
- bfloat16 是工业标准训练精度（HuggingFace Trainer 默认 --bf16，Llama/Qwen 均用 bf16 训练）
- GPU: RTX 4090 (Compute Capability 8.9) 原生支持 bfloat16，速度不损失

**改动**: `models/lora.py` 中 `bnb_4bit_compute_dtype=torch.float16` → `torch.bfloat16`

**方法论优势**: 
- 两个模型使用**完全相同的训练配置**（7 modules + lr=1e-4 + bf16），消除审稿人的方法论质疑
- 如果 Qwen2.5-VL 用 4 modules 而 LLaVA 用 7 modules，审稿人会质疑"性能差异来自模块数不同而非模型能力差异"
- bf16 不是"给 Qwen2.5-VL 打补丁"，而是"全局训练精度选择"，这在论文中完全可以正面论述

**论文表述**: 
> "We use bfloat16 mixed precision for all QLoRA training runs to ensure numerical stability across models with different architectural designs."

### 弃用方案: Qwen2.5-VL 单独使用 4 modules

- **方法论文风险**: 审稿人质疑公平性 — "Qwen2.5-VL 性能不如 LLaVA，是否因为只微调了 attention 而非 attention+MLP？"
- 无法辩护：这引入了一个无法消除的实验混淆因子

---

## 实施清单

- [ ] `models/lora.py`: `bnb_4bit_compute_dtype` → `torch.bfloat16`
- [ ] `scripts/train_lora.py`: 添加 NaN 检测（安全网）
- [ ] 重新训练 LLaVA-LoRA (bf16, 7 modules, lr=1e-4)
- [ ] 重新训练 Qwen2VL-LoRA (bf16, 7 modules, lr=1e-4)
- [ ] 验证两者无 NaN，loss 曲线正常
- [ ] 更新论文中的训练配置描述

---

## 相关讨论

- 实验计划: [[2026-06-04-vlm-baseline-plan]]
- 相关 commits: `237926a` (Qwen2.5-VL 低分辨率), `28f5a1b` (image_grid_thw), `e0a0a4db` (loss mask 对齐), `f32cf48` (warmup_ratio), `c9b84b1` (community-standard hyperparams)
