#  组会汇报总结

# 1. GRPO 与相关技术

## 1.1 GRPO 的 Advantage 计算流程

- 对同一 Prompt 执行 **N 次 Rollout**，获得奖励。
- 按 UID 分组，计算组内 **均值、方差**，得到优势 Advantage。
- GRPO 将 **序列级优势广播** 为 token-level advantage。
- Rollout 保持等长；使用 **response-mask** 过滤无效 token（mask=0）。

------

## 1.2 GRPO 训练增强技巧

### ● Clip-Higher（非对称裁剪）

- Low = 0.2，High = 0.28
- 提升策略熵，增加样本多样性
- 解耦上/下限裁剪，提高训练稳定性

### ● 动态采样

- 过滤 reward 全 0 或全 1 的组
- 保证有效梯度

### ● 移除 KL 项

- 长 CoT 推理任务后期可移除 KL，使策略分布更多样

### ● 规则奖励（Rule Reward）

- 适合代码、数学推理等可验证领域

------

# 2. 动机：粗粒度 A 值 的问题

### ● GRPO 的统一 Advantage 存在局限

- 所有步骤同等奖励或同等惩罚 → 无法区分关键推理步骤
- 导致模型过早收敛于少量“安全路径”
- 出现 **熵坍缩（Entropy Collapse）**
  - 输出多样性下降
  - 探索能力变弱

**为此提出：UCAS（不确定性优势塑形）**

------

# 3. UCAS：基于不确定性的 Advantage Shaping

## 3.1 目标

- 在 RLVR 中引入模型自身“不确定性”信息
- 同时保持：
  - 推理精度
  - 输出多样性
  - 避免过度自信

------

## 3.2 UCAS 的双层机制

### （1）响应级（Response-Level）不确定性调制

- 基于模型整体自信度
- **正确但不自信 → 奖励增强**
- **错误但自信 → 惩罚增强**

### （2）Token-Level 确定性惩罚

- 使用 logits（未 softmax）衡量局部确定性
- 抑制“局部过度自信 token”
- 保留必要的多样性探索性

### （3）最终 Advantage 塑形

- 综合响应级 + token 级
- 提升 RL 稳定性与推理质量

------

# 4. 相关基线方法

| 方法                  | 核心思想                              |
| --------------------- | ------------------------------------- |
| PRIME-Zero            | 无标注过程奖励（PRM）                 |
| Oat-Zero              | 规则奖励 + Dr.GRPO 改进               |
| GRPO with Entropy Adv | 熵项作为 Advantage 的一部分，鼓励探索 |
| KTAE                  | token-level 关键 token 贡献估计       |

------

# 5. 主要实验结果

## 5.1 状态遮罩（State Masking Loss）

- 外部信息 token 不计算策略梯度与 KL
- 只优化模型自身生成 token
- 解决训练早期模型不遵守格式的问题

------

## 5.2 数据集

- **通用 QA**：NQ, TriviaQA, PopQA
- **多跳推理**：HotpotQA, 2Wiki, Musique, Bamboogle

------

## 5.3 Instr vs Base

- **Instruct**
  - 收敛更快
  - 初期性能更高
- **Base**
  - 更易在 RL 中进行分布迁移
  - 最终性能与 instruct 接近甚至追平

------

## 5.4 训练动态

- GRPO：收敛快，但后期可能不稳定
- PPO：更稳但慢
- RL 训练整体趋势：
  - 早期：response length↓，reward↑
  - 后期：length↑ + reward↑（模型使用 search）

------

# 6. IKEA：内外知识协同推理代理

## 6.1 Motivation

- LLM 在工具调用/检索过程中常出现：
  - 过度依赖检索
  - 内部知识未充分利用
  - 内外知识冲突
  - 推理成本高

### IKEA 的核心思想：

**优先利用内部知识 → 判断不足 → 再检索外部知识**

------

## 6.2 方法概述

### ● Knowledge-Boundary Aware Dataset

模型学习“何时需要检索”。

### 评价指标

- **准确率（EM）**
- **检索次数（RT）**

### 对比方法

- IR-COT
- FLARE
- R1
- Search-R1

------

## 6.3 训练动态与关键发现

### ● Zero-start 模型

一开始严重过度检索 → 最终仍可学到“何时搜索”。

### ● 核心结论

**内部知识不足时才应检索**

- IR-COT：Hard 表现好，但 Easy 崩塌（知识冲突）
- FLARE：检索次数极少（触发信号不可靠）
- R1：内部增强
- Search-R1：Hard 强，但过度检索

------

# 7. 口头置信度（Verbal Confidence）与 TTS 推理预算

## 7.1 Motivation

- 传统置信度针对单步任务
- 多步 agent 中会：遗忘、反复试错、置信度失真

### 发现

尽管口头置信度校准差，但与正确率 **强正相关**

- <70%：几乎必错

- > 95%：显著高正确率

**可作为有效的“相对质量信号”**

------

## 7.2 TTS：基于置信度的推理预算调度

### 置信度阈值 τ 的确定（开发集）

1. 每个样本跑一次，记录 verbal confidence
2. 按区间统计准确率
3. 找出“ ≥ τ 时可信度足够高”的阈值

------

## 7.3 三种 TTS 推理策略

### ● BrowseConf-Zero

- 每次从头
- 达到阈值立即停止
- 否则返回最高置信度答案

### ● BrowseConf-Summary

- 每轮生成摘要
- 下一轮注入摘要上下文
- **尝试次数最少**

### ● BrowseConf-Neg

- 把前面低置信度答案当“禁止重复”
- 强制生成不同答案
- **准确率最高**

------

## 7.4 实验结果

- BrowseConf-Neg 在多个模型上均最佳
- 在更少尝试次数下达到或超越 Self-Consistency（10 次采样）
- 动态推理预算明显节省 token & 计算量

------

#  总结

### ● GRPO 存在粗粒度信用分配 → 导致熵坍缩与收敛不足

### ● UCAS 利用不确定性，引入双层 Advantage Shaping → 稳定且更强推理

### ● IKEA 实现内部与外部知识的真正协同

### ● 口头置信度可用于推理预算调度（TTS），显著节省成本