# WebDancer: Towards Autonomous Information Seeking Agency

【NIPS2025/阿里巴巴deep-research系列】构建“会搜索、会点网页、会多步推理”的agent，从数据构建 → 轨迹采样 → SFT 冷启动 → on-policy RL 强化的完整训练范式,测试数据集有（GAIA/WebWalkerQA）

## 1.动机

1.提供“会搜索、会点网页、会多步推理”的 Web Agent，提供一套从数据构建 → 轨迹采样 → SFT 冷启动 → on-policy RL 强化的完整训练范式，而不是只拼 prompt 或直接对比 agent 框架。

2.怎么“从零训练”一个像 Deep Research 那样的 Web Agent，怎么构建数据、怎么训练模型。

**现有 Web Agent 的 问题**
1.数据问题：

- 现有 Web QA 数据太浅（2–3 步就能搜完）
- 真正难的 benchmark（GAIA、WebWalkerQA）：只有测试集，样本规模极小  

2.轨迹问题：
- 直接用人写的或 prompt 生成的轨迹：不稳定、不可扩展
- 现有的多数工作只是： SFT或off-policy RL，agent 学不到“长时序 + 多工具”的真实行为分布

3.训练方法问题：

- 直接 on-policy RL：前期 agent 只会学“怎么调用工具”；探索极差、成本极高，缺少一个合理的“冷启动 + 强化”的训练路径

## 2. 方法(WebDancer 的整体框架)

Step I：构建“深信息需求”的 QA 数据

数据一：CRAWLQA

- 从权威网站 root page 出发（wiki / arxiv / github
- 递归点击子页面
- 再用 GPT-4o基于真实页面内容生成 QA

数据二：E2HQA（Easy-to-Hard QA）
- 从一个简单事实问答（SimpleQA）
- 每一步：抽取一个实体; 搜索新信息; 利用搜索到的信息将问题改写得更复杂，但最终答案保持不变

Step II：轨迹采样（Rejection Sampling）

Agent 形式：ReAct ：Thought → Action → Observation → ...

调用的工具：`search`：使用谷歌搜索相关信息。`visit`：访问某一个页面

轨迹过滤：

1. 必须符合格式  2. 最后得到的答案正确（LLM-as-Judge）3. 调用工具次数必须大于2次

得到“可教 agent 的高质量行为轨迹”

Step III：SFT 冷启动（Cold Start）

学习：React行为模式, 工具调用格式

损失射击：Observation token（调用工具的观察结果） 不算 loss

作者实验明确表明：没有 SFT，直接 RL，几乎学不会 agent（Pass@3 ≈ 5%）

Step IV：on-policy RL（DAPO）

使用DAPO进行强化学习

- 奖励设计：
    - `score-format`（输出是否是React格式，SFT已学习React格式数据，给小权重）
    - `score-answer`（模型最后答案的准确性，使用LLM judge回答是否正确，主奖励）

## 3. 实验结果

1. WebDancer的提升不是“模型变大”，而是学习到了如何做Deep-Research任务
- 同一个 backbone（Qwen / QwQ / GPT-4o）
- WebDancer > Vanilla ReAct
- 训练范式 + 数据，比“换模型”更重要**

2. 和 闭源 Deep Research 相比
- 现有的 DR 是：闭源， 大规模 online RL
- WebDancer：开源，可复现，系统性训练 pipeline

# RPO: Retrieval Preference Optimization for Robust Retrieval-Augmented Generation

【ACL】从DPO的视角引出DPO奖励中不可抵消项，然后推出新的基于DPO的奖励，并且引入了检索内容偏好奖励去训练模型

## 1. 动机

问题背景：RAG 的隐性假设是错误的
经典 RAG 默认假设：
检索到的内容 ≈ 真实可信知识
但现实是：

- 检索可能 **不相关**
- 检索可能 **过时 / 错误**
- 检索可能 **和模型参数知识冲突**

LLM偏好于检索到的文档：
一旦给了 context，就更信它（over-reliance），导致：检索到正确文档时候，模型回答效果很好。检索错误， 更严重的hallucination

现有 Adaptive RAG 的三类方案

| 类别            | 思路                   | 问题           |
| --------------- | ---------------------- | -------------- |
| Pre-Eval        | 生成前判断检索好不好   | 多一次模型调用 |
| Post-Eval       | 多个回答，事后挑最好的 | 推理成本爆炸   |
| Integrated-Eval | **边生成边评估**       | 本文工作       |

## 2. 理论分析

DPO / RLHF 直接用在 RAG 上是错的

DPO 的基本假设：

对于同一个输入 x
- y_w（好回答）
- y_l（坏回答）
- 来自同一个条件分布π(y | x)

在 RAG 场景中，**偏好对来自两个不同分布**：
- 参数回答：yp∼π(y∣x)y_p \sim \pi(y \mid x)yp∼π(y∣x)
- 检索回答：yn∼π(y∣x,Dr)y_n \sim \pi(y \mid x, D_r)yn∼π(y∣x,Dr)
**输入条件不一致**  
导致：
- DPO 推导里的*partition function 无法消掉
- 数学目标函数变得不可计算

Fabricated Answer 的致命问题（非常关键）
很多工作（包括 KnowPO）用一个 trick：
- “假装 parametric answer 也是在有检索的条件下生成的”
- parametric answer 在 `(x, Dr)` 下概率可能极低
- DPO 有 KL 约束 → 不允许模型大幅改变分布  
- 结果是：即使训练时告诉你 parametric answer 是对的，推理时模型还是选检索答案，DPO 在 RAG 里，会“形式上对齐，行为上仍偏向检索”
  

RPO 的核心思想：奖励=回答奖励 + 检索奖励
传统偏好优化：reward = r(x, y)
RPO 的视角：reward = r(x, y, R)
检索 R 本身也是一个“被奖励/惩罚的对象”



论文提出新的优化目标

```math
\max_{\pi_\theta} \mathbb{E}[r(x, y, R)] - \beta \mathrm{KL}(\pi_\theta(y, R|x) \| \pi_{ref}(y, R|x))
```

- 学习怎么回答
- 学习 什么时候信检索/什么时候忽略检索

## 3.实验

两阶段训练（Fig.2）
Phase 1：SFT（激活检索意识）
- 生成：
    - `y_p`（无检索）
    - `y_n+p`（有检索）
- 只保留 **发生冲突的样本**
- 目的不是学知识，而是：
- 让模型意识到：检索不一定对

Phase 2：RPO（偏好优化）
- 再生成一次冲突对
- 用优化目标做 preference optimization
- 检索质量被隐式编码进策略


六、实验结论
没有额外组件
- 推理时比 Pre/Post Eval 快得多 
  

提升稳定
- PopQA / NQ / TriviaQA / RGB 全提升
- 在 “检索错 & 参数对”的子集提升最明显

