# 视频生成相关论文汇报

**张志冰 · 2025.11.15**

------

# #1 图像生成 & 视频生成框架概览

- Diffusion Models 在图像/视频生成中表现强大，但：
  - **迭代步骤多**
  - **模型结构复杂**
- → 推理成本高，**难以部署到资源受限的设备**（移动端/边缘端等）

------

# #2 Motivation

> 在后训练（post-training）阶段，优化方向是什么？

- 减少 **denoising steps**（步骤数）？
- 还是减少 **每一步的推理计算量（per-step cost）**？

本文提出 **PostDiff**：
 一种 **训练后、无需重新训练（training-free）** 的 diffusion 压缩框架，从：

- **输入层面：Mixed-Resolution Denoising（混合分辨率去噪）**
- **模块层面：Module Caching（模块缓存策略）**

来提升推理效率。

------

# #3 方法（Method）

## ## 3.1 Mixed-Resolution Denoising（混合分辨率去噪）

### **核心思想**

- 在生成早期，仅使用低分辨率（如 1/2 或 1/4 resolution）进行去噪
- 后期恢复到高分辨率

### **为什么有效？**

- 低分辨率更容易学习**低频结构**（轮廓/布局）
- 能促进最终图像质量
- 推理速度显著提升

### **实验结果**

- FID 更低
- FLOPs 大幅下降
- 不同配置（T=8–20，s=0.1–0.6）中：
   → **低分辨率步数越少，效果越好**

------

## ## 3.2 Module Caching（模块缓存）

包含两个策略：

### ① **DeepCache**

缓存部分模型模块的计算结果，后续步骤重复利用。

### ② **Cross-Attention 缓存 + CFG 截断**

- 在后期步骤中复用 cross-attention
- CFG（Classifier Free Guidance）在布局已稳定后冗余
   → 后半程不再使用 CFG，可加速推理

------

# #4 PostDiff 综合方法

将三项技术组合：

1. Mixed-Resolution Denoising
2. Cross-Attention Caching
3. CFG Truncation

在多个 SOTA diffusion 模型上的实验表明：

- 运行速度显著加快
- 图像质量保持甚至提升
- 优于其他 training-free 压缩方法

------

# #5 第二篇：视频属性渐变生成（Attribute Transition）

## ## 5.1 问题：Attribute Transition

**难点**：
 视频中的属性变化往往是一个需要多帧呈现的渐变过程（如老人变年轻、灯光变亮）。

当前视频生成模型缺乏自然的、平滑的属性过渡能力。

------

# #6 目标 & 贡献

### **目标**

在无需额外训练的前提下，使模型能生成 **平滑、渐进的属性变化视频**。

### **贡献**

1. 提出一种简单有效的方法，让视频呈现平滑的属性渐变；
2. 构建新的 benchmark：
    **CAT-Bench（Controlled Attribute Transition Benchmark）**
3. 提出两个新指标：
   - **Wholistic Transition Score**
   - **Frame-wise Transition Score**

------

# #7 方法（Method）

## ## 7.1 Transitional Direction Crafting

为不同属性构建“过渡方向”，指导视频在每一帧逐步变化。

## ## 7.2 Neutral Prompt Anchoring

使用**中性 prompt** 固定生成稳定度，使得属性渐变不至于破坏主体。

## ## 7.3 Overall Sampling Flow

方法流程：

1. 构建 transitional direction
2. 应用 neutral prompt anchoring
3. 对每一帧进行属性引导

------

# #8 CAT-Bench 数据集

定义 8 类属性变化：

### **Human Attributes（4类）**

- Age（变年轻/变老）
- Beard（胡子生长）
- Makeup（妆容变换）
- Hair（头发变化）

### **Subject Attributes（2类）**

- Color
- Material

### **Background Attributes（2类）**

- Light condition
- Weather condition

------

# #9 评价指标（Metrics）

1. **Wholistic Transition Score**
   - 衡量整体过渡的自然程度
2. **Frame-wise Transition Score**
   - 衡量邻近帧之间属性变化的平滑性

------

# #10 实验结果（Experiments）

### Base model:

- **VideoCrafter2**

### Baselines:

#### 单 prompt（文本到视频基础模型）：

- AnimateDiff
- ModelScope
- Latte
- VideoCrafter2

#### 多 prompt：

- FreeBloom
- GenL
- Freenoise
- VideoTetris

### 结果：

- CAT-Bench 多项指标优于现有 baseline
- 过渡更自然、更连续
- Neutral Prompt Anchoring + Transitional Direction 效果最佳

------

# #11 Ablation Study

比较：

- **Neutral Prompt**（保持视频主体稳定）
   vs
- **Full Prompt**（可能导致不稳定）

结果：
 中性 prompt 有利于平滑过渡、不破坏主体内容。

------

# #12 总结

- PostDiff 提供了一种训练后、高效的 diffusion 加速框架
- Attribute Transition 方法在无训练条件下实现自然属性渐变
- CAT-Bench 为评估视频属性过渡提供新标准
- 实验结果在多个模型和 baseline 上表现优越

