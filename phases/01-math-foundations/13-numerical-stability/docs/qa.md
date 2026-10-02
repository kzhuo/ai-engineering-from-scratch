# 数值稳定性问答

## 问题：如何推导

\[
\log\left(\sum_i \exp(x_i)\right)
= \max_i(x_i) + \log\left(\sum_i \exp\left(x_i-\max_i(x_i)\right)\right)？
\]

## 解答

设

\[
m=\max_i(x_i)。
\]

关键是先对每个指数项“加上并减去”同一个数 \(m\)：

\[
x_i=(x_i-m)+m。
\]

因此，根据指数函数的加法规则 \(\exp(a+b)=\exp(a)\exp(b)\)，有

\[
\exp(x_i)
=\exp((x_i-m)+m)
=\exp(x_i-m)\exp(m)。
\]

把这个结果代入原来的求和：

\[
\sum_i\exp(x_i)
=\sum_i\left[\exp(x_i-m)\exp(m)\right]。
\]

因为 \(\exp(m)\) 与索引 \(i\) 无关，可以从求和中提出来：

\[
\sum_i\exp(x_i)
=\exp(m)\sum_i\exp(x_i-m)。
\]

两边取对数，并使用 \(\log(ab)=\log(a)+\log(b)\)：

\[
\begin{aligned}
\log\left(\sum_i\exp(x_i)\right)
&=\log\left(\exp(m)\sum_i\exp(x_i-m)\right)\\
&=\log(\exp(m))
  +\log\left(\sum_i\exp(x_i-m)\right)\\
&=m+\log\left(\sum_i\exp(x_i-m)\right)。
\end{aligned}
\]

最后代回 \(m=\max_i(x_i)\)，得到

\[
\boxed{
\log\left(\sum_i\exp(x_i)\right)
=\max_i(x_i)
+\log\left(\sum_i\exp\left(x_i-\max_i(x_i)\right)\right)
}
\]

## 为什么要取最大值？

对所有 \(i\)，都有 \(x_i\le m\)，所以

\[
x_i-m\le 0。
\]

于是

\[
0<\exp(x_i-m)\le 1。
\]

至少有一个 \(x_i=m\)，对应的项满足

\[
\exp(x_i-m)=\exp(0)=1。
\]

因此，平移后的指数项既不会很大到溢出，也至少有一个项等于 1，求和结果不会因为全部下溢而变成 0。这就是数值稳定性来自哪里。

更一般地，对任意有限常数 \(c\)，恒等式

\[
\log\left(\sum_i\exp(x_i)\right)
=c+\log\left(\sum_i\exp(x_i-c)\right)
\]

都成立；选择 \(c=\max_i(x_i)\) 只是让所有指数的输入都不为正，从而最有效地避免溢出。

## 数值例子

令

\[
x=[1000,1001,1002],\qquad m=1002。
\]

直接计算 \(\exp(1000),\exp(1001),\exp(1002)\) 很容易溢出。使用恒等式后：

\[
\begin{aligned}
\log(\exp(1000)+\exp(1001)+\exp(1002))
&=1002+\log(\exp(-2)+\exp(-1)+\exp(0))\\
&=1002+\log(0.1353+0.3679+1)\\
&\approx1002.4076。
\end{aligned}
\]

这里实际参与指数运算的数只有 \(-2,-1,0\)，所以计算安全；结果仍然与原表达式完全相等（忽略浮点舍入误差）。

## 与 softmax 的关系

softmax 的分母正是 \(\sum_j\exp(x_j)\)。将每个分子和分母同时除以 \(\exp(m)\)，得到

\[
\frac{\exp(x_i)}{\sum_j\exp(x_j)}
=\frac{\exp(x_i-m)}{\sum_j\exp(x_j-m)}。
\]

所以，softmax 的“减最大值”与 log-sum-exp 的恒等变形是同一个数学事实的两种用法：先提出公共因子 \(\exp(m)\)，再在安全范围内计算。

## 问题：如何理解这些 NaN/Inf 预防策略？

训练过程中出现 `nan` 或 `inf`，通常不是某一条公式突然失效，而是中间值超出了浮点数能够表示的范围，或者进行了除零、对负数取对数等非法运算。下面逐条说明常用的预防策略。

### 1. 在进入 `exp()` 前限制输入范围

写作：

```python
safe_value = exp(clamp(x, -80, 80))
```

其中

\[
\operatorname{clamp}(x,-80,80)=
\begin{cases}
-80,&x<-80\\
x,&-80\le x\le80\\
80,&x>80
\end{cases}
\]

原因是指数函数增长极快。以 float32 为例，`exp(88.7)` 已经接近最大可表示数，`exp(89)` 通常就会溢出为 `inf`。把输入限制在 `[-80, 80]` 后，`exp(x)` 不会因为正向指数过大而溢出；当输入小于 `-80` 时，结果本来就非常接近 0，截断通常只会带来很小的近似误差。

但这是一种**保护性近似**，不是恒等变形。若模型确实需要区分 `x=100` 和 `x=1000`，直接 clamp 会把它们都变成 80；标准自动微分中，区间外的 clamp 梯度通常为 0，可能阻断学习。因此优先使用 log-sum-exp、`log_softmax` 等结构稳定的公式；只有在应用允许这种截断时才 clamp。`[-80, 80]` 是针对 float32 的示例，float16 的 `exp(80)` 仍会溢出。对很小的指数项，下界截断的绝对误差可能小，但相对误差可能很大，还会抬高概率尾部。

### 2. 在分母中加入 epsilon

写作：

```python
result = x / (y + 1e-8)
```

如果 \(y=0\)，直接计算 \(x/y\) 会产生 `inf`，而 `0/0` 会产生 `nan`。加入一个很小的正数 \(\varepsilon\)：

\[
\frac{x}{y}\quad\longrightarrow\quad\frac{x}{y+\varepsilon}
\]

可以避免分母恰好为零。例如归一化中的标准差可能为零：

\[
\frac{x-\mu}{\sigma+\varepsilon}。
\]

需要注意，epsilon 会改变原来的数学结果，尤其是当 \(y\) 本身也很小时。因此它不是越大越安全；应根据数据尺度和计算精度选择，例如 `1e-5` 或 `1e-8`。在 float16 中，`1e-8` 可能直接舍入为 0，保护就会失效；必要时在 float32 中计算。即使分母不为零，商也可能大到溢出。如果分母可能为负数，简单地加 epsilon 也不一定能避免接近零的问题，例如 \(y=-\varepsilon\) 时分母恰好为 0，此时应重新检查公式或使用带符号的稳定处理。LayerNorm 通常使用 \(\sqrt{\operatorname{Var}(x)+\varepsilon}\) 作为分母，与 \(\sigma+\varepsilon\) 并非同一个公式，应按算法定义选择位置。

### 3. 在 `log()` 内部加入 epsilon

写作：

```python
safe_log = log(x + 1e-8)
```

因为

\[
\log(0)=-\infty，\qquad \log(x)\text{ 在 }x<0\text{ 时没有实数结果}。
\]

概率、方差或激活值经过浮点运算后可能变成 0。加入 epsilon 后，输入至少为一个很小的正数：

\[
\log(x)\quad\longrightarrow\quad\log(x+\varepsilon)。
\]

例如交叉熵中常见的写法是 `-log(probability + eps)`。

它同样会引入偏差，而且只适用于理论上应当满足 \(x\ge0\) 的量。如果 \(x<0\) 是由真正的逻辑错误造成的，加入 epsilon 并不能修复问题，反而可能掩盖错误。对 softmax 和交叉熵，通常应直接使用稳定的 `log_softmax` 或 log-sum-exp，而不是事后给概率加 epsilon。

### 4. 使用结构上稳定的实现

最可靠的做法是改写公式，避免危险的中间值。

对于 log-sum-exp：

\[
\log\sum_i e^{x_i}
=m+\log\sum_i e^{x_i-m},\qquad m=\max_i x_i。
\]

对于 softmax：

\[
\operatorname{softmax}(x_i)
=\frac{e^{x_i-m}}{\sum_j e^{x_j-m}}。
\]

这种改写在数学上保持等价（只考虑实数运算时），同时让指数输入不大于 0，避免 `exp()` 溢出。框架中的 `logsumexp`、`log_softmax`、交叉熵融合算子通常已经实现了这些稳定处理。

这比“先计算一个已经溢出的结果，再用 epsilon 补救”更可靠，因为一旦中间结果已经是 `inf`，后面可能出现 `inf-inf=nan`，信息已经无法恢复。

### 5. 梯度裁剪，防止权重爆炸

如果梯度范数突然变得很大，一次参数更新可能把权重推到极端范围，下一次前向计算就可能溢出。按范数裁剪的典型做法是：

\[
g'=
\begin{cases}
g,&\lVert g\rVert\le t\\
g\dfrac{t}{\lVert g\rVert},&\lVert g\rVert>t
\end{cases}
\]

其中 \(t\) 是最大范数。代码形式：

```python
norm = sqrt(sum(g * g for g in gradients))
if norm > max_norm:
    gradients = [g * max_norm / norm for g in gradients]
```

按范数裁剪会整体乘以一个正数，因此在实数运算中保持梯度方向；按元素裁剪则是分别把每个值限制在区间内，可能改变方向。这里的代码只展示原理：平方求和本身也可能溢出，实际实现应使用稳定的范数计算。梯度已经是 `nan` 或 `inf` 时，裁剪不能恢复它；应先检查有限性。梯度裁剪不能修复错误的梯度公式，也不能替代合适的学习率，它主要用于限制偶发的梯度爆炸。裁剪应在反向传播之后、参数更新之前执行；使用混合精度 loss scaling 时，应先还原梯度尺度再裁剪。

### 6. 调试时检查每次前向传播

应尽早定位第一个出现非有限值的张量，而不是等训练完全失效后再检查最终 loss：

```python
import math

def check_finite(name, values):
    for index, value in enumerate(values):
        if not math.isfinite(value):
            raise FloatingPointError(
                f"{name}[{index}] is not finite: {value}"
            )
```

检查的重点是：输入、线性层输出、激活输出、softmax/logits、loss，以及反向传播后的梯度。`isfinite(x)` 同时排除 `nan`、`+inf` 和 `-inf`。

调试时在每层检查，可以回答“第一个坏值在哪里产生”；只检查最终 loss，只能知道“已经坏了”。定位到具体层后，再根据操作判断原因：`exp` 过大、`log(0)`、除零、负数开方，还是学习率过大导致权重发散。

## 如何组合这些策略？

一个合理的优先顺序是：

1. 先使用数学上稳定的公式，例如稳定 softmax 和 log-sum-exp。
2. 对确实可能为零的分母或对数输入加入合适的 epsilon。
3. 对梯度使用按范数裁剪，限制异常更新。
4. 在调试阶段检查每次前向传播和梯度中的非有限值。
5. 最后才把 clamp 作为边界保护，避免极端输入直接让 `exp()` 溢出。

这些措施解决的是不同层次的问题：稳定公式减少危险中间值，epsilon 防止定义域边界问题，梯度裁剪控制参数更新，有限值检查负责定位故障，clamp 则提供最后一道数值防线。

## 问题：`log_softmax` 和 stable softmax 有什么不同？

### 1. 它们输出的对象不同

Stable softmax 输出概率：

\[
\operatorname{softmax}(x_i)
=\frac{\exp(x_i-m)}{\sum_j\exp(x_j-m)},
\qquad m=\max_j x_j。
\]

输出满足：

\[
p_i\in(0,1),\qquad \sum_i p_i=1。
\]

因此它适合需要概率的场景，例如显示分类概率、采样类别或计算概率加权结果。

`log_softmax` 输出的是概率的自然对数：

\[
\operatorname{log\_softmax}(x_i)
=\log\left(\operatorname{softmax}(x_i)\right)。
\]

展开后得到：

\[
\operatorname{log\_softmax}(x_i)
=x_i-\log\left(\sum_j\exp(x_j)\right)。
\]

使用稳定的 log-sum-exp 后，它实际计算为：

\[
\operatorname{log\_softmax}(x_i)
=x_i-m-\log\left(\sum_j\exp(x_j-m)\right)。
\]

它的输出一定不大于 0，因为概率不会大于 1；多个类别的输出经过指数化后才会加和为 1：

\[
\exp(\operatorname{log\_softmax}(x_i))
=\operatorname{softmax}(x_i)。
\]

### 2. 两者的数学关系

先计算 stable softmax，再取对数：

\[
\log\left(
\frac{\exp(x_i-m)}{\sum_j\exp(x_j-m)}
\right)
=x_i-m-\log\left(\sum_j\exp(x_j-m)\right)。
\]

这就是 stable `log_softmax` 的公式。因此在精确实数运算中：

\[
\operatorname{log\_softmax}(x_i)
=\log(\operatorname{stable\_softmax}(x_i))。
\]

但在浮点计算中，最好不要真的先算 softmax 再调用 `log()`，而是直接使用上面的 log-sum-exp 公式。原因是 softmax 的某些概率可能下溢为 0；一旦发生，就会出现：

\[
\log(0)=-\infty。
\]

直接计算 `log_softmax` 可以保留这些极小概率对应的有限对数值，避免不必要的 `-inf`。

### 3. 一个例子

设 logits 为：

\[
x=[2,1,0]。
\]

stable softmax 的结果约为：

\[
[0.6652,0.2447,0.0900]。
\]

对应的 `log_softmax` 为：

\[
[\log(0.6652),\log(0.2447),\log(0.0900)]
\approx[-0.4076,-1.4076,-2.4076]。
\]

对这些 log 概率逐项取指数，会回到 softmax 概率：

\[
[e^{-0.4076},e^{-1.4076},e^{-2.4076}]
\approx[0.6652,0.2447,0.0900]。
\]

### 4. 为什么交叉熵通常使用 `log_softmax`？

对真实类别 (y)，交叉熵为：

\[
\mathcal L=-\log p_y。
\]

如果先计算 softmax：

```python
probabilities = softmax(logits)
loss = -log(probabilities[target])
```

可能遇到概率下溢为 0 的问题。

使用 `log_softmax` 后，直接取真实类别的 log 概率：

```python
log_probabilities = log_softmax(logits)
loss = -log_probabilities[target]
```

这样省去了“先得到概率，再取对数”的不稳定中间步骤。实际框架中的交叉熵通常进一步把 `log_softmax` 和负对数似然损失融合成一个算子，以减少中间张量和舍入误差。

### 5. 梯度是否不同？

两者的输出不同，所以直接对输出求导时梯度形式不同。

Stable softmax 的 Jacobian 为：

\[
\frac{\partial p_i}{\partial x_j}
=p_i(\mathbf 1_{i=j}-p_j)。
\]

如果对某个类别的 log 概率求导：

\[
\frac{\partial \log p_y}{\partial x_j}
=\mathbf 1_{j=y}-p_j。
\]

因此，`log_softmax` 与负对数似然损失组合后，梯度通常是：

\[
\frac{\partial\mathcal L}{\partial x_j}
=p_j-\mathbf 1_{j=y}。
\]

这正是分类训练所需的简洁形式。手动实现时，不要在已经调用交叉熵的情况下再次对 logits 做 softmax，否则可能重复计算或导致梯度含义错误。

### 6. 应该选择哪一个？

| 需求 | 推荐 |
|---|---|
| 需要类别概率 | stable softmax |
| 需要从概率分布采样 | stable softmax，或先对 log 概率取指数 |
| 计算交叉熵 | `log_softmax` + NLL，或框架的 fused cross-entropy |
| 计算序列的 log 概率 | `log_softmax` |
| 比较概率大小但不需要实际概率 | `log_softmax` |
| 需要避免极小概率下溢 | `log_softmax` |

一句话概括：**stable softmax 给你稳定的概率，`log_softmax` 给你稳定的对数概率；后者通常更适合损失函数和概率相乘、相加都发生在对数空间的场景。**

## 问题：什么是 Batch Normalization、Layer Normalization 和 RMS Normalization？

这三种方法都先把激活值调整到比较可控的数值范围，再用可学习参数恢复模型需要的尺度。它们的主要区别不是“有没有归一化”，而是**沿哪些维度计算统计量**，以及是否减去均值。

为便于说明，设一个 batch 的输入张量形状为

\[
X\in\mathbb{R}^{B\times T\times D},
\]

其中 (B) 是 batch size，(T) 可以表示序列长度，(D) 是特征维度或隐藏维度。

### 1. Batch Normalization（批归一化，BatchNorm）

BatchNorm 对**同一个特征通道**，跨 batch 中的样本统计均值和方差。对二维输入 (X\in\mathbb{R}^{B\times D})，第 (d) 个特征的统计量为：

\[
\mu_d=\frac{1}{B}\sum_{b=1}^{B}x_{b,d},
\]

\[
\sigma_d^2=\frac{1}{B}\sum_{b=1}^{B}(x_{b,d}-\mu_d)^2。
\]

归一化并重新缩放：

\[
\hat{x}_{b,d}=\frac{x_{b,d}-\mu_d}{\sqrt{\sigma_d^2+\varepsilon}},
\]

\[
y_{b,d}=\gamma_d\hat{x}_{b,d}+\beta_d。
\]

其中：

- \(\varepsilon\) 防止方差为 0 时除零；
- \(\gamma_d\) 和 \(\beta_d\) 是可学习的缩放和偏移参数；
- 归一化不会永久限制网络表达能力，因为模型可以学习合适的 \(\gamma\) 和 \(\beta\)。

**训练和推理的区别：**训练时使用当前 mini-batch 的均值和方差，同时维护它们的移动平均；推理时通常使用训练期间累积的 running mean 和 running variance，而不是依赖当前输入 batch。

BatchNorm 的优点是卷积网络中效果很好，并且可以让不同 batch 的激活尺度更一致。缺点是它依赖 batch 统计量：batch 太小时统计噪声大；在线推理、变长序列或 batch size 为 1 时不够方便。

### 2. Layer Normalization（层归一化，LayerNorm）

LayerNorm 对**每一个样本自己的特征维度**计算均值和方差，不依赖其他样本。对 (X\in\mathbb{R}^{B\times T\times D})，通常对最后一个维度 (D) 归一化：

\[
\mu_{b,t}=\frac{1}{D}\sum_{d=1}^{D}x_{b,t,d},
\]

\[
\sigma_{b,t}^2=\frac{1}{D}\sum_{d=1}^{D}(x_{b,t,d}-\mu_{b,t})^2。
\]

然后：

\[
\hat{x}_{b,t,d}=\frac{x_{b,t,d}-\mu_{b,t}}{\sqrt{\sigma_{b,t}^2+\varepsilon}},
\]

\[
y_{b,t,d}=\gamma_d\hat{x}_{b,t,d}+\beta_d。
\]

这里每个样本、每个位置都有自己的均值和方差。因此 LayerNorm：

- 不需要 running statistics；
- 训练和推理使用相同的计算方式；
- 不依赖 batch size；
- 特别适合 RNN、Transformer 和小 batch 训练。

Transformer 中常见的 `LayerNorm(hidden_size)`，就是对每个 token 的 hidden vector 做归一化。它不会把不同 token 或不同样本混在一起统计。

### 3. RMS Normalization（均方根归一化，RMSNorm）

RMSNorm 可以看作 LayerNorm 的简化版本：它只计算均方根，不减去均值。

对最后一个维度 (D)：

\[
\operatorname{RMS}(x)=
\sqrt{\frac{1}{D}\sum_{d=1}^{D}x_d^2+\varepsilon}。
\]

归一化为：

\[
\hat{x}_d=\frac{x_d}{\operatorname{RMS}(x)},
\]

再乘以可学习的缩放参数：

\[
y_d=\gamma_d\hat{x}_d。
\]

与 LayerNorm 相比，RMSNorm 不计算均值，也没有可学习的偏移 \(\beta\)。因此它通常计算更少，速度略快；同时仍然能控制向量的整体幅度。

RMSNorm 的名称来自 root mean square（均方根）：

\[
\operatorname{RMS}(x)=\sqrt{\operatorname{mean}(x^2)}。
\]

它不保证归一化后的特征均值为 0。这个差异很重要：LayerNorm 同时控制中心位置和尺度，RMSNorm 主要控制尺度。

### 4. 三者的核心区别

| 方法 | 统计哪些值 | 是否减均值 | 是否依赖 batch | 常见场景 |
|---|---|---:|---:|---|
| BatchNorm | 同一特征跨样本统计 | 是 | 是 | CNN、视觉模型 |
| LayerNorm | 单个样本的特征维度 | 是 | 否 | Transformer、RNN |
| RMSNorm | 单个样本的特征维度 | 否 | 否 | Transformer、大语言模型 |

可以用一句话记忆：

- **BatchNorm：跨样本看同一个特征。**
- **LayerNorm：对一个样本的特征做完整标准化。**
- **RMSNorm：对一个样本的特征只做幅度标准化。**

### 5. 为什么它们也是数值稳定器？

如果网络层层相乘，激活可能逐渐变大：

\[
[0,1]\rightarrow[0,100]\rightarrow[0,10^4]\rightarrow[0,\infty)。
\]

过大的激活会让线性层、平方、指数函数或梯度计算溢出；过小的激活则可能导致梯度下溢或信号逐渐消失。

归一化把输入变成大致受控的尺度：

\[
\frac{x-\mu}{\sqrt{\sigma^2+\varepsilon}}
\quad\text{或}\quad
\frac{x}{\sqrt{\operatorname{mean}(x^2)+\varepsilon}}。
\]

分母中的 \(\varepsilon\) 防止所有值相同时出现除零；归一化后的结果通常不会因为上一层的整体尺度变大而无限变大。这样可以降低：

- 前向传播中的溢出风险；
- 反向传播中的梯度爆炸风险；
- 不同层之间激活尺度差异过大的问题。

归一化不能保证永远不会出现 `nan` 或 `inf`。如果输入本身已经是非有限值，或者学习率、权重、损失计算存在其他错误，归一化也无法恢复它；它只是让数值更容易保持在安全范围内。

### 6. 一个直观例子

对向量

\[
x=[1000,1001,999]
\]

LayerNorm 会先减去均值约 1000，再除以标准差，因此得到近似中心为 0、尺度适中的向量。RMSNorm 则用这个向量的均方根作为分母，把整体幅度缩小到约 1，但不会把均值显式移到 0。

如果另一个向量只是它的 100 倍，归一化后两者的幅度仍会接近；这就是它们对数值尺度的稳定作用。

### 7. 实际选择

- 处理卷积图像、batch 足够大：通常考虑 BatchNorm。
- 处理 Transformer 或 batch 很小：通常使用 LayerNorm。
- 需要更轻量的 Transformer 归一化，并且模型架构允许保留均值信息：可以使用 RMSNorm。

最终选择还取决于网络结构、归一化放置位置（Pre-Norm 或 Post-Norm）、硬件和训练配置。归一化层通常放在残差块或主要变换附近，用来控制每次子层计算接收到的数值尺度。
