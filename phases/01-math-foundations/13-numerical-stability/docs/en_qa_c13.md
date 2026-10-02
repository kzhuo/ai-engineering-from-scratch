# Numerical Stability

> Floating point is a leaky abstraction. It will bite you during training, and you will not see it coming.

**Type:** Build
**Language:** Python
**Prerequisites:** Phase 1, Lessons 01-04
**Time:** ~120 minutes

## Learning Objectives

- Implement numerically stable softmax and log-sum-exp using the max-subtraction trick
- Identify overflow, underflow, and catastrophic cancellation in floating-point computations
- Verify analytical gradients against numerical gradients using centered finite differences
- Explain why bfloat16 is preferred over float16 for training and how loss scaling prevents gradient underflow

## The Problem

Your model trains for three hours, then the loss becomes NaN. You add a print statement. The logits are fine at step 9,000. At step 9,001 they are `inf`. By step 9,002 every gradient is `nan` and training is dead.

Or: your model trains to completion but accuracy is 2% worse than the paper claims. You check everything. Architecture matches. Hyperparameters match. Data matches. The problem is that the paper used float32 and you used float16 without the right scaling. Thirty-two bits of accumulated rounding error quietly ate your accuracy.

Or: you implement cross-entropy loss from scratch. It works on small logits. When logits exceed 100, it returns `inf`. The softmax overflowed because `exp(100)` is larger than float32 can represent. Every ML framework handles this with a two-line trick. You did not know the trick existed.

Numerical stability is not a theoretical concern. It is the difference between a training run that succeeds and one that silently fails. Every serious ML bug you will debug eventually comes down to floating point.

## The Concept

### IEEE 754: How Computers Store Real Numbers

Computers store real numbers as floating point values following the IEEE 754 standard. A float has three parts: a sign bit, an exponent, and a mantissa (significand).

```
Float32 layout (32 bits total):
[1 sign] [8 exponent] [23 mantissa]

Value = (-1)^sign * 2^(exponent - 127) * 1.mantissa
```

The mantissa determines precision (how many significant digits). The exponent determines range (how large or small a number can be).

```
Format     Bits   Exponent  Mantissa  Decimal digits  Range (approx)
float64    64     11        52        ~15-16          +/- 1.8e308
float32    32     8         23        ~7-8            +/- 3.4e38
float16    16     5         10        ~3-4            +/- 65,504
bfloat16   16     8         7         ~2-3            +/- 3.4e38
```

float32 gives you about 7 decimal digits of precision. That means it can tell apart 1.0000001 and 1.0000002, but not 1.00000001 and 1.00000002. After 7 digits, everything is rounding noise.

float16 gives you about 3 digits. The largest number it can represent is 65,504. That is disturbingly small for ML where logits, gradients, and activations routinely exceed this.

bfloat16 is Google's answer to float16's range problem. It has the same 8-bit exponent as float32 (same range, up to 3.4e38) but only 7 mantissa bits (less precision than float16). For training neural networks, range matters more than precision, so bfloat16 usually wins.

### Why 0.1 + 0.2 != 0.3

The number 0.1 cannot be represented exactly in binary floating point. In base 2, it is a repeating fraction:

```
0.1 in binary = 0.0001100110011001100110011... (repeating forever)
```

Float32 truncates this to 23 bits of mantissa. The stored value is approximately 0.100000001490116. Similarly, 0.2 is stored as approximately 0.200000002980232. Their sum is 0.300000004470348, not 0.3.

```
In Python:
>>> 0.1 + 0.2
0.30000000000000004

>>> 0.1 + 0.2 == 0.3
False
```

This matters for ML because:

1. Loss comparisons like `if loss < threshold` can give wrong answers
2. Accumulating many small values (gradient updates over thousands of steps) drifts from the true sum
3. Checksums and reproducibility tests fail if you compare floats with `==`

The fix: never compare floats with `==`. Use `abs(a - b) < epsilon` or `math.isclose()`.

### Catastrophic Cancellation

When you subtract two nearly equal floating point numbers, the significant digits cancel and you are left with rounding noise promoted to leading digits.

```
a = 1.0000001    (stored as 1.00000011920929 in float32)
b = 1.0000000    (stored as 1.00000000000000 in float32)

True difference:  0.0000001
Computed:         0.00000011920929

Relative error: 19.2%
```

That is a 19% relative error from a single subtraction. In ML, this happens whenever you:

- Compute variance of data with a large mean: `E[x^2] - E[x]^2` when E[x] is large
- Subtract nearly equal log-probabilities
- Compute finite-difference gradients with too-small epsilon

The fix: rearrange formulas to avoid subtracting large, nearly equal numbers. For variance, use the Welford algorithm or center the data first. For log-probabilities, work in log-space throughout.

### Overflow and Underflow

Overflow happens when a result is too large to represent. Underflow happens when it is too small (closer to zero than the smallest representable positive number).

```
Float32 boundaries:
  Maximum:  3.4028235e+38
  Minimum positive (normal): 1.175e-38
  Minimum positive (denorm): 1.401e-45
  Overflow:  anything > 3.4e38 becomes inf
  Underflow: anything < 1.4e-45 becomes 0.0
```

The `exp()` function is the primary source of overflow in ML:

```
exp(88.7)  = 3.40e+38   (barely fits in float32)
exp(89.0)  = inf         (overflow)
exp(-87.3) = 1.18e-38   (barely above underflow)
exp(-104)  = 0.0         (underflow to zero)
```

The `log()` function hits the other direction:

```
log(0.0)   = -inf
log(-1.0)  = nan
log(1e-45) = -103.3      (fine)
log(1e-46) = -inf        (input underflowed to 0, then log(0) = -inf)
```

In ML, `exp()` appears in softmax, sigmoid, and probability computations. `log()` appears in cross-entropy, log-likelihoods, and KL divergence. The combination `log(exp(x))` is a minefield without the right tricks.

### The Log-Sum-Exp Trick

Computing `log(sum(exp(x_i)))` directly is numerically dangerous. If any `x_i` is large, `exp(x_i)` overflows. If all `x_i` are very negative, every `exp(x_i)` underflows to zero and `log(0)` is `-inf`.

The trick: subtract the maximum value before exponentiating.

```
log(sum(exp(x_i))) = max(x) + log(sum(exp(x_i - max(x))))
```

Why this works: after subtracting `max(x)`, the largest exponent is `exp(0) = 1`. No overflow is possible. At least one term in the sum is 1, so the sum is at least 1, and `log(1) = 0`. No underflow to `-inf` is possible.

Proof:

```
log(sum(exp(x_i)))
= log(sum(exp(x_i - c + c)))                    (add and subtract c)
= log(sum(exp(x_i - c) * exp(c)))               (exp(a+b) = exp(a)*exp(b))
= log(exp(c) * sum(exp(x_i - c)))               (factor out exp(c))
= c + log(sum(exp(x_i - c)))                    (log(a*b) = log(a) + log(b))
```

Set `c = max(x)` and overflow is eliminated.

This trick appears everywhere in ML:
- Softmax normalization
- Cross-entropy loss computation
- Log-probability summation in sequence models
- Mixture of Gaussians
- Variational inference

#### 问题：如何推导

$$
\log\left(\sum_i \exp(x_i)\right)
= \max_i(x_i) + \log\left(\sum_i \exp\left(x_i-\max_i(x_i)\right)\right)？
$$

#### 解答

设

$$
m=\max_i(x_i)。
$$

关键是先对每个指数项“加上并减去”同一个数 $m$：

$$
x_i=(x_i-m)+m。
$$

因此，根据指数函数的加法规则 $\exp(a+b)=\exp(a)\exp(b)$，有

$$
\exp(x_i)
=\exp((x_i-m)+m)
=\exp(x_i-m)\exp(m)。
$$

把这个结果代入原来的求和：

$$
\sum_i\exp(x_i)
=\sum_i\left[\exp(x_i-m)\exp(m)\right]。
$$

因为 $\exp(m)$ 与索引 $i$ 无关，可以从求和中提出来：

$$
\sum_i\exp(x_i)
=\exp(m)\sum_i\exp(x_i-m)。
$$

两边取对数，并使用 $\log(ab)=\log(a)+\log(b)$：

$$
\begin{aligned}
\log\left(\sum_i\exp(x_i)\right)
&=\log\left(\exp(m)\sum_i\exp(x_i-m)\right)\\
&=\log(\exp(m))
  +\log\left(\sum_i\exp(x_i-m)\right)\\
&=m+\log\left(\sum_i\exp(x_i-m)\right)。
\end{aligned}
$$

最后代回 $m=\max_i(x_i)$，得到

$$
\boxed{
\log\left(\sum_i\exp(x_i)\right)
=\max_i(x_i)
+\log\left(\sum_i\exp\left(x_i-\max_i(x_i)\right)\right)
}
$$

#### 为什么要取最大值？

对所有 $i$，都有 $x_i\le m$，所以

$$
x_i-m\le 0。
$$

于是

$$
0<\exp(x_i-m)\le 1。
$$

至少有一个 $x_i=m$，对应的项满足

$$
\exp(x_i-m)=\exp(0)=1。
$$

因此，平移后的指数项既不会很大到溢出，也至少有一个项等于 1，求和结果不会因为全部下溢而变成 0。这就是数值稳定性来自哪里。

更一般地，对任意有限常数 $c$，恒等式

$$
\log\left(\sum_i\exp(x_i)\right)
=c+\log\left(\sum_i\exp(x_i-c)\right)
$$

都成立；选择 $c=\max_i(x_i)$ 只是让所有指数的输入都不为正，从而最有效地避免溢出。

#### 数值例子

令

$$
x=[1000,1001,1002],\qquad m=1002。
$$

直接计算 $\exp(1000),\exp(1001),\exp(1002)$ 很容易溢出。使用恒等式后：

$$
\begin{aligned}
\log(\exp(1000)+\exp(1001)+\exp(1002))
&=1002+\log(\exp(-2)+\exp(-1)+\exp(0))\\
&=1002+\log(0.1353+0.3679+1)\\
&\approx1002.4076。
\end{aligned}
$$

这里实际参与指数运算的数只有 $-2,-1,0$，所以计算安全；结果仍然与原表达式完全相等（忽略浮点舍入误差）。

#### 与 softmax 的关系

softmax 的分母正是 $\sum_j\exp(x_j)$。将每个分子和分母同时除以 $\exp(m)$，得到

$$
\frac{\exp(x_i)}{\sum_j\exp(x_j)}
=\frac{\exp(x_i-m)}{\sum_j\exp(x_j-m)}。
$$

所以，softmax 的“减最大值”与 log-sum-exp 的恒等变形是同一个数学事实的两种用法：先提出公共因子 $\exp(m)$，再在安全范围内计算。

### Why Softmax Needs the Max-Subtraction Trick

Softmax converts logits to probabilities:

```
softmax(x_i) = exp(x_i) / sum(exp(x_j))
```

Without the trick, logits of [100, 101, 102] cause overflow:

```
exp(100) = 2.69e43
exp(101) = 7.31e43
exp(102) = 1.99e44
sum      = 2.99e44

These overflow float32 (max ~3.4e38)? No, 2.69e43 < 3.4e38? Actually:
exp(88.7) is already at the float32 limit.
exp(100) = inf in float32.
```

With the trick, subtract max(x) = 102:

```
exp(100 - 102) = exp(-2) = 0.135
exp(101 - 102) = exp(-1) = 0.368
exp(102 - 102) = exp(0)  = 1.000
sum = 1.503

softmax = [0.090, 0.245, 0.665]
```

The probabilities are identical. The computation is safe. This is not an optimization. It is a requirement for correctness.

#### 问题：`log_softmax` 和 stable softmax 有什么不同？

##### 1. 它们输出的对象不同

Stable softmax 输出概率：

$$
\operatorname{softmax}(x_i)
=\frac{\exp(x_i-m)}{\sum_j\exp(x_j-m)},
\qquad m=\max_j x_j。
$$

输出满足：

$$
p_i\in(0,1),\qquad \sum_i p_i=1。
$$

因此它适合需要概率的场景，例如显示分类概率、采样类别或计算概率加权结果。

`log_softmax` 输出的是概率的自然对数：

$$
\operatorname{log\_softmax}(x_i)
=\log\left(\operatorname{softmax}(x_i)\right)。
$$

展开后得到：

$$
\operatorname{log\_softmax}(x_i)
=x_i-\log\left(\sum_j\exp(x_j)\right)。
$$

使用稳定的 log-sum-exp 后，它实际计算为：

$$
\operatorname{log\_softmax}(x_i)
=x_i-m-\log\left(\sum_j\exp(x_j-m)\right)。
$$

它的输出一定不大于 0，因为概率不会大于 1；多个类别的输出经过指数化后才会加和为 1：

$$
\exp(\operatorname{log\_softmax}(x_i))
=\operatorname{softmax}(x_i)。
$$

##### 2. 两者的数学关系

先计算 stable softmax，再取对数：

$$
\log\left(
\frac{\exp(x_i-m)}{\sum_j\exp(x_j-m)}
\right)
=x_i-m-\log\left(\sum_j\exp(x_j-m)\right)。
$$

这就是 stable `log_softmax` 的公式。因此在精确实数运算中：

$$
\operatorname{log\_softmax}(x_i)
=\log(\operatorname{stable\_softmax}(x_i))。
$$

但在浮点计算中，最好不要真的先算 softmax 再调用 `log()`，而是直接使用上面的 log-sum-exp 公式。原因是 softmax 的某些概率可能下溢为 0；一旦发生，就会出现：

$$
\log(0)=-\infty。
$$

直接计算 `log_softmax` 可以保留这些极小概率对应的有限对数值，避免不必要的 `-inf`。

##### 3. 一个例子

设 logits 为：

$$
x=[2,1,0]。
$$

stable softmax 的结果约为：

$$
[0.6652,0.2447,0.0900]。
$$

对应的 `log_softmax` 为：

$$
[\log(0.6652),\log(0.2447),\log(0.0900)]
\approx[-0.4076,-1.4076,-2.4076]。
$$

对这些 log 概率逐项取指数，会回到 softmax 概率：

$$
[e^{-0.4076},e^{-1.4076},e^{-2.4076}]
\approx[0.6652,0.2447,0.0900]。
$$

##### 4. 为什么交叉熵通常使用 `log_softmax`？

对真实类别 $y$，交叉熵为：

$$
\mathcal L=-\log p_y。
$$

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

##### 5. 梯度是否不同？

两者的输出不同，所以直接对输出求导时梯度形式不同。

Stable softmax 的 Jacobian 为：

$$
\frac{\partial p_i}{\partial x_j}
=p_i(\mathbf 1_{i=j}-p_j)。
$$

如果对某个类别的 log 概率求导：

$$
\frac{\partial \log p_y}{\partial x_j}
=\mathbf 1_{j=y}-p_j。
$$

因此，`log_softmax` 与负对数似然损失组合后，梯度通常是：

$$
\frac{\partial\mathcal L}{\partial x_j}
=p_j-\mathbf 1_{j=y}。
$$

这正是分类训练所需的简洁形式。手动实现时，不要在已经调用交叉熵的情况下再次对 logits 做 softmax，否则可能重复计算或导致梯度含义错误。

##### 6. 应该选择哪一个？

| 需求 | 推荐 |
|---|---|
| 需要类别概率 | stable softmax |
| 需要从概率分布采样 | stable softmax，或先对 log 概率取指数 |
| 计算交叉熵 | `log_softmax` + NLL，或框架的 fused cross-entropy |
| 计算序列的 log 概率 | `log_softmax` |
| 比较概率大小但不需要实际概率 | `log_softmax` |
| 需要避免极小概率下溢 | `log_softmax` |

一句话概括：**stable softmax 给你稳定的概率，`log_softmax` 给你稳定的对数概率；后者通常更适合损失函数和概率相乘、相加都发生在对数空间的场景。**

### NaN and Inf: Detection and Prevention

`nan` (Not a Number) and `inf` (infinity) propagate virally through computation. One `nan` in a gradient update makes the weight `nan`, which makes every subsequent output `nan`. Training is dead within one step.

How `inf` appears:
- `exp()` of a large positive number
- Division by zero: `1.0 / 0.0`
- `float32` overflow in accumulations

How `nan` appears:
- `0.0 / 0.0`
- `inf - inf`
- `inf * 0`
- `sqrt()` of a negative number
- `log()` of a negative number
- Any arithmetic involving an existing `nan`

Detection:

```python
import math

math.isnan(x)       # True if x is nan
math.isinf(x)       # True if x is +inf or -inf
math.isfinite(x)    # True if x is neither nan nor inf
```

Prevention strategies:

1. Clamp inputs to `exp()`: `exp(clamp(x, -80, 80))`
2. Add epsilon to denominators: `x / (y + 1e-8)`
3. Add epsilon inside `log()`: `log(x + 1e-8)`
4. Use stable implementations (log-sum-exp, stable softmax)
5. Gradient clipping to prevent weight explosion
6. Check for `nan`/`inf` after every forward pass during debugging

#### 问题：如何理解这些 NaN/Inf 预防策略？

训练过程中出现 `nan` 或 `inf`，通常不是某一条公式突然失效，而是中间值超出了浮点数能够表示的范围，或者进行了除零、对负数取对数等非法运算。下面逐条说明常用的预防策略。

##### 1. 在进入 `exp()` 前限制输入范围

写作：

```python
safe_value = exp(clamp(x, -80, 80))
```

其中

$$
\operatorname{clamp}(x,-80,80)=
\begin{cases}
-80,&x<-80\\
x,&-80\le x\le80\\
80,&x>80
\end{cases}
$$

原因是指数函数增长极快。以 float32 为例，`exp(88.7)` 已经接近最大可表示数，`exp(89)` 通常就会溢出为 `inf`。把输入限制在 `[-80, 80]` 后，`exp(x)` 不会因为正向指数过大而溢出；当输入小于 `-80` 时，结果本来就非常接近 0，截断通常只会带来很小的近似误差。

但这是一种**保护性近似**，不是恒等变形。若模型确实需要区分 `x=100` 和 `x=1000`，直接 clamp 会把它们都变成 80；标准自动微分中，区间外的 clamp 梯度通常为 0，可能阻断学习。因此优先使用 log-sum-exp、`log_softmax` 等结构稳定的公式；只有在应用允许这种截断时才 clamp。`[-80, 80]` 是针对 float32 的示例，float16 的 `exp(80)` 仍会溢出。对很小的指数项，下界截断的绝对误差可能小，但相对误差可能很大，还会抬高概率尾部。

##### 2. 在分母中加入 epsilon

写作：

```python
result = x / (y + 1e-8)
```

如果 $y=0$，直接计算 $x/y$ 会产生 `inf`，而 `0/0` 会产生 `nan`。加入一个很小的正数 $\varepsilon$：

$$
\frac{x}{y}\quad\longrightarrow\quad\frac{x}{y+\varepsilon}
$$

可以避免分母恰好为零。例如归一化中的标准差可能为零：

$$
\frac{x-\mu}{\sigma+\varepsilon}。
$$

需要注意，epsilon 会改变原来的数学结果，尤其是当 $y$ 本身也很小时。因此它不是越大越安全；应根据数据尺度和计算精度选择，例如 `1e-5` 或 `1e-8`。在 float16 中，`1e-8` 可能直接舍入为 0，保护就会失效；必要时在 float32 中计算。即使分母不为零，商也可能大到溢出。如果分母可能为负数，简单地加 epsilon 也不一定能避免接近零的问题，例如 $y=-\varepsilon$ 时分母恰好为 0，此时应重新检查公式或使用带符号的稳定处理。LayerNorm 通常使用 $\sqrt{\operatorname{Var}(x)+\varepsilon}$ 作为分母，与 $\sigma+\varepsilon$ 并非同一个公式，应按算法定义选择位置。

##### 3. 在 `log()` 内部加入 epsilon

写作：

```python
safe_log = log(x + 1e-8)
```

因为

$$
\log(0)=-\infty，\qquad \log(x)\text{ 在 }x<0\text{ 时没有实数结果}。
$$

概率、方差或激活值经过浮点运算后可能变成 0。加入 epsilon 后，输入至少为一个很小的正数：

$$
\log(x)\quad\longrightarrow\quad\log(x+\varepsilon)。
$$

例如交叉熵中常见的写法是 `-log(probability + eps)`。

它同样会引入偏差，而且只适用于理论上应当满足 $x\ge0$ 的量。如果 $x<0$ 是由真正的逻辑错误造成的，加入 epsilon 并不能修复问题，反而可能掩盖错误。对 softmax 和交叉熵，通常应直接使用稳定的 `log_softmax` 或 log-sum-exp，而不是事后给概率加 epsilon。

##### 4. 使用结构上稳定的实现

最可靠的做法是改写公式，避免危险的中间值。

对于 log-sum-exp：

$$
\log\sum_i e^{x_i}
=m+\log\sum_i e^{x_i-m},\qquad m=\max_i x_i。
$$

对于 softmax：

$$
\operatorname{softmax}(x_i)
=\frac{e^{x_i-m}}{\sum_j e^{x_j-m}}。
$$

这种改写在数学上保持等价（只考虑实数运算时），同时让指数输入不大于 0，避免 `exp()` 溢出。框架中的 `logsumexp`、`log_softmax`、交叉熵融合算子通常已经实现了这些稳定处理。

这比“先计算一个已经溢出的结果，再用 epsilon 补救”更可靠，因为一旦中间结果已经是 `inf`，后面可能出现 `inf-inf=nan`，信息已经无法恢复。

##### 5. 梯度裁剪，防止权重爆炸

如果梯度范数突然变得很大，一次参数更新可能把权重推到极端范围，下一次前向计算就可能溢出。按范数裁剪的典型做法是：

$$
g'=
\begin{cases}
g,&\lVert g\rVert\le t\\
g\dfrac{t}{\lVert g\rVert},&\lVert g\rVert>t
\end{cases}
$$

其中 $t$ 是最大范数。代码形式：

```python
norm = sqrt(sum(g * g for g in gradients))
if norm > max_norm:
    gradients = [g * max_norm / norm for g in gradients]
```

按范数裁剪会整体乘以一个正数，因此在实数运算中保持梯度方向；按元素裁剪则是分别把每个值限制在区间内，可能改变方向。这里的代码只展示原理：平方求和本身也可能溢出，实际实现应使用稳定的范数计算。梯度已经是 `nan` 或 `inf` 时，裁剪不能恢复它；应先检查有限性。梯度裁剪不能修复错误的梯度公式，也不能替代合适的学习率，它主要用于限制偶发的梯度爆炸。裁剪应在反向传播之后、参数更新之前执行；使用混合精度 loss scaling 时，应先还原梯度尺度再裁剪。

##### 6. 调试时检查每次前向传播

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

#### 如何组合这些策略？

一个合理的优先顺序是：

1. 先使用数学上稳定的公式，例如稳定 softmax 和 log-sum-exp。
2. 对确实可能为零的分母或对数输入加入合适的 epsilon。
3. 对梯度使用按范数裁剪，限制异常更新。
4. 在调试阶段检查每次前向传播和梯度中的非有限值。
5. 最后才把 clamp 作为边界保护，避免极端输入直接让 `exp()` 溢出。

这些措施解决的是不同层次的问题：稳定公式减少危险中间值，epsilon 防止定义域边界问题，梯度裁剪控制参数更新，有限值检查负责定位故障，clamp 则提供最后一道数值防线。

### Numerical Gradient Checking

Analytical gradients (from backpropagation) can have bugs. Numerical gradient checking verifies them by computing gradients with finite differences.

The centered difference formula:

```
df/dx ~= (f(x + h) - f(x - h)) / (2h)
```

This is O(h^2) accurate, much better than the forward difference `(f(x+h) - f(x)) / h` which is only O(h).

Choosing h: too large and the approximation is wrong. Too small and catastrophic cancellation destroys the answer. `h = 1e-5` to `1e-7` is typical.

The check: compute the relative difference between analytical and numerical gradients.

```
relative_error = |grad_analytical - grad_numerical| / max(|grad_analytical|, |grad_numerical|, 1e-8)
```

Rules of thumb:
- relative_error < 1e-7: perfect, gradient is correct
- relative_error < 1e-5: acceptable, probably correct
- relative_error > 1e-3: something is wrong
- relative_error > 1: gradient is completely wrong

Always check gradients when implementing a new layer or loss function. PyTorch provides `torch.autograd.gradcheck()` for this.

### Mixed Precision Training

Modern GPUs have specialized hardware (Tensor Cores) that compute float16 matrix multiplications 2-8x faster than float32. Mixed precision training exploits this:

```
1. Maintain float32 master copy of weights
2. Forward pass in float16 (fast)
3. Compute loss in float32 (prevents overflow)
4. Backward pass in float16 (fast)
5. Scale gradients to float32
6. Update float32 master weights
```

The problem with pure float16 training: gradients are often very small (1e-8 or smaller). Float16 underflows anything below ~6e-8 to zero. Your model stops learning because all gradient updates are zero.

The fix is loss scaling:

```
1. Multiply loss by a large scale factor (e.g., 1024)
2. Backward pass computes gradients of (loss * 1024)
3. All gradients are 1024x larger (pushed above float16 underflow)
4. Divide gradients by 1024 before updating weights
5. Net effect: same update, but no underflow
```

Dynamic loss scaling adjusts the scale factor automatically. Start with a large value (65536). If gradients overflow to `inf`, halve it. If N steps pass without overflow, double it.

### bfloat16 vs float16: Why bfloat16 Wins for Training

```
float16:   [1 sign] [5 exponent]  [10 mantissa]
bfloat16:  [1 sign] [8 exponent]  [7 mantissa]
```

float16 has more precision (10 mantissa bits vs 7) but limited range (max ~65,504). bfloat16 has less precision but the same range as float32 (max ~3.4e38).

For training neural networks:

- Activations and logits regularly exceed 65,504 during training spikes. float16 overflows; bfloat16 handles it.
- Loss scaling is required with float16 but usually unnecessary with bfloat16 because its range covers the gradient magnitude spectrum.
- bfloat16 is a simple truncation of float32: drop the bottom 16 bits of the mantissa. Conversion is trivial and lossless in the exponent.

float16 is preferred for inference where values are bounded and precision matters more. bfloat16 is preferred for training where range matters more. This is why TPUs and modern NVIDIA GPUs (A100, H100) have native bfloat16 support.

### Gradient Clipping

Exploding gradients happen when gradients grow exponentially through many layers (common in RNNs, deep networks, and transformers). A single large gradient can corrupt all weights in one step.

Two types of clipping:

**Clip by value:** clamp each gradient element independently.

```
grad = clamp(grad, -max_val, max_val)
```

Simple but can change the direction of the gradient vector.

**Clip by norm:** scale the entire gradient vector so its norm does not exceed a threshold.

```
if ||grad|| > max_norm:
    grad = grad * (max_norm / ||grad||)
```

Preserves the direction of the gradient. This is what `torch.nn.utils.clip_grad_norm_()` does. It is the standard choice.

Typical values: `max_norm=1.0` for transformers, `max_norm=0.5` for RL, `max_norm=5.0` for simpler networks.

Gradient clipping is not a hack. It is a safety mechanism. Without it, a single outlier batch can produce a gradient large enough to ruin weeks of training.

### Normalization Layers as Numerical Stabilizers

Batch normalization, layer normalization, and RMS normalization are usually presented as regularizers that help training converge. They are also numerical stabilizers.

Without normalization, activations can grow or shrink exponentially through layers:

```
Layer 1: values in [0, 1]
Layer 5: values in [0, 100]
Layer 10: values in [0, 10,000]
Layer 50: values in [0, inf]
```

Normalization recenters and rescales activations at every layer:

```
LayerNorm(x) = (x - mean(x)) / (std(x) + epsilon) * gamma + beta
```

The `epsilon` (typically 1e-5) prevents division by zero when all activations are identical. The learned parameters `gamma` and `beta` let the network restore any scale it needs.

This keeps values in a numerically safe range throughout the network, preventing both overflow in the forward pass and gradient explosion in the backward pass.

#### 问题：什么是 Batch Normalization、Layer Normalization 和 RMS Normalization？

这三种方法都先把激活值调整到比较可控的数值范围，再用可学习参数恢复模型需要的尺度。它们的主要区别不是“有没有归一化”，而是**沿哪些维度计算统计量**，以及是否减去均值。

为便于说明，设一个 batch 的输入张量形状为

$$
X\in\mathbb{R}^{B\times T\times D},
$$

其中 $B$ 是 batch size，$T$ 可以表示序列长度，$D$ 是特征维度或隐藏维度。

##### 1. Batch Normalization（批归一化，BatchNorm）

BatchNorm 对**同一个特征通道**，跨 batch 中的样本统计均值和方差。对二维输入 $X\in\mathbb{R}^{B\times D}$，第 $d$ 个特征的统计量为：

$$
\mu_d=\frac{1}{B}\sum_{b=1}^{B}x_{b,d},
$$

$$
\sigma_d^2=\frac{1}{B}\sum_{b=1}^{B}(x_{b,d}-\mu_d)^2。
$$

归一化并重新缩放：

$$
\hat{x}_{b,d}=\frac{x_{b,d}-\mu_d}{\sqrt{\sigma_d^2+\varepsilon}},
$$

$$
y_{b,d}=\gamma_d\hat{x}_{b,d}+\beta_d。
$$

其中：

- $\varepsilon$ 防止方差为 0 时除零；
- $\gamma_d$ 和 $\beta_d$ 是可学习的缩放和偏移参数；
- 归一化不会永久限制网络表达能力，因为模型可以学习合适的 $\gamma$ 和 $\beta$。

**训练和推理的区别：**训练时使用当前 mini-batch 的均值和方差，同时维护它们的移动平均；推理时通常使用训练期间累积的 running mean 和 running variance，而不是依赖当前输入 batch。

BatchNorm 的优点是卷积网络中效果很好，并且可以让不同 batch 的激活尺度更一致。缺点是它依赖 batch 统计量：batch 太小时统计噪声大；在线推理、变长序列或 batch size 为 1 时不够方便。

##### 2. Layer Normalization（层归一化，LayerNorm）

LayerNorm 对**每一个样本自己的特征维度**计算均值和方差，不依赖其他样本。对 $X\in\mathbb{R}^{B\times T\times D}$，通常对最后一个维度 $D$ 归一化：

$$
\mu_{b,t}=\frac{1}{D}\sum_{d=1}^{D}x_{b,t,d},
$$

$$
\sigma_{b,t}^2=\frac{1}{D}\sum_{d=1}^{D}(x_{b,t,d}-\mu_{b,t})^2。
$$

然后：

$$
\hat{x}_{b,t,d}=\frac{x_{b,t,d}-\mu_{b,t}}{\sqrt{\sigma_{b,t}^2+\varepsilon}},
$$

$$
y_{b,t,d}=\gamma_d\hat{x}_{b,t,d}+\beta_d。
$$

这里每个样本、每个位置都有自己的均值和方差。因此 LayerNorm：

- 不需要 running statistics；
- 训练和推理使用相同的计算方式；
- 不依赖 batch size；
- 特别适合 RNN、Transformer 和小 batch 训练。

Transformer 中常见的 `LayerNorm(hidden_size)`，就是对每个 token 的 hidden vector 做归一化。它不会把不同 token 或不同样本混在一起统计。

##### 3. RMS Normalization（均方根归一化，RMSNorm）

RMSNorm 可以看作 LayerNorm 的简化版本：它只计算均方根，不减去均值。

对最后一个维度 $D$：

$$
\operatorname{RMS}(x)=
\sqrt{\frac{1}{D}\sum_{d=1}^{D}x_d^2+\varepsilon}。
$$

归一化为：

$$
\hat{x}_d=\frac{x_d}{\operatorname{RMS}(x)},
$$

再乘以可学习的缩放参数：

$$
y_d=\gamma_d\hat{x}_d。
$$

与 LayerNorm 相比，RMSNorm 不计算均值，也没有可学习的偏移 $\beta$。因此它通常计算更少，速度略快；同时仍然能控制向量的整体幅度。

RMSNorm 的名称来自 root mean square（均方根）：

$$
\operatorname{RMS}(x)=\sqrt{\operatorname{mean}(x^2)}。
$$

它不保证归一化后的特征均值为 0。这个差异很重要：LayerNorm 同时控制中心位置和尺度，RMSNorm 主要控制尺度。

##### 4. 三者的核心区别

| 方法 | 统计哪些值 | 是否减均值 | 是否依赖 batch | 常见场景 |
|---|---|---:|---:|---|
| BatchNorm | 同一特征跨样本统计 | 是 | 是 | CNN、视觉模型 |
| LayerNorm | 单个样本的特征维度 | 是 | 否 | Transformer、RNN |
| RMSNorm | 单个样本的特征维度 | 否 | 否 | Transformer、大语言模型 |

可以用一句话记忆：

- **BatchNorm：跨样本看同一个特征。**
- **LayerNorm：对一个样本的特征做完整标准化。**
- **RMSNorm：对一个样本的特征只做幅度标准化。**

##### 5. 为什么它们也是数值稳定器？

如果网络层层相乘，激活可能逐渐变大：

$$
[0,1]\rightarrow[0,100]\rightarrow[0,10^4]\rightarrow[0,\infty)。
$$

过大的激活会让线性层、平方、指数函数或梯度计算溢出；过小的激活则可能导致梯度下溢或信号逐渐消失。

归一化把输入变成大致受控的尺度：

$$
\frac{x-\mu}{\sqrt{\sigma^2+\varepsilon}}
\quad\text{或}\quad
\frac{x}{\sqrt{\operatorname{mean}(x^2)+\varepsilon}}。
$$

分母中的 $\varepsilon$ 防止所有值相同时出现除零；归一化后的结果通常不会因为上一层的整体尺度变大而无限变大。这样可以降低：

- 前向传播中的溢出风险；
- 反向传播中的梯度爆炸风险；
- 不同层之间激活尺度差异过大的问题。

归一化不能保证永远不会出现 `nan` 或 `inf`。如果输入本身已经是非有限值，或者学习率、权重、损失计算存在其他错误，归一化也无法恢复它；它只是让数值更容易保持在安全范围内。

##### 6. 一个直观例子

对向量

$$
x=[1000,1001,999]
$$

LayerNorm 会先减去均值约 1000，再除以标准差，因此得到近似中心为 0、尺度适中的向量。RMSNorm 则用这个向量的均方根作为分母，把整体幅度缩小到约 1，但不会把均值显式移到 0。

如果另一个向量只是它的 100 倍，归一化后两者的幅度仍会接近；这就是它们对数值尺度的稳定作用。

##### 7. 实际选择

- 处理卷积图像、batch 足够大：通常考虑 BatchNorm。
- 处理 Transformer 或 batch 很小：通常使用 LayerNorm。
- 需要更轻量的 Transformer 归一化，并且模型架构允许保留均值信息：可以使用 RMSNorm。

最终选择还取决于网络结构、归一化放置位置（Pre-Norm 或 Post-Norm）、硬件和训练配置。归一化层通常放在残差块或主要变换附近，用来控制每次子层计算接收到的数值尺度。

### Common ML Numerical Bugs

**Bug: Loss is NaN after a few epochs.**
Cause: logits grew too large, softmax overflowed. Or learning rate is too high and weights diverged.
Fix: use stable softmax (max subtraction), reduce learning rate, add gradient clipping.

**Bug: Loss is stuck at log(num_classes).**
Cause: model outputs are near-uniform probabilities. Often means gradients are vanishing or the model is not learning at all.
Fix: check that data labels are correct, verify the loss function, check for dead ReLUs.

**Bug: Validation accuracy is lower than expected by 1-3%.**
Cause: mixed precision without proper loss scaling. Gradient underflow silently zeroes out small updates.
Fix: enable dynamic loss scaling, or switch to bfloat16.

**Bug: Gradient norms are 0.0 for some layers.**
Cause: dead ReLU neurons (all inputs negative), or float16 underflow.
Fix: use LeakyReLU or GELU, use gradient scaling, check weight initialization.

**Bug: Model works on one GPU but gives different results on another.**
Cause: non-deterministic floating point accumulation order. GPU parallel reductions sum in different orders on different hardware, and floating point addition is not associative.
Fix: accept small differences (1e-6), or set `torch.use_deterministic_algorithms(True)` and accept the speed penalty.

**Bug: `exp()` returns `inf` in loss computation.**
Cause: raw logits passed to `exp()` without the max-subtraction trick.
Fix: use `torch.nn.functional.log_softmax()` which implements log-sum-exp internally.

**Bug: Training diverges after switching from float32 to float16.**
Cause: float16 cannot represent gradient magnitudes below 6e-8 or activations above 65,504.
Fix: use mixed precision with loss scaling (AMP), or use bfloat16 instead.

```figure
logsumexp-stability
```

## Build It

### Step 1: Demonstrate floating point precision limits

```python
print("=== Floating Point Precision ===")
print(f"0.1 + 0.2 = {0.1 + 0.2}")
print(f"0.1 + 0.2 == 0.3? {0.1 + 0.2 == 0.3}")
print(f"Difference: {(0.1 + 0.2) - 0.3:.2e}")
```

### Step 2: Implement naive vs stable softmax

```python
import math

def softmax_naive(logits):
    exps = [math.exp(z) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def softmax_stable(logits):
    max_logit = max(logits)
    exps = [math.exp(z - max_logit) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

safe_logits = [2.0, 1.0, 0.1]
print(f"Naive:  {softmax_naive(safe_logits)}")
print(f"Stable: {softmax_stable(safe_logits)}")

dangerous_logits = [100.0, 101.0, 102.0]
print(f"Stable: {softmax_stable(dangerous_logits)}")
# softmax_naive(dangerous_logits) would return [nan, nan, nan]
```

### Step 3: Implement stable log-sum-exp

```python
def logsumexp_naive(values):
    return math.log(sum(math.exp(v) for v in values))

def logsumexp_stable(values):
    c = max(values)
    return c + math.log(sum(math.exp(v - c) for v in values))

safe = [1.0, 2.0, 3.0]
print(f"Naive:  {logsumexp_naive(safe):.6f}")
print(f"Stable: {logsumexp_stable(safe):.6f}")

large = [500.0, 501.0, 502.0]
print(f"Stable: {logsumexp_stable(large):.6f}")
# logsumexp_naive(large) returns inf
```

### Step 4: Implement stable cross-entropy

```python
def cross_entropy_naive(true_class, logits):
    probs = softmax_naive(logits)
    return -math.log(probs[true_class])

def cross_entropy_stable(true_class, logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = math.log(sum(math.exp(s) for s in shifted))
    log_prob = shifted[true_class] - log_sum_exp
    return -log_prob

logits = [2.0, 5.0, 1.0]
true_class = 1
print(f"Naive:  {cross_entropy_naive(true_class, logits):.6f}")
print(f"Stable: {cross_entropy_stable(true_class, logits):.6f}")
```

### Step 5: Gradient checking

```python
def numerical_gradient(f, x, h=1e-5):
    grad = []
    for i in range(len(x)):
        x_plus = x[:]
        x_minus = x[:]
        x_plus[i] += h
        x_minus[i] -= h
        grad.append((f(x_plus) - f(x_minus)) / (2 * h))
    return grad

def check_gradient(analytical, numerical, tolerance=1e-5):
    for i, (a, n) in enumerate(zip(analytical, numerical)):
        denom = max(abs(a), abs(n), 1e-8)
        rel_error = abs(a - n) / denom
        status = "OK" if rel_error < tolerance else "FAIL"
        print(f"  param {i}: analytical={a:.8f} numerical={n:.8f} "
              f"rel_error={rel_error:.2e} [{status}]")

def f(params):
    x, y = params
    return x**2 + 3*x*y + y**3

def f_grad(params):
    x, y = params
    return [2*x + 3*y, 3*x + 3*y**2]

point = [2.0, 1.0]
analytical = f_grad(point)
numerical = numerical_gradient(f, point)
check_gradient(analytical, numerical)
```

## Use It

### Mixed precision simulation

```python
import struct

def float32_to_float16_round(x):
    packed = struct.pack('f', x)
    f32 = struct.unpack('f', packed)[0]
    packed16 = struct.pack('e', f32)
    return struct.unpack('e', packed16)[0]

def simulate_bfloat16(x):
    packed = struct.pack('f', x)
    as_int = int.from_bytes(packed, 'little')
    truncated = as_int & 0xFFFF0000
    repacked = truncated.to_bytes(4, 'little')
    return struct.unpack('f', repacked)[0]
```

### Gradient clipping

```python
def clip_by_norm(gradients, max_norm):
    total_norm = math.sqrt(sum(g**2 for g in gradients))
    if total_norm > max_norm:
        scale = max_norm / total_norm
        return [g * scale for g in gradients]
    return gradients

grads = [10.0, 20.0, 30.0]
clipped = clip_by_norm(grads, max_norm=5.0)
print(f"Original norm: {math.sqrt(sum(g**2 for g in grads)):.2f}")
print(f"Clipped norm:  {math.sqrt(sum(g**2 for g in clipped)):.2f}")
print(f"Direction preserved: {[c/clipped[0] for c in clipped]} == {[g/grads[0] for g in grads]}")
```

### NaN/Inf detection

```python
def check_tensor(name, values):
    has_nan = any(math.isnan(v) for v in values)
    has_inf = any(math.isinf(v) for v in values)
    if has_nan or has_inf:
        print(f"WARNING {name}: nan={has_nan} inf={has_inf}")
        return False
    return True

check_tensor("good", [1.0, 2.0, 3.0])
check_tensor("bad",  [1.0, float('nan'), 3.0])
check_tensor("ugly", [1.0, float('inf'), 3.0])
```

See `code/numerical.py` for complete implementations with all edge cases demonstrated.

## Ship It

This lesson produces:
- `code/numerical.py` with stable softmax, log-sum-exp, cross-entropy, gradient checking, and mixed precision simulation
- `outputs/prompt-numerical-debugger.md` for diagnosing NaN/Inf and numerical issues in training

These stable implementations reappear in Phase 3 when building the training loop and in Phase 4 when implementing attention mechanisms.

## Exercises

1. **Catastrophic cancellation.** Compute the variance of [1000000.0, 1000001.0, 1000002.0] using the naive formula `E[x^2] - E[x]^2` in float32. Then compute it using Welford's online algorithm. Compare the errors against the true variance (0.6667).

2. **Precision hunt.** Find the smallest positive float32 value `x` such that `1.0 + x == 1.0` in Python. This is the machine epsilon. Verify it matches `numpy.finfo(numpy.float32).eps`.

3. **Log-sum-exp edge cases.** Test your `logsumexp_stable` function with: (a) all values equal, (b) one value much larger than the rest, (c) all values very negative (-1000). Verify it gives correct results where the naive version fails.

4. **Gradient checking a neural network layer.** Implement a single linear layer `y = Wx + b` and its analytical backward pass. Use `numerical_gradient` to verify correctness for a 3x2 weight matrix.

5. **Loss scaling experiment.** Simulate training with float16: create random gradients in the range [1e-9, 1e-3], convert to float16, and measure what fraction become zero. Then apply loss scaling (multiply by 1024), convert to float16, scale back, and measure the zero fraction again.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| IEEE 754 | "The float standard" | International standard defining binary floating point formats, rounding rules, and special values (inf, nan). Every modern CPU and GPU implements it. |
| Machine epsilon | "The precision limit" | The smallest value e such that 1.0 + e != 1.0 in a given float format. For float32, it is about 1.19e-7. |
| Catastrophic cancellation | "Precision loss from subtraction" | When subtracting nearly equal floating point numbers, significant digits cancel and rounding noise dominates the result. |
| Overflow | "Number too big" | A result exceeds the maximum representable value and becomes inf. exp(89) overflows float32. |
| Underflow | "Number too small" | A result is closer to zero than the smallest representable positive number and becomes 0.0. exp(-104) underflows float32. |
| Log-sum-exp trick | "Subtract the max first" | Computing log(sum(exp(x))) by factoring out exp(max(x)) to prevent overflow and underflow. Used in softmax, cross-entropy, and log-probability math. |
| Stable softmax | "Softmax that does not explode" | Subtracting max(logits) before exponentiating. Numerically identical result, no overflow possible. |
| Gradient checking | "Verify your backprop" | Comparing analytical gradients from backpropagation against numerical gradients from finite differences to catch implementation bugs. |
| Mixed precision | "Float16 forward, float32 backward" | Using lower-precision floats for speed-critical operations and higher-precision floats for numerically sensitive operations. Typical speedup is 2-3x. |
| Loss scaling | "Prevent gradient underflow" | Multiplying the loss by a large constant before backprop so gradients stay in float16's representable range, then dividing by the same constant before weight updates. |
| bfloat16 | "Brain floating point" | Google's 16-bit format with 8 exponent bits (same range as float32) and 7 mantissa bits (less precision than float16). Preferred for training. |
| Gradient clipping | "Cap the gradient norm" | Scaling the gradient vector so its norm does not exceed a threshold. Prevents exploding gradients from ruining weights. |
| NaN | "Not a Number" | Special float value from undefined operations (0/0, inf-inf, sqrt(-1)). Propagates through all subsequent arithmetic. |
| Inf | "Infinity" | Special float value from overflow or division by zero. Can combine to produce NaN (inf - inf, inf * 0). |
| Numerical gradient | "Brute force derivative" | Approximating a derivative by evaluating f(x+h) and f(x-h) and dividing by 2h. Slow but reliable for verification. |

## Further Reading

- [What Every Computer Scientist Should Know About Floating-Point Arithmetic (Goldberg 1991)](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html) -- the definitive reference, dense but complete
- [Mixed Precision Training (Micikevicius et al., 2018)](https://arxiv.org/abs/1710.03740) -- the NVIDIA paper that introduced loss scaling for float16 training
- [AMP: Automatic Mixed Precision (PyTorch docs)](https://pytorch.org/docs/stable/amp.html) -- practical guide to mixed precision in PyTorch
- [bfloat16 format (Google Cloud TPU docs)](https://cloud.google.com/tpu/docs/bfloat16) -- why Google chose this format for TPUs
- [Kahan Summation (Wikipedia)](https://en.wikipedia.org/wiki/Kahan_summation_algorithm) -- algorithm for reducing rounding error in floating point sums
