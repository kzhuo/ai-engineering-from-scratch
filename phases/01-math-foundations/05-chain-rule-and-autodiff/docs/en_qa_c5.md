# Chain Rule & Automatic Differentiation

> The chain rule is the engine behind every neural network that learns.

**Type:** Build
**Language:** Python
**Prerequisites:** Phase 1, Lesson 04 (Derivatives & Gradients)
**Time:** ~90 minutes

## Learning Objectives

- Build a minimal autograd engine (Value class) that records operations and computes gradients via reverse-mode autodiff
- Implement forward and backward passes through a computation graph using topological sort
- Construct and train a multi-layer perceptron on XOR using only the from-scratch autograd engine
- Verify autodiff correctness using gradient checking against numerical finite differences

## The Problem

You can compute derivatives of simple functions. But a neural network is not a simple function. It is hundreds of functions composed together: matrix multiply, add bias, apply activation, matrix multiply again, softmax, cross-entropy loss. The output is a function of a function of a function.

To train the network, you need the gradient of the loss with respect to every single weight. Doing this by hand is impossible for millions of parameters. Doing it numerically (finite differences) is too slow.

The chain rule gives you the math. Automatic differentiation gives you the algorithm. Together they let you compute exact gradients through arbitrary compositions of functions in time proportional to a single forward pass.

This is how PyTorch, TensorFlow, and JAX work. You will build a miniature version from scratch.

## The Concept

### The Chain Rule

If `y = f(g(x))`, the derivative of `y` with respect to `x` is:

```
dy/dx = dy/dg * dg/dx = f'(g(x)) * g'(x)
```

Multiply the derivatives along the chain. Each link contributes its local derivative.

Example: `y = sin(x^2)`

```
g(x) = x^2       g'(x) = 2x
f(g) = sin(g)     f'(g) = cos(g)

dy/dx = cos(x^2) * 2x
```

For deeper compositions, the chain extends:

```
y = f(g(h(x)))

dy/dx = f'(g(h(x))) * g'(h(x)) * h'(x)
```

Every layer in a neural network is one link in this chain.

### Computational Graphs

A computational graph makes the chain rule visual. Every operation becomes a node. Data flows forward through the graph. Gradients flow backward.

**Forward pass (compute values):**

```mermaid
graph TD
    x1["x1 = 2"] --> mul["* (multiply)"]
    x2["x2 = 3"] --> mul
    mul -->|"a = 6"| add["+ (add)"]
    b["b = 1"] --> add
    add -->|"c = 7"| relu["relu"]
    relu -->|"y = 7"| y["output y"]
```

**Backward pass (compute gradients):**

```mermaid
graph TD
    dy["dy/dy = 1"] -->|"relu'(c)=1 since c>0"| dc["dy/dc = 1"]
    dc -->|"dc/da = 1"| da["dy/da = 1"]
    dc -->|"dc/db = 1"| db["dy/db = 1"]
    da -->|"da/dx1 = x2 = 3"| dx1["dy/dx1 = 3"]
    da -->|"da/dx2 = x1 = 2"| dx2["dy/dx2 = 2"]
```

The backward pass applies the chain rule at every node, propagating gradients from output to inputs.

### Forward Mode vs Reverse Mode

There are two ways to apply the chain rule through a graph.

**Forward mode** starts at the inputs and pushes derivatives forward. It computes `dx/dx = 1` and propagates through each operation. Good when you have few inputs and many outputs.

```
Forward mode: seed dx/dx = 1, propagate forward

  x = 2       (dx/dx = 1)
  a = x^2     (da/dx = 2x = 4)
  y = sin(a)  (dy/dx = cos(a) * da/dx = cos(4) * 4 = -2.615)
```

**Reverse mode** starts at the output and pulls gradients backward. It computes `dy/dy = 1` and propagates through each operation in reverse. Good when you have many inputs and few outputs.

```
Reverse mode: seed dy/dy = 1, propagate backward

  y = sin(a)  (dy/dy = 1)
  a = x^2     (dy/da = cos(a) = cos(4) = -0.654)
  x = 2       (dy/dx = dy/da * da/dx = -0.654 * 4 = -2.615)
```

Neural networks have millions of inputs (weights) and one output (loss). Reverse mode computes all gradients in one backward pass. This is why backpropagation uses reverse mode.

| Mode | Seed | Direction | Best when |
|------|------|-----------|-----------|
| Forward | `dx_i/dx_i = 1` | Input to output | Few inputs, many outputs |
| Reverse | `dy/dy = 1` | Output to input | Many inputs, few outputs (neural nets) |

### Dual Numbers for Forward Mode

Forward mode can be implemented elegantly with dual numbers. A dual number has the form `a + b*epsilon` where `epsilon^2 = 0`.

```text
Dual number: (value, derivative)

(2, 1) means: value is 2, derivative w.r.t. x is 1

Arithmetic rules:
  (a, a') + (b, b') = (a+b, a'+b')
  (a, a') * (b, b') = (a*b, a'*b + a*b')
  sin(a, a')         = (sin(a), cos(a)*a')
```

Seed the input variable with derivative 1. The derivative propagates automatically through every operation.

核心想法：**把“值”和“导数”绑成同一个数一起做运算**。你只算一遍算术，导数就跟着出来了。

#### Dual number 是什么

普通复数是 `a + b i`，规定 `i² = -1`。
Dual number 长得很像：`a + b ε`，但规定 **`ε² = 0`**。

课里把它写成一对：

```text
(a, a')  ⇔  a + a'·ε
```

- `a`：函数值
- `a'`：这个值对某个自变量（通常是 `x`）的导数

所以 `(2, 1)` 的意思是：当前值是 2，而且我们对 `x` 求导，种子是 `dx/dx = 1`。

`ε² = 0` 不是物理事实，是**故意砍掉二阶及以上项**。Taylor 展开里：

```text
f(a + a'ε) = f(a) + f'(a)·(a'ε) + (1/2)f''(a)·(a'ε)² + ...
           = f(a) + f'(a)·a'·ε
```

因为 `(ε)² = 0`，后面全消失。于是：**对 dual number 求一次 `f`，输出的 ε 系数正好是一阶导数。**

#### 运算规则从哪来

把两个 dual number 当真的代数去乘，就能推出上面那三条规则。

**加法**（导数线性）

```text
(a + a'ε) + (b + b'ε) = (a+b) + (a'+b')ε
⇒  (a, a') + (b, b') = (a+b, a'+b')
```

**乘法**（乘积法则）

```text
(a + a'ε)(b + b'ε)
  = ab + a·b'ε + a'ε·b + a'b' ε²
  = ab + (a'b + ab')ε          ← ε² = 0，最后一项没了
⇒  (a, a') * (b, b') = (a·b, a'·b + a·b')
```

ε 系数恰好是微积分里的 `(uv)' = u'v + uv'`。

**sin**（链式法则）

```text
sin(a + a'ε) = sin(a) + cos(a)·a'·ε
⇒  sin(a, a') = (sin(a), cos(a)·a')
```

任何光滑函数都一样：值走 `f(a)`，导数走 `f'(a)·a'`。

| 运算 | 值 | 导数（ε 系数） |
|------|----|----------------|
| `+` | `a+b` | `a'+b'` |
| `*` | `ab` | `a'b + ab'` |
| `sin` | `sin(a)` | `cos(a)·a'` |
| `exp` | `e^a` | `e^a · a'` |
| 常数 `c` | `c` | `0` |
| 自变量 `x` | `x` | `1`（seed） |

常数导数必须是 0：`3` 写成 `(3, 0)`。如果你写成 `(3, 1)`，等于在说“3 会跟着 x 变”，后面全错。

#### Seed：为什么输入要写成 `(x, 1)`

Forward mode 的问题是：**给定 `x`，`y` 怎么随 `x` 变？**

所以：

- 自变量：`(x, 1)` —— `dx/dx = 1`
- 常数 / 暂时不当作自变量的量：`(c, 0)`

之后你**不用再写求导公式**。每次 `+`、`*`、`sin` 都按上面的规则同时更新值和导数。

这就是上面那句：The derivative propagates automatically through every operation.

#### 完整走一遍：`y = sin(x²)`，`x = 2`

前面用手工链式法则算过：`dy/dx = cos(x²)·2x = 4·cos(4) ≈ -2.615`。用 dual number 会得到同一个数。

**Step 0 — seed**

```text
x = (2, 1)     # 值 2，dx/dx = 1
```

**Step 1 — `a = x * x`（也就是 x²）**

```text
(2, 1) * (2, 1)
  = (2·2, 1·2 + 2·1)
  = (4, 4)
```

值和导数同时出来：`a = 4`，`da/dx = 4`。手工：`(x²)' = 2x = 4`。

**Step 2 — `y = sin(a)`**

```text
sin(4, 4) = (sin(4), cos(4)·4) ≈ (-0.757, -2.615)
```

输出是：

- 函数值 `y ≈ -0.757`
- 导数 `dy/dx ≈ -2.615`

一次前向计算，两个结果都有。没有建图，没有 backward。

用计算图看，导数是**顺着数据往前推**的：

```mermaid
graph LR
    x["x = (2, 1)"] --> mul["*"]
    x --> mul
    mul -->|"a = (4, 4)"| s["sin"]
    s -->|"y ≈ (-0.757, -2.615)"| y["output"]
```

#### 再看一个更“神经网络味”的式子

`y = x² + 3x + 1`，在 `x = 2`。手工：`y' = 2x + 3 = 7`。

```text
x      = (2, 1)
x*x    = (4, 4)
3      = (3, 0)          # 常数，导数 0
3*x    = (6, 3)
1      = (1, 0)
y      = (4,4) + (6,3) + (1,0) = (11, 7)
```

`y = 11`，`dy/dx = 7`。这就是后面 PyTorch 那段 `x.grad == 7.0` 的 **forward-mode 版本**。

#### 和 Reverse mode 的差别

同一条链 `y = sin(x²)`，`x = 2`：

| | Forward（dual numbers） | Reverse（后面要写的 Value 引擎） |
|--|------------------------|----------------------------------|
| Seed | `dx/dx = 1`，从输入出发 | `dy/dy = 1`，从输出出发 |
| 带什么 | 每个中间量带 `d(·)/dx` | 每个中间量带 `dy/d(·)` |
| 一次 pass 得到 | **一个输入**对所有输出的导数 | **一个输出**对所有输入的导数 |
| 神经网络 | 权重几百万个 → 要几百万次 forward-mode | 损失只有 1 个 → **一次 backward 全拿到** |

所以：

- Dual numbers：实现 forward-mode 最干净的方式
- 课里的 `Value` + `backward()`：reverse-mode（backprop）
- 两者数学上都是链式法则，方向相反

Exercise 4 让你写 `Dual` class，就是为了亲手对比：同一个表达式，forward dual 和 reverse `Value` 应给出相同导数。

#### 实现时脑子里要有的接口

把 `(value, derivative)` 当成类型，重载运算即可：

```python
import math

class Dual:
    def __init__(self, real, dual=0.0):
        self.real = real   # a
        self.dual = dual   # a'

    def __add__(self, other):
        other = other if isinstance(other, Dual) else Dual(other)
        return Dual(self.real + other.real, self.dual + other.dual)

    def __mul__(self, other):
        other = other if isinstance(other, Dual) else Dual(other)
        return Dual(
            self.real * other.real,
            self.dual * other.real + self.real * other.dual,
        )

    def sin(self):
        return Dual(math.sin(self.real), math.cos(self.real) * self.dual)

x = Dual(2.0, 1.0)      # seed
y = (x * x).sin()
print(y.real, y.dual)   # sin(4), 4*cos(4)
```

注意：`Dual(2.0, 1.0)` 是“对 x 求导”；如果表达式里还有 `w`，而你这次只想要 `∂y/∂x`，那么 `w` 必须是 `Dual(w_val, 0)`。换一次 seed（让 `w` 的 dual 分量 = 1，`x` 的 = 0），才能得到 `∂y/∂w`。这就是 **few inputs, many outputs** 时 forward mode 合适、神经网络里不合适的原因：每个参数都要单独 seed 跑一遍。

**一句话：** dual number 把链式法则塞进四则运算里。`ε² = 0` 保证只保留一阶项；输入 seed 成 `(x, 1)` 之后，中间每一步的第二个分量都已经是对 `x` 的导数。

### Building an Autograd Engine

An autograd engine needs three things:

1. **Value wrapping.** Wrap every number in an object that stores its value and gradient.
2. **Graph recording.** Every operation records its inputs and the local gradient function.
3. **Backward pass.** Topological sort the graph, then walk it in reverse, applying the chain rule at each node.

#### 这三部分如何协作

自动求导的本质是：前向传播时计算数值并记录计算过程，反向传播时沿记录好的计算图应用链式法则。

先注意两个容易混淆的概念：

- `gradient` 是梯度或导数，不是一个数值模糊的“变化趋势”。
- `local gradient function` 是计算局部导数的函数，不是前向传播时已经算好的最终梯度。

##### 1. Value wrapping：同时保存数值与求导信息

普通浮点数只保存数值。一个教学版 `Value` 对象还会保存：

- `data`：该节点在前向传播中计算出的数值
- `grad`：最终输出 `L` 对该节点的导数
- `_prev`：产生该节点的输入节点
- `_op`：产生该节点的操作
- `_backward`：把上游梯度传给输入节点的方法

如果中间节点是 `v`，那么 `v.grad` 的含义是：

```text
v.grad = ∂L/∂v
```

这里的梯度取决于最终选择哪个量作为输出 `L`，因此不是 `v` 本身固有的属性。

前向传播期间只计算 `data` 并记录图，通常不会计算 `grad`。在本课的 `Value` 引擎中，所有节点的 `grad` 初始值都是 `0.0`。

##### 2. Graph recording：每次运算都创建一个新节点

考虑：

```text
a = x × y
L = a + x
```

执行 `a = x * y` 时，引擎不仅计算 `a.data`，还记录：

```text
a 的输入：x、y
a 的操作：乘法
局部导数：∂a/∂x = y，∂a/∂y = x
```

执行 `L = a + x` 时，再记录：

```text
L 的输入：a、x
L 的操作：加法
局部导数：∂L/∂a = 1，∂L/∂x = 1
```

因此，同一个 `x` 可以通过两条路径影响 `L`：

```mermaid
graph LR
    x --> mul["a = x * y"]
    y --> mul
    mul --> add["L = a + x"]
    x --> add
```

局部导数只描述一个操作节点附近的变化。最终需要的全局梯度则由链式法则把各节点的局部导数组合起来：

```text
输入梯度 += 上游梯度 × 局部导数
```

例如乘法节点向 `x` 传播梯度时：

```text
∂L/∂x（经过 a 的路径） = (∂L/∂a)(∂a/∂x)
```

对应实现：

```python
self.grad += out.grad * other.data
```

其中 `out.grad` 是上游传来的梯度，`other.data` 是乘法对 `self` 的局部导数。

##### 3. Backward pass：从输出一直传播到所有上游叶子节点

设 `x = 2`、`y = 3`。前向传播结束后：

```text
a.data = 6
L.data = 8

x.grad = 0
y.grad = 0
a.grad = 0
L.grad = 0
```

此时 `a.grad` 仍然只是默认值。调用 `L.backward()` 后才开始计算梯度。

第一步把输出作为反向传播的起点。任何量对自身的导数都是 1：

```text
L.grad = ∂L/∂L = 1
```

然后反向经过 `L = a + x`：

```text
a.grad += 1 × 1 = 1
x.grad += 1 × 1 = 1    # x 到 L 的直接路径
```

接着反向经过 `a = x × y`：

```text
x.grad += a.grad × y = 1 × 3 = 3
y.grad += a.grad × x = 1 × 2 = 2
```

最终：

```text
x.grad = 1 + 3 = 4
y.grad = 2
a.grad = 1
L.grad = 1
```

手工求导可以验证：

```text
∂L/∂x = ∂(xy + x)/∂x = y + 1 = 4
∂L/∂y = ∂(xy + x)/∂y = x = 2
```

这里必须使用 `+=` 而不是 `=`，因为一个变量可能通过多条路径影响输出。各条路径贡献的梯度需要相加，不能互相覆盖。

反向传播前还需要拓扑排序。对于上面的图，一个合法的正向拓扑顺序是：

```text
x, y, a, L
```

将它反向遍历：

```text
L, a, y, x
```

这样可以保证某个节点先收齐所有下游路径传回的梯度，再把完整梯度继续传给它的输入。因此，`L.backward()` 会沿计算图一直计算到所有与 `L` 相连的上游叶子节点，而不只计算 `L` 附近的梯度。

This is exactly what PyTorch's `autograd` does. The `torch.Tensor` class wraps values, records operations when `requires_grad=True`, and computes gradients when you call `.backward()`.

### How PyTorch Autograd Works Under the Hood

When you write PyTorch code:

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2 + 3 * x + 1
y.backward()
print(x.grad)  # 7.0 = 2*x + 3 = 2*2 + 3
```

PyTorch internally:

1. Creates a `Tensor` node for `x` with `requires_grad=True`
2. Every operation (`**`, `*`, `+`) creates a new node and records the backward function
3. `y.backward()` triggers reverse-mode autodiff through the recorded graph
4. Each node's `grad_fn` computes local gradients and passes them to parent nodes
5. Gradients accumulate in `.grad` attributes via addition (not replacement)

The graph is dynamic (define-by-run). A new graph is built on every forward pass. This is why PyTorch supports control flow (if/else, loops) inside models.

需要区分“梯度参与了反向传播”和“梯度保存在 `.grad` 中”：

- 前向传播时，PyTorch 计算张量值并建立 `grad_fn` 计算图，不会提前算出各节点的梯度。
- 调用 `y.backward()` 后，梯度会沿所有可导且与 `y` 相连的上游路径传播。
- 默认情况下，只有 `requires_grad=True` 的叶子张量会把累积结果保存在 `.grad` 中。
- 像 `a = x * y` 这样的非叶子张量会参与梯度计算，但 `a.grad` 默认通常是 `None`；要查看它，需要在反向传播前调用 `a.retain_grad()`。
- `requires_grad=False`、`.detach()` 或 `torch.no_grad()` 会阻止相应路径被记录或继续传播梯度。

PyTorch 的梯度会累加。多次训练迭代之间通常要先调用优化器的 `zero_grad()`，或者手动把参数梯度清零，否则本轮梯度会叠加到上一轮结果上。

## Build It

### Step 1: The Value class

```python
class Value:
    def __init__(self, data, children=(), op=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(children)
        self._op = op

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"
```

Every `Value` stores its numeric data, its gradient (initially zero), a backward function, and pointers to child nodes that produced it.

### Step 2: Arithmetic operations with gradient tracking

```python
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0, self.data), (self,), 'relu')
        def _backward():
            self.grad += (1.0 if out.data > 0 else 0.0) * out.grad
        out._backward = _backward
        return out
```

Each operation creates a closure that knows how to compute local gradients and multiply by the upstream gradient (`out.grad`). The `+=` handles the case where a value is used in multiple operations.

### Step 3: The backward pass

```python
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1.0
        for v in reversed(topo):
            v._backward()
```

Topological sort ensures every node's gradient is fully computed before it propagates to its children. The seed gradient is 1.0 (dy/dy = 1).

### Step 4: More operations for a complete engine

The basic Value class handles addition, multiplication, and relu. A real autograd engine needs more. Here are the operations you need to build neural networks:

```python
    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return other + (-self)

    def __pow__(self, n):
        out = Value(self.data ** n, (self,), f'**{n}')
        def _backward():
            self.grad += n * (self.data ** (n - 1)) * out.grad
        out._backward = _backward
        return out

    def __truediv__(self, other):
        return self * (other ** -1) if isinstance(other, Value) else self * (Value(other) ** -1)

    def exp(self):
        import math
        e = math.exp(self.data)
        out = Value(e, (self,), 'exp')
        def _backward():
            self.grad += e * out.grad
        out._backward = _backward
        return out

    def log(self):
        import math
        out = Value(math.log(self.data), (self,), 'log')
        def _backward():
            self.grad += (1.0 / self.data) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        import math
        t = math.tanh(self.data)
        out = Value(t, (self,), 'tanh')
        def _backward():
            self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward
        return out
```

**Why each operation matters:**

| Operation | Backward rule | Used in |
|-----------|--------------|---------|
| `__sub__` | Reuses add + neg | Loss computation (pred - target) |
| `__pow__` | n * x^(n-1) | Polynomial activations, MSE (error^2) |
| `__truediv__` | Reuses mul + pow(-1) | Normalization, learning rate scaling |
| `exp` | exp(x) * upstream | Softmax, log-likelihood |
| `log` | (1/x) * upstream | Cross-entropy loss, log probabilities |
| `tanh` | (1 - tanh^2) * upstream | Classic activation function |

The clever part: `__sub__` and `__truediv__` are defined in terms of existing operations. They get correct gradients for free because the chain rule composes through the underlying add/mul/pow operations.

### Step 5: Mini MLP from scratch

With a complete Value class, you can build a neural network. No PyTorch. No NumPy. Just Values and the chain rule.

```python
import random

class Neuron:
    def __init__(self, n_inputs):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_inputs)]
        self.b = Value(0.0)

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh()

    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, n_inputs, n_outputs):
        self.neurons = [Neuron(n_inputs) for _ in range(n_outputs)]

    def __call__(self, x):
        return [n(x) for n in self.neurons]

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

class MLP:
    def __init__(self, sizes):
        self.layers = [Layer(sizes[i], sizes[i+1]) for i in range(len(sizes)-1)]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x[0] if len(x) == 1 else x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
```

A `Neuron` computes `tanh(w1*x1 + w2*x2 + ... + b)`. A `Layer` is a list of neurons. An `MLP` stacks layers. Every weight is a `Value`, so calling `loss.backward()` propagates gradients to every parameter.

**Training on XOR:**

```python
random.seed(42)
model = MLP([2, 4, 1])  # 2 inputs, 4 hidden neurons, 1 output

xs = [[0, 0], [0, 1], [1, 0], [1, 1]]
ys = [-1, 1, 1, -1]  # XOR pattern (using -1/1 for tanh)

for step in range(100):
    preds = [model(x) for x in xs]
    loss = sum((p - y) ** 2 for p, y in zip(preds, ys))

    for p in model.parameters():
        p.grad = 0.0
    loss.backward()

    lr = 0.05
    for p in model.parameters():
        p.data -= lr * p.grad

    if step % 20 == 0:
        print(f"step {step:3d}  loss = {loss.data:.4f}")

print("\nPredictions after training:")
for x, y in zip(xs, ys):
    print(f"  input={x}  target={y:2d}  pred={model(x).data:6.3f}")
```

This is micrograd. A complete neural network training loop in pure Python with automatic differentiation. Every commercial deep learning framework does the same thing at massive scale.

### Step 6: Gradient checking

How do you know your autodiff is correct? Compare it against numerical derivatives. This is gradient checking.

```python
def gradient_check(build_expr, x_val, h=1e-7):
    x = Value(x_val)
    y = build_expr(x)
    y.backward()
    autodiff_grad = x.grad

    y_plus = build_expr(Value(x_val + h)).data
    y_minus = build_expr(Value(x_val - h)).data
    numerical_grad = (y_plus - y_minus) / (2 * h)

    diff = abs(autodiff_grad - numerical_grad)
    return autodiff_grad, numerical_grad, diff
```

Test it on a complex expression:

```python
def expr(x):
    return (x ** 3 + x * 2 + 1).tanh()

ad, num, diff = gradient_check(expr, 0.5)
print(f"Autodiff:  {ad:.8f}")
print(f"Numerical: {num:.8f}")
print(f"Difference: {diff:.2e}")
# Difference should be < 1e-5
```

Gradient checking is essential when implementing new operations. If your backward pass has a bug, the numerical check catches it. Every serious deep learning implementation runs gradient checks during development.

**When to use gradient checking:**

| Situation | Do gradient check? |
|-----------|-------------------|
| Adding a new operation to your autograd | Yes, always |
| Debugging a training loop that won't converge | Yes, check gradients first |
| Production training | No, too slow (2x forward passes per parameter) |
| Unit tests for autograd code | Yes, automate it |

### Step 7: Verify against manual calculation

```python
x1 = Value(2.0)
x2 = Value(3.0)
a = x1 * x2          # a = 6.0
b = a + Value(1.0)    # b = 7.0
y = b.relu()          # y = 7.0

y.backward()

print(f"y = {y.data}")          # 7.0
print(f"dy/dx1 = {x1.grad}")   # 3.0 (= x2)
print(f"dy/dx2 = {x2.grad}")   # 2.0 (= x1)
```

Manual check: `y = relu(x1*x2 + 1)`. Since `x1*x2 + 1 = 7 > 0`, relu is identity.
`dy/dx1 = x2 = 3`. `dy/dx2 = x1 = 2`. The engine matches.

## Use It

### Verify against PyTorch

```python
import torch

x1 = torch.tensor(2.0, requires_grad=True)
x2 = torch.tensor(3.0, requires_grad=True)
a = x1 * x2
b = a + 1.0
y = torch.relu(b)
y.backward()

print(f"PyTorch dy/dx1 = {x1.grad.item()}")  # 3.0
print(f"PyTorch dy/dx2 = {x2.grad.item()}")  # 2.0
```

Same gradients. Your engine computes the same result as PyTorch because the math is the same: reverse-mode autodiff via the chain rule.

### A more complex expression

```python
a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)
f = (a * b + c).relu()  # relu(2*(-3) + 10) = relu(4) = 4

f.backward()
print(f"df/da = {a.grad}")  # -3.0 (= b)
print(f"df/db = {b.grad}")  #  2.0 (= a)
print(f"df/dc = {c.grad}")  #  1.0
```

## Ship It

This lesson produces:
- `outputs/skill-autodiff.md` -- a skill for building and debugging autograd systems
- `code/autodiff.py` -- a minimal autograd engine you can extend

The Value class built here is the foundation for the neural network training loop in Phase 3.

## Exercises

1. Add `__pow__` to the Value class so you can compute `x ** n`. Verify that `d/dx(x^3)` at `x=2` equals `12.0`.

2. Add `tanh` as an activation function. Verify that `tanh'(0) = 1` and `tanh'(2) = 0.0707` (approx).

3. Build a computation graph for a single neuron: `y = relu(w1*x1 + w2*x2 + b)`. Compute all five gradients and verify against PyTorch.

4. Implement forward-mode autodiff using dual numbers. Create a `Dual` class and verify it gives the same derivatives as your reverse-mode engine.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Chain rule | "Multiply the derivatives" | The derivative of composed functions equals the product of each function's local derivative, evaluated at the right point |
| Computational graph | "The network diagram" | A directed acyclic graph where nodes are operations and edges carry values (forward) or gradients (backward) |
| Forward mode | "Push derivatives forward" | Autodiff that propagates derivatives from inputs to outputs. One pass per input variable. |
| Reverse mode | "Backpropagation" | Autodiff that propagates gradients from outputs to inputs. One pass per output variable. |
| Autograd | "Automatic gradients" | A system that records operations on values, builds a graph, and computes exact gradients via the chain rule |
| Dual numbers | "Value plus derivative" | Numbers of the form a + b*epsilon (epsilon^2 = 0) that carry derivative information through arithmetic |
| Topological sort | "Dependency order" | Ordering graph nodes so every node comes after all its dependencies. Required for correct gradient propagation. |
| Gradient accumulation | "Add, don't replace" | When a value feeds into multiple operations, its gradient is the sum of all incoming gradient contributions |
| Dynamic graph | "Define by run" | A computation graph rebuilt on every forward pass, allowing Python control flow inside models (PyTorch style) |
| Gradient checking | "Numerical verification" | Comparing autodiff gradients against numerical finite-difference gradients to verify correctness. Essential for debugging. |
| MLP | "Multi-layer perceptron" | A neural network with one or more hidden layers of neurons. Each neuron computes a weighted sum plus bias, then applies an activation function. |
| Neuron | "Weighted sum + activation" | The basic unit: output = activation(w1*x1 + w2*x2 + ... + b). The weights and bias are learnable parameters. |

## Further Reading

- [3Blue1Brown: Backpropagation calculus](https://www.youtube.com/watch?v=tIeHLnjs5U8) -- visual explanation of the chain rule in neural networks
- [PyTorch Autograd mechanics](https://pytorch.org/docs/stable/notes/autograd.html) -- how the real system works
- [Baydin et al., Automatic Differentiation in Machine Learning: a Survey](https://arxiv.org/abs/1502.05767) -- comprehensive reference
