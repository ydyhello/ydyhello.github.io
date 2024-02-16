---
title: minitorch系列记录
date: 2023-11-24 17:05:22
tags:
categories:
    - Pytorch算子
---
> MiniTorch是对 torch api的纯Python重新实现，设计简单、易于阅读、容易进行测试和扩充。miniTorch最终的库可以运行Torch代码。
<!--more-->
Mycode:

- [module0](https://github.com/ydyhello/minitorch-module-0-ydyhello)
- [module1](https://github.com/ydyhello/minitorch-module-1-ydyhello)

MiniTorch官方文档:

- [Docs](https://minitorch.github.io/)

还可以参考：MiniTorch-学习全攻略

<br>
 
{% pdf  ./MiniTorch-学习全攻略.pdf %} 
 
<br>

## module0

[Overview](https://minitorch.github.io/module0/module0/#guides)

### 环境

https://blog.csdn.net/ChaoFeiLi/article/details/124760367

测试一下

![1.png](https://s2.loli.net/2024/02/16/DZgpTwjJ2V6FOEx.png)

### Task 0.1: Operators

```python
# ## Task 0.1
#
# Implementation of a prelude of elementary functions.


def mul(x: float, y: float) -> float:
    "$f(x, y) = x * y$"
    # TODO: Implement for Task 0.1.
    return x * y

    raise NotImplementedError("Need to implement for Task 0.1")


def id(x: float) -> float:
    "$f(x) = x$"
    # TODO: Implement for Task 0.1.
    return x
    raise NotImplementedError("Need to implement for Task 0.1")


def add(x: float, y: float) -> float:
    "$f(x, y) = x + y$"
    # TODO: Implement for Task 0.1.
    return x + y
    raise NotImplementedError("Need to implement for Task 0.1")


def neg(x: float) -> float:
    "$f(x) = -x$"
    # TODO: Implement for Task 0.1.
    return -x
    raise NotImplementedError("Need to implement for Task 0.1")


def lt(x: float, y: float) -> float:
    "$f(x) =$ 1.0 if x is less than y else 0.0"
    # TODO: Implement for Task 0.1.
    return 1.0 if x < y else 0.0
    raise NotImplementedError("Need to implement for Task 0.1")


def eq(x: float, y: float) -> float:
    "$f(x) =$ 1.0 if x is equal to y else 0.0"
    # TODO: Implement for Task 0.1.
    return 1.0 if x == y else 0.0
    raise NotImplementedError("Need to implement for Task 0.1")


def max(x: float, y: float) -> float:
    "$f(x) =$ x if x is greater than y else y"
    # TODO: Implement for Task 0.1.
    return x if x > y else y

    raise NotImplementedError("Need to implement for Task 0.1")


def is_close(x: float, y: float) -> float:
    "$f(x) = |x - y| < 1e-2$"
    # TODO: Implement for Task 0.1.
    return abs(x - y) < 1e-2
    raise NotImplementedError("Need to implement for Task 0.1")


def sigmoid(x: float) -> float:
    r"""
    $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$

    (See https://en.wikipedia.org/wiki/Sigmoid_function )

    Calculate as

    $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$

    for stability.
    """
    # TODO: Implement for Task 0.1.
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))
    raise NotImplementedError("Need to implement for Task 0.1")


def relu(x: float) -> float:
    """
    $f(x) =$ x if x is greater than 0, else 0

    (See https://en.wikipedia.org/wiki/Rectifier_(neural_networks) .)
    """
    # TODO: Implement for Task 0.1.
    return x if x > 0.0 else 0.0
    raise NotImplementedError("Need to implement for Task 0.1")


EPS = 1e-6


def log(x: float) -> float:
    "$f(x) = log(x)$"
    return math.log(x + EPS)


def exp(x: float) -> float:
    "$f(x) = e^{x}$"
    return math.exp(x)


def log_back(x: float, d: float) -> float:
    r"If $f = log$ as above, compute $d \times f'(x)$"
    # TODO: Implement for Task 0.1.
    return d * 1.0 / x
    raise NotImplementedError("Need to implement for Task 0.1")


def inv(x: float) -> float:
    "$f(x) = 1/x$"
    # TODO: Implement for Task 0.1.
    return 1.0 / x
    raise NotImplementedError("Need to implement for Task 0.1")


def inv_back(x: float, d: float) -> float:
    r"If $f(x) = 1/x$ compute $d \times f'(x)$"
    # TODO: Implement for Task 0.1.
    return -d / x ** 2
    raise NotImplementedError("Need to implement for Task 0.1")


def relu_back(x: float, d: float) -> float:
    r"If $f = relu$ compute $d \times f'(x)$"
    # TODO: Implement for Task 0.1.
    return d if x > 0 else 0.0
    raise NotImplementedError("Need to implement for Task 0.1")
```

![2.png](https://s2.loli.net/2024/02/16/3nwMcTAP75mIYvj.png)

### Task 0.2: Testing and Debugging

```python
# ## Task 0.2 - Property Testing

# Implement the following property checks
# that ensure that your operators obey basic
# mathematical rules.


@pytest.mark.task0_2
@given(small_floats)
def test_sigmoid(a: float) -> None:
    """Check properties of the sigmoid function, specifically
    * It is always between 0.0 and 1.0.
    * one minus sigmoid is the same as sigmoid of the negative
    * It crosses 0 at 0.5
    * It is  strictly increasing.
    """
    # TODO: Implement for Task 0.2.
    assert sigmoid(a) >= 0.0 # 断言检测大于等于0
    assert sigmoid(a) <= 1.0 # 小于等于0
    assert_close(1 - sigmoid(a), sigmoid(-a)) # sigmoid(-x) = 1 - sigmoid(x)
    assert_close(sigmoid(0), 0.5) # sigmoid(0)接近0.5
    assert sigmoid(a + 1.0) >= sigmoid(a) # 递增
    #raise NotImplementedError("Need to implement for Task 0.2")


@pytest.mark.task0_2
@given(small_floats, small_floats, small_floats)
def test_transitive(a: float, b: float, c: float) -> None:
    "Test the transitive property of less-than (a < b and b < c implies a < c)"
    # 如果 a < b 和 b < c 成立，那么 a < c 也应该成立
    # 传递律
    # TODO: Implement for Task 0.2.
    if lt(a, b) and lt(b, c):
        assert lt(a, c)
    elif lt(a, c) and lt(c, b):
        assert lt(a, b)
    elif lt(b, c) and lt(c, a):
        assert lt(b, a)
    # raise NotImplementedError("Need to implement for Task 0.2")


@pytest.mark.task0_2
@given(small_floats, small_floats)
def test_symmetric(a:float, b:float) -> None:
    """
    Write a test that ensures that :func:`minitorch.operators.mul` is symmetric, i.e.
    gives the same value regardless of the order of its input.
    """
    # 对称性
    # a*b=b*a
    # TODO: Implement for Task 0.2.
    assert mul(a, b) == mul(b, a)
    # raise NotImplementedError("Need to implement for Task 0.2")


@pytest.mark.task0_2
@given(small_floats, small_floats, small_floats)
def test_distribute(x: float, y: float, z: float) -> None:
    """
    Write a test that ensures that your operators distribute, i.e.
    :math:`z \times (x + y) = z \times x + z \times y`
    """
    # TODO: Implement for Task 0.2.
    # z * (x + y) 和 z * x + z * y
    # 分配律
    assert_close(mul(z, add(x, y)), add(mul(z, x), mul(z, y)))
    # raise NotImplementedError("Need to implement for Task 0.2")


@pytest.mark.task0_2
@given(small_floats, small_floats)
# 想不出来其他的了
def test_other(a:float, b:float) -> None:
    """
    Write a test that ensures some other property holds for your functions.
    """
    # TODO: Implement for Task 0.2.
    assert mul(a, b) == mul(b, a)
    # raise NotImplementedError("Need to implement for Task 0.2")
```

![3.png](https://s2.loli.net/2024/02/16/a9M5AnwsfdgBuOH.png)

### Task 0.3: Functional Python

```python
# ## Task 0.3

# Small practice library of elementary higher-order functions.


def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """
    Higher-order map.

    See https://en.wikipedia.org/wiki/Map_(higher-order_function)

    Args:
        fn: Function from one value to one value.

    Returns:
        A function that takes a list, applies `fn` to each element, and returns a
         new list
    """
    # 闭包
    # 捕获了传入的 fn 参数，并可以在以后的调用中使用它
    # TODO: Implement for Task 0.3.
    def myFn(ls: Iterable[float]) -> Iterable[float]:
        return [fn(e) for e in ls]
    return myFn
    # raise NotImplementedError("Need to implement for Task 0.3")


def negList(ls: Iterable[float]) -> Iterable[float]:
    "Use `map` and `neg` to negate each element in `ls`"
    # TODO: Implement for Task 0.3.
    # 将 neg 函数应用到 ls 中的每个元素，以生成一个新的可迭代的序列，其中每个元素都是 ls 中对应元素的相反数
    return map(neg)(ls)
    # raise NotImplementedError("Need to implement for Task 0.3")


def zipWith(
    fn: Callable[[float, float], float], ls1: Iterable[float], ls2: Iterable[float]
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """
    Higher-order zipwith (or map2).

    See https://en.wikipedia.org/wiki/Map_(higher-order_function)

    Args:
        fn: combine two values

    Returns:
        Function that takes two equally sized lists `ls1` and `ls2`, produce a new list by
         applying fn(x, y) on each pair of elements.

    """
    # TODO: Implement for Task 0.3.
    ls = [] # 用于存储 fn 函数应用后的结果
    # 使用 zip 函数将 ls1 和 ls2 中对应位置的元素进行配对，然后使用 for 循环遍历这些配对。在循环中，对每一对元素 (x, y) 调用 fn(x, y)，将结果追加到 ls 列表中
    for x, y in zip(ls1, ls2):
        ls.append(fn(x, y))
    return ls
    # raise NotImplementedError("Need to implement for Task 0.3")


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    "Add the elements of `ls1` and `ls2` using `zipWith` and `add`"
    # TODO: Implement for Task 0.3.
    return zipWith(add, ls1, ls2) # 相加
    # raise NotImplementedError("Need to implement for Task 0.3")


def reduce(
    fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    r"""
    Higher-order reduce.

    Args:
        fn: combine two values
        start: start value $x_0$

    Returns:
        Function that takes a list `ls` of elements
         $x_1 \ldots x_n$ and computes the reduction :math:`fn(x_3, fn(x_2,
         fn(x_1, x_0)))`
    """
    # TODO: Implement for Task 0.3.
    # start是起始值
    # 归约
    # 依次应用 fn 函数将序列中的元素合并在一起
    def myFn(ls: Iterable[float]) -> float:
        # 闭包中并不会修改外部start的值
        t = start
        for e in ls:
            t = fn(e, t) # 使用 fn(e, t) 更新变量 t
        return t
    return myFn
    # raise NotImplementedError("Need to implement for Task 0.3")


def sum(ls: Iterable[float]) -> float:
    "Sum up a list using `reduce` and `add`."
    # TODO: Implement for Task 0.3.
    # 0 是起始值
    return reduce(add, 0)(ls) # 从0开始，依次对元素求和
    # raise NotImplementedError("Need to implement for Task 0.3")


def prod(ls: Iterable[float]) -> float:
    "Product of a list using `reduce` and `mul`."
    # TODO: Implement for Task 0.3.
    return reduce(mul, 1)(ls) # 累积
    # raise NotImplementedError("Need to implement for Task 0.3")
```

```python
@pytest.mark.task0_3
@given(
    lists(small_floats, min_size=5, max_size=5),
    lists(small_floats, min_size=5, max_size=5),
)
def test_sum_distribute(ls1: List[float], ls2: List[float]) -> None:
    """
    Write a test that ensures that the sum of `ls1` plus the sum of `ls2`
    is the same as the sum of each element of `ls1` plus each element of `ls2`.
    """
    # TODO: Implement for Task 0.3.
    l1 = addLists(ls1, ls2)
    l2 = [x + y for x, y in zip(ls1, ls2)]
    for i in range(len(l1)):
        assert_close(l1[i], l2[i]) # 比较两个浮点数是否接近
    # raise NotImplementedError("Need to implement for Task 0.3")
```

![4.png](https://s2.loli.net/2024/02/16/p7jwtIo1VBiEJru.png)

### Task 0.4: Modules

```python
    def train(self):
        # 将当前模块及其所有子模块切换到训练模式，以便在模型训练时使用
        "Set the mode of this module and all descendent modules to `train`."
        # TODO: Implement for Task 0.4.
        self.training = True # 切换到训练模式
        for child_ in self._modules:
            self._modules[child_].train() # 对每个子模块调用 train 方法
        # raise NotImplementedError('Need to implement for Task 0.4')

    def eval(self):
        # 评估模型
        "Set the mode of this module and all descendent modules to `eval`."
        # TODO: Implement for Task 0.4.
        self.training = False
        for child_ in self._modules:
            self._modules[child_].eval()

    def named_parameters(self): 
        # 收集当前模块及其所有子模块的参数
        # 并以层次结构的方式返回参数的名称和对应的 Parameter 对象
        """
        Collect all the parameters of this module and its descendents.

        Returns:
            list of pairs: Contains the name and :class:`Parameter` of each ancestor parameter.
        """
        # TODO: Implement for Task 0.4.
        res = [] # 存储参数的名称和 Parameter 对象
        for k, v in self._parameters.items():
            res.append((k, v)) # 参数的名称 k 和 Parameter 对象 v 添加到列表 res 中

        for child_ in self._modules:
            child_params = self._modules[child_].named_parameters() # 获取子模块及其子模块的参数列表
            for item in child_params: # 将子模块的参数添加到 res 列表中
                res.append((child_ + '.' + item[0], item[1]))

        return res

        # raise NotImplementedError('Need to implement for Task 0.4')

    def parameters(self):
        # 遍历当前模块及其所有子模块的参数，并以列表形式返回这些参数
        "Enumerate over all the parameters of this module and its descendents."
        # TODO: Implement for Task 0.4.
        # 初始化为当前模块的 _parameters 字典中所有参数的值（Parameter 对象）构成的列表
        res = list(self._parameters.values())

        for child_ in self._modules:
            child_params = self._modules[child_].parameters()
            for item in child_params:
                res.append(item)
        return res
```

![5.png](https://s2.loli.net/2024/02/16/59L2ZIXSF6tuMrx.png)

### 结果

![6.png](https://s2.loli.net/2024/02/16/Ugny8P3vNsQCl41.png)

提交到Github上编译测试点都能过，但是环境配置那里编译不过，奇怪。。。

## module1

[Overview](https://minitorch.github.io/module1/module1/)

### Task 1.1: Numerical Derivatives

```python
def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    
    # TODO: Implement for Task 1.1.
    my_vals = list(vals)
    my_vals[arg] += epsilon 
    delta = f(*my_vals) - f(*vals)
    return delta / epsilon
    # raise NotImplementedError("Need to implement for Task 1.1")
```

![7.png](https://s2.loli.net/2024/02/16/c5V2fvKtYwejiaF.png)

### Task 1.2: Scalars

后面的代码和结果就不一一列举了

![8.png](https://s2.loli.net/2024/02/16/BpX1oQjWFvGS87z.png)

### Task 1.3: Chain Rule

![9.png](https://s2.loli.net/2024/02/16/Ih9Atv52Li3exMc.png)

### Task 1.4: Backpropagation

### Task 1.5: Training

提交到Github上面编译测试点也都能过