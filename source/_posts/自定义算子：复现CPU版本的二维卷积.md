---
title: 自定义算子：复现CPU版本的二维卷积
date: 2024-01-19 18:35:07
tags:
categories:
    - 学习
---


参考链接：[知乎](https://zhuanlan.zhihu.com/p/541302472)

## **搭建项目**

项目地址：[Github](https://github.com/ydyhello/studydemo/tree/main/MyConv2d%2Bcpu)

Windows环境

提前安装：`cl.exe`(参考：[配置cl](https://blog.csdn.net/HaoZiHuang/article/details/125795675))

## 编译

运行`python setup.py develop`，就能一键编译和安装。如果运行后没有报编译错误，就可以把实现的卷积用起来了

`setup.py`内容：

- `cpp_extension.CppExtension`：和编译相关的内容
  - 文件路径
  - 头文件搜索路径
- `name`：包的名称
- `cmdclass`：指定使用`cpp_extension.BuildExtension`类来构建C++扩展模块

如下是编译后的目录树

```
E:.
│  my_ops.cp311-win_amd64.pyd
│  setup.py
│  test.py
│
├─.vs
│  │  slnx.sqlite
│  │  VSWorkspaceState.json
│  │
│  └─panoflow
│      ├─FileContentIndex
│      │      4ea7c304-f0a5-4921-b532-67a89eef2f1e.vsidx
│      │      85545400-021b-45eb-a926-17f428dfafc7.vsidx
│      │      c1cd6aa0-353b-4942-b0b3-62d2e7989a6a.vsidx
│      │      cffc09b9-7625-4705-972e-360fc2b94695.vsidx
│      │      d975a3f7-8477-4464-9cbb-bafa4d00c5b8.vsidx
│      │
│      └─v17
├─.vscode
│      settings.json
│
├─build
│  ├─lib.win-amd64-cpython-311
│  │      my_ops.cp311-win_amd64.pyd
│  │
│  └─temp.win-amd64-cpython-311
│      └─Release
│          └─panoflow
│              └─core
│                  └─op
│                          my_conv.obj
│                          my_ops.cp311-win_amd64.exp
│                          my_ops.cp311-win_amd64.lib
│
├─panoflow
│  └─core
│      └─op
│          │  common_cuda_helper.hpp
│          │  my_conv.cpp
│          │  my_conv.py
│          │  pytorch_cpp_helper.hpp
│          │  pytorch_cuda_helper.hpp
│          │
│          └─__pycache__
│                  my_conv.cpython-311.pyc
│
└─panoflow.egg-info
        dependency_links.txt
        PKG-INFO
        SOURCES.txt
        top_level.txt
```

## 测试

运行`python test.py`

测试函数 `test_one` 和 `test_two`，用于测试单通道和多通道的情况

如果没有任何输出（报错信息），就说明卷积实现成功了

```python
def test_one():
    ts = torch.ones([1, 1, 3, 3]).to(device_name)
    layer = nn.Conv2d(1, 1, 3, 1, 1, bias=open_bias).to(device_name)
    gt = layer(ts)
    my_layer = MyConv2d(1, 1, 3, 1, 1).to(device_name)
    my_layer.load_state_dict(layer.state_dict(), strict=False)
    res = my_layer(ts)
    res = res.to('cpu')
    gt = gt.to('cpu')
    assert torch.allclose(res, gt, 1e-3, 1e-5)


def test_two():
    ts = torch.rand([1, inc, *img_shaspe]).to(device_name)
    layer = nn.Conv2d(inc, outc, 3, 1, 1, bias=open_bias).to(device_name)
    gt = layer(ts)
    my_layer = MyConv2d(inc, outc, 3, 1, 1).to(device_name)
    my_layer.load_state_dict(layer.state_dict(), strict=False)
    res = my_layer(ts)
    res = res.to('cpu')
    gt = gt.to('cpu')
    assert torch.allclose(res, gt, 1e-3, 1e-5)
```



## CPU实现过程

- 头文件：`pytorch_cpp_helper.hpp`,`pytorch_cuda_helper.hpp`,`common_cuda_helper.hpp`

### C++实现

- C++实现加法算子

算子的实现函数和C++接口绑定

```c++
#include <torch/torch.h>

torch::Tensor my_add(torch::Tensor t1, torch::Tensor t2)
{
 return t1 + t2;
}

TORCH_LIBRARY(my_ops, m)
{
 m.def("my_add", my_add);
}
```

- 卷积的实现`my_conv.cpp`

  - `my_conv_forward`是卷积的主函数

  ```c++
  void my_conv_forward(Tensor input, Tensor weight, Tensor bias,
                       Tensor output, Tensor columns, int kW,
                       int kH, int dW, int dH, int padW, int padH,
                       int dilationW, int dilationH, int group,
                       int im2col_step)
  ```

  - 先做`im2col`操作，再做了矩阵乘法

    - 使用循环，对每个批次的输入进行 im2col 操作，根据设备调用相应的 im2col 函数。
    - 将处理后的列矩阵进行形状变换，以便后续矩阵乘法操作。

    - 利用矩阵乘法（`addmm_`函数）更新输出缓冲区

  ```c++
  for (int elt = 0; elt < batchSize / im2col_step; elt++)
  {
      if (isCuda)
      {
          my_conv_im2col_cuda(input[elt], nInputPlane, inputHeight,
                          inputWidth, kH, kW, padH, padW, dH, dW, dilationH,
                          dilationW, im2col_step, columns);
      }
      else
      {
          my_conv_im2col_cpu(input[elt], nInputPlane, inputHeight,
                          inputWidth, kH, kW, padH, padW, dH, dW, dilationH,
                          dilationW, im2col_step, columns);
      }
      
  
      columns = columns.view({group, columns.size(0) / group, columns.size(1)});
      weight = weight.view({group, weight.size(0) / group, weight.size(1),
                            weight.size(2), weight.size(3)});
  
      for (int g = 0; g < group; g++)
      {
          output_buffer[elt][g] = output_buffer[elt][g]
                                      .flatten(1)
                                      .addmm_(weight[g].flatten(1), columns[g])
                                      .view_as(output_buffer[elt][g]);
      }
      columns =
          columns.view({columns.size(0) * columns.size(1), columns.size(2)});
      weight = weight.view({weight.size(0) * weight.size(1), weight.size(2),
                            weight.size(3), weight.size(4)});
  }
  ```

  - CPU 实现的 im2col (`my_conv_im2col_cpu`)

  ```c++
  void my_conv_im2col_cpu(Tensor data_im,
                          const int channels, const int height,
                          const int width, const int ksize_h,
                          const int ksize_w, const int pad_h, const int pad_w,
                          const int stride_h, const int stride_w,
                          const int dilation_h, const int dilation_w,
                          const int parallel_imgs, Tensor data_col)
  {
      // 计算 im2col 输出的高度和宽度
      int height_col =
          (height + 2 * pad_h - (dilation_h * (ksize_h - 1) + 1)) / stride_h + 1;
      int width_col =
          (width + 2 * pad_w - (dilation_w * (ksize_w - 1) + 1)) / stride_w + 1;
  
      // 计算要处理的总 kernel 数量
      int num_kernels = channels * height_col * width_col * parallel_imgs;
  
      // 使用 PyTorch 的宏 AT_DISPATCH_FLOATING_TYPES_AND_HALF 遍历浮点数类型和半精度类型
      AT_DISPATCH_FLOATING_TYPES_AND_HALF(
          data_im.scalar_type(), "", [&]
          {
              // 调用具体的 im2col 核函数，根据不同的数据类型执行不同的实现
              my_conv_im2col_cpu_kernel<scalar_t>(
                  num_kernels, data_im.data_ptr<scalar_t>(),
                  height, width, ksize_h, ksize_w,
                  pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
                  parallel_imgs, channels,
                  height_col, width_col, data_col.data_ptr<scalar_t>());
          });
  }
  
  ```

### Python封装

- 调用

```python
import my_ops
my_ops.my_conv_forward(...)
```

- 使用 Pybind11 封装 C++ 函数

```c++
PYBIND11_MODULE(my_ops, m)
{
      m.def("my_conv_forward", my_conv_forward, "my_conv_forward",
            py::arg("input"), py::arg("weight"), py::arg("bias"),
            py::arg("output"), py::arg("columns"), py::arg("kW"),
            py::arg("kH"), py::arg("dW"), py::arg("dH"), py::arg("padW"),
            py::arg("padH"), py::arg("dilationW"), py::arg("dilationH"),
            py::arg("group"), py::arg("im2col_step"));
}
```

通过 Pybind11，可以在 Python 中直接调用名为 `my_conv_forward` 的函数，并将参数传递给底层的 C++ 实现，完成卷积操作

- `MyConvF` 类和`MyConv2d` 类

```python
import torch
from torch.autograd import Function
import torch.nn as nn
from torch import Tensor
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter

import my_ops


class MyConvF(Function):


    def forward(ctx,
                input: torch.Tensor,
                weight,
                bias,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                im2col_step=32):
        if input is not None and input.dim() != 4:
            raise ValueError(
                f'Expected 4D tensor as input, got {input.dim()}D tensor \
                  instead.')
        ctx.stride = _pair(stride)
        ctx.padding = _pair(padding)
        ctx.dilation = _pair(dilation)
        ctx.groups = groups
        ctx.im2col_step = im2col_step

        weight = weight.type_as(input)
        ctx.save_for_backward(input, weight)

        output = input.new_empty(MyConvF._output_size(ctx, input, weight))

        ctx.bufs_ = [input.new_empty(0), input.new_empty(0)]  # columns, ones

        cur_im2col_step = min(ctx.im2col_step, input.size(0))
        assert (input.size(0) % cur_im2col_step
                ) == 0, 'batch size must be divisible by im2col_step'

        my_ops.my_conv_forward(
            input,
            weight,
            bias,
            output,
            ctx.bufs_[0],
            kW=weight.size(3),
            kH=weight.size(2),
            dW=ctx.stride[1],
            dH=ctx.stride[0],
            padW=ctx.padding[1],
            padH=ctx.padding[0],
            dilationW=ctx.dilation[1],
            dilationH=ctx.dilation[0],
            group=ctx.groups,
            im2col_step=cur_im2col_step)
        return output


    def _output_size(ctx, input, weight):
        channels = weight.size(0)
        output_size = (input.size(0), channels)
        for d in range(input.dim() - 2):
            in_size = input.size(d + 2)
            pad = ctx.padding[d]
            kernel = ctx.dilation[d] * (weight.size(d + 2) - 1) + 1
            stride_ = ctx.stride[d]
            output_size += ((in_size + (2 * pad) - kernel) // stride_ + 1, )
        if not all(map(lambda s: s > 0, output_size)):
            raise ValueError(
                'convolution input is too small (output would be ' +
                'x'.join(map(str, output_size)) + ')')
        return output_size


my_conv = MyConvF.apply


class MyConv2d(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups: int = 1,
                 bias: bool = True):
        super().__init__()
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = _pair(padding)
        dilation_ = _pair(dilation)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size_
        self.stride = stride_
        self.padding = padding_
        self.dilation = dilation_
        self.groups = groups
        self.weight = Parameter(
            torch.Tensor(out_channels, in_channels // groups, *kernel_size_))
        self.bias = Parameter(torch.Tensor(out_channels))

        # Useless attributes
        self.transposed = None
        self.output_padding = None
        self.padding_mode = None

    def forward(self, input: Tensor) -> Tensor:
        return my_conv(input, self.weight, self.bias, self.stride,
                       self.padding, self.dilation, self.groups)
```

