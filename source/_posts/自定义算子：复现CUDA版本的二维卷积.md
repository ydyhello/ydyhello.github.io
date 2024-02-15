---
title: 自定义算子：复现CUDA版本的二维卷积
date: 2024-02-16 00:34:55
tags:
categories:
    - Pytorch算子
---

参考链接：[知乎](https://zhuanlan.zhihu.com/p/541302472)

## **搭建项目**

项目地址：[Github](https://github.com/ydyhello/studydemo/tree/main/MyConv2d%2Bcuda)

Windows环境

提前安装:
- NVIDIA GPU Computing Toolkit
- `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117`

## 编译
运行`python setup.py develop`，就能一键编译和安装。如果运行后没有报编译错误，就可以把实现的卷积用起来了

![bianyi1.png](https://s2.loli.net/2024/02/16/VLdQk6rZn4hpt3I.png)

`python setup.py develop`编译
把源文件加入cpp_src里。之后，把CppExtension改成CUDAExtension。

```python
from setuptools import setup
from torch.utils import cpp_extension
import os

src_root = 'panoflow_cuda/core/op/'
cpp_src = ['my_conv.cpp', 'my_conv_cuda.cu']

if __name__ == '__main__':
    include_dirs = ['panoflow_cuda/core/op/']
    cpp_path = [os.path.join(src_root, src) for src in cpp_src]

    setup(
        name='panoflow_cuda',
        ext_modules=[
            cpp_extension.CUDAExtension(
                'my_ops', cpp_path, include_dirs=include_dirs)
        ],
        cmdclass={'build_ext': cpp_extension.BuildExtension})
```
## CUDA实现
> 和CPU实现基本一样 [自定义算子：复现CPU版本的二维卷积](https://ydyhello.github.io/2024/01/19/%E8%87%AA%E5%AE%9A%E4%B9%89%E7%AE%97%E5%AD%90%EF%BC%9A%E5%A4%8D%E7%8E%B0CPU%E7%89%88%E6%9C%AC%E7%9A%84%E4%BA%8C%E7%BB%B4%E5%8D%B7%E7%A7%AF/)

**但是在CUDA版本中，要实现`my_conv_im2col_cuda`这个函数**

`my_conv_cuda.cu`代码：
```C++
// Modify from https://github.com/open-mmlab/mmcv/blob/my_conv/mmcv/ops/csrc/common/cuda/deform_conv_cuda_kernel.cuh
// Copyright (c) OpenMMLab. All rights reserved.
#include <torch/types.h>

#include "pytorch_cuda_helper.hpp"

template <typename T>
__global__ void my_conv_im2col_gpu_kernel(
    const int n, const T *data_im, const int height,
    const int width, const int kernel_h, const int kernel_w, const int pad_h,
    const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int batch_size,
    const int num_channels, const int height_col,
    const int width_col, T *data_col)
{
    CUDA_1D_KERNEL_LOOP(index, n)
    {
        // index index of output matrix
        const int w_col = index % width_col;
        const int h_col = (index / width_col) % height_col;
        const int b_col = (index / width_col / height_col) % batch_size;
        const int c_im = (index / width_col / height_col) / batch_size;
        const int c_col = c_im * kernel_h * kernel_w;

        const int h_in = h_col * stride_h - pad_h;
        const int w_in = w_col * stride_w - pad_w;
        T *data_col_ptr =
            data_col +
            ((c_col * batch_size + b_col) * height_col + h_col) * width_col + w_col;
        const T *data_im_ptr =
            data_im + (b_col * num_channels + c_im) * height * width;

        for (int i = 0; i < kernel_h; ++i)
        {
            for (int j = 0; j < kernel_w; ++j)
            {
                T val = static_cast<T>(0);
                const int h_im = h_in + i * dilation_h;
                const int w_im = w_in + j * dilation_w;
                if (h_im > -1 && w_im > -1 && h_im < height && w_im < width)
                {
                    val = data_im_ptr[h_im * width + w_im];
                }
                *data_col_ptr = val;
                data_col_ptr += batch_size * height_col * width_col;
            }
        }
    }
}

void my_conv_im2col_cuda(Tensor data_im,
                         const int channels, const int height,
                         const int width, const int ksize_h,
                         const int ksize_w, const int pad_h, const int pad_w,
                         const int stride_h, const int stride_w,
                         const int dilation_h, const int dilation_w,
                         const int parallel_imgs, Tensor data_col)
{
    int height_col =
        (height + 2 * pad_h - (dilation_h * (ksize_h - 1) + 1)) / stride_h + 1;
    int width_col =
        (width + 2 * pad_w - (dilation_w * (ksize_w - 1) + 1)) / stride_w + 1;
    int num_kernels = channels * height_col * width_col * parallel_imgs;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        data_im.scalar_type(), "my_conv_im2col_gpu", [&]
        { my_conv_im2col_gpu_kernel<scalar_t><<<GET_BLOCKS(num_kernels),
                                                THREADS_PER_BLOCK, 0,
                                                at::cuda::getCurrentCUDAStream()>>>(
              num_kernels, data_im.data_ptr<scalar_t>(),
              height, width, ksize_h, ksize_w,
              pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
              parallel_imgs, channels,
              height_col, width_col, data_col.data_ptr<scalar_t>()); });

    AT_CUDA_CHECK(cudaGetLastError());
}
```

## 测试
把`device_name`改成`cuda:0`
```python
import torch
import torch.nn as nn
from panoflow_cuda.core.op.my_conv import MyConv2d

inc = 3
outc = 4
img_shaspe = (50, 50)

device_name = 'cuda:0'
# device_name = 'cpu'
open_bias = True


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


if __name__ == '__main__':
    test_one()
    test_two()
```
没报错即可
