---
title: BlackScholesKernel函数移植优化
date: 2024-02-02 22:51:05
tags:
categories:
    - HPC
---
[Github](https://github.com/ydyhello/studydemo/tree/main/BlackScholes_mt3000)

编写MT3000设备端代码
- 为`bsKernel.dev.c`的`BlackScholesKernel`函数进行优化
```c
__global__
void BlackScholesKernel(uint64_t optionCount, \
                                    float  R, \
                                    float  V, \
                               float *d_Call, \
                                float *d_Put, \
                                  float *d_S, \
                                  float *d_X, \
                                   float *d_T)
```
- 使用多线程编程/AM缓存数据/向量Intrinsic编程/异步DMA 等手段进行性能优化(至少使用前面3种优化方法)
- 使用MT-Libvm处理kernel函数中对超越函数的调用
- 能通过预提供的CPU端程序的正确性校验
- 至少获得较原始函数10x的性能提升
<!--more-->

> 向量Intrinsic的bug还没有解决，最后的结果只有多线程优化

## 设备端代码
```c
static void BlackScholesBodyCPU(
    float* call, //Call option price
    float* put,  //Put option price
    float Sf,    //Current stock price
    float Xf,    //Option strike price
    float Tf,    //Option years
    float Rf,    //Riskless rate of return
    float Vf,     //Stock volatility
    unsigned int i
){
    double S = Sf, X = Xf, T = Tf, R = Rf, V = Vf;

    double sqrtT = sqrt(T);
    double    d1 = (log(S / X) + (R + 0.5 * V * V) * T) / (V * sqrtT);
    double    d2 = d1 - V * sqrtT;
    double CNDD1 = CND(d1, i);
    double CNDD2 = CND(d2, i);

    //Calculate Call and Put simultaneously
    double expRT = exp(- R * T);
    *call = (float)(S * CNDD1 - X * expRT * CNDD2);
    *put  = (float)(X * expRT * (1.0 - CNDD2) - S * (1.0 - CNDD1));
}

__global__
void BlackScholesKernel(uint64_t optionCount, \
                                    float  R, \
                                    float  V, \
                               float *d_Call, \
                                float *d_Put, \
                                  float *d_S, \
                                  float *d_X, \
                                   float *d_T){
    //
    for(unsigned int i = 0; i < optionCount; i++){
        BlackScholesBodyCPU(
            &d_Call[i],
            &d_Put[i],
            d_S[i],
            d_X[i],
            d_T[i],
            R,
            V,
            i
        );
    }   
}

__global__
void BlackScholesthreads(uint64_t optionCount, \
                                    float  R, \
                                    float  V, \
                               float *d_Call, \
                                float *d_Put, \
                                  float *d_S, \
                                  float *d_X, \
                                   float *d_T){
    int threadId = get_thread_id();
    int threadsNum = get_group_size();
    uint64_t optionCount_p = optionCount/threadsNum;
    uint64_t extras = optionCount%threadsNum;
    uint64_t offset;
    if(threadId < extras){
        optionCount_p++;
        offset=threadId*optionCount_p;
    }else{
        offset=threadId*(optionCount_p+1)-(threadId-extras);
    }
    // BlackScholes(
    BlackScholesKernel(
            optionCount_p,
            R,
            V,
            d_Call+offset,
            d_Put+offset,
            d_S+offset,
            d_X+offset,
            d_T+offset
    );
}
```


## 结果对比
![image-20240126144156081.png](https://s2.loli.net/2024/02/23/u4MCwjnyFcJKmxP.png)

![image-20240126150606299.png](https://s2.loli.net/2024/02/23/XckgjeHLrRInVQE.png)