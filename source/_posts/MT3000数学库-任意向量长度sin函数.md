---
title: MT3000数学库-任意向量长度sin函数
date: 2024-02-02 22:28:26
tags:
categories:
    - HPC
---

[Github](https://github.com/ydyhello/studydemo/tree/main/helloSin)

> 编写MT3000设备端代码,为下面的kernel函数接口进行编程实现
```c
__global__ void kernel_evaluSin(uint64_t len,uint64_t coreNum,\
                                  double *optBuf,double *resBuf)
```
<!--more-->
- 使用AM缓存数据
- 使用向量Intrinsic编程
- 调用libvm向量sin函数`lvector double vm_sind16_u18(lvector double);`
- 完成设备端代码的编译/链接/dat文件生成
- 只需使用一个设备端线程
- 要求代码能处理数组长度不对齐的情况

## makefile
> 之前对makefile模模糊糊，这次从头完整的写了一遍makefile，感觉慢慢就熟悉了


- host端
```makefile
MT3k_ENV=/vol8/appsoftware/mt3000_programming_env-inbox/mt3000_programming_env-20230315
ENV_ROOT=${MT3k_ENV}/hthreads
MT_LIBVM=/vol8/home/hnu_ydy/libvm_expr/libvm_mt_public

EXE=helloSin.hos

ALL:
	gcc -O2 ${EXE}.c -std=c99 -I./ -I$(ENV_ROOT)/include -I$(MT_LIBVM)/include -I$(MT_LIBVM)/lib $(ENV_ROOT)/lib/libhthread_host.a -lpthread -fopenmp -lm -o ${EXE}
	

clean:
	rm ${EXE}
```
- device端
```makefile
MT3k_ENV=/vol8/appsoftware/mt3000_programming_env-inbox/mt3000_programming_env-20230315
ENV_ROOT=${MT3k_ENV}/hthreads
GCCROOT=${MT3k_ENV}/dsp_compiler
MT_LIBVM=/vol8/home/hnu_ydy/libvm_expr/libvm_mt_public

CC=MT-3000-gcc
AR=MT-3000-ar
LD=MT-3000-ld
AS=MT-3000-as
OB=MT-3000-objdump
DAT=MT-3000-makedat

export LD_LIBRARY_PATH=/vol8/appsoftware/mt3000_programming_env-inbox/mt3000_programming_env-20230315/third-party-lib/:$LD_LIBRARY_PATH

CFLAGS=-c -O2 -g -gdwarf-2 -fenable-m3000 -ffunction-sections -flax-vector-conversions -I./ -I$(ENV_ROOT)/include -I$(GCCROOT)/include/
LDFLAGS= -L$(ENV_ROOT)/lib --gc-sections -Tdsp.lds

SRC=helloSin.dev.c
OBJ=helloSin.dev.o
EXE=helloSin.dev.out
DAT=helloSin.dev.dat

ALL: $(EXE)
	$(GCCROOT)/bin/MT-3000-makedat -J $(EXE)


$(OBJ): $(SRC)
	$(GCCROOT)/bin/$(CC) -I$(MT_LIBVM)/include $(CFLAGS) $(SRC) -o $(OBJ) 

$(EXE): $(OBJ)
	$(GCCROOT)/bin/$(LD) $(LDFLAGS) $(OBJ) $(MT_LIBVM)/lib/libvm.a $(ENV_ROOT)/lib/libhthread_device.a $(GCCROOT)/lib/vlib3000.a $(GCCROOT)/lib/slib3000.a -o $(EXE) 

clean:
	rm $(EXE) $(OBJ) $(DAT)
```
## 设备端
```c
#include <compiler/m3000.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include "hthread_device.h"
#include "vector_math.h"

__global__ void kernel_evaluSin(uint64_t len,uint64_t coreNum,\
                                  double *optBuf,double *resBuf)
{   
    int core_id = get_thread_id();
    uint64_t offset = core_id * len;
    double *optBuf_fix = &optBuf[offset];
    double *resBuf_fix = &resBuf[offset];

    size_t dataNum = (16 * 1000 + 345 ) * 24 + 13;
    //lvector double *cache=(lvector double *)vector_malloc(cacheSize);
    lvector double * src1 = vector_malloc(len*sizeof(double));
    lvector double * src2 = vector_malloc(len*sizeof(double));
    
    vector_load(optBuf_fix,src1,len*sizeof(double));
    
    long i = 0;
    for(i = 0; i < dataNum/16; i++){
        src1[i] = (double)((i % 10000) - 5000);
        src2[i] = vm_asind16_u10(src1[i]);
    }

    vector_store(src2,resBuf_fix,len*sizeof(double));


    vector_free(src1);
	vector_free(src2);

}
```
## 主机端

```c
//check ulp
int comBuf_f64(double *optBuf, double *resBufH, double *resBufD, size_t bufLen, uint64_t maxDiff){
    size_t i;
    size_t errNum = 0;
    for(i = 0; i < bufLen; i++){
        double resH = resBufH[i];
        double resD = resBufD[i];
        uint64_t reResH = doubleToRawBits(resH);
        uint64_t reResD = doubleToRawBits(resD);
        uint64_t diff =  reResH > reResD ? reResH - reResD : reResD - reResH;
        if(diff >= maxDiff){
            fprintf(stdout, \
            "Error : {index : %lu, diff : %lu, opt : %lf, res of Hos : %lf(%016lx), res of Dev : %lf(%016lx)}\n",\
            i, diff, optBuf[i], resH, reResH, resD, reResD);
	    errNum++;
	}
    }
    if(errNum != 0){
    	fprintf(stdout, "Failed to test sin\n");
	return -1;
    }else{
    	fprintf(stdout, "PASS : sin\n");
	return 0;
    }
    
}

void testEvaluSin(size_t evaNum, int clusterId, char *program){
    hthread_dev_open(clusterId);
    hthread_dat_load(clusterId, program);
    size_t bufSize  = evaNum * sizeof(double);
    double *optBuf  = (double *)hthread_malloc(clusterId, bufSize, HT_MEM_RO);
    double *resBufD = (double *)hthread_malloc(clusterId, bufSize, HT_MEM_WO);
    double *resBufH = (double *)malloc(bufSize);
    int i;
    for(i = 0; i < evaNum; i++){
        optBuf[i] = (double)((i % 10000) - 5000);
        resBufH[i] = sin(optBuf[i]);
    }
    #if 1
    unsigned long int args[4];
    args[0] = (uint64_t)evaNum;
    args[1] = CORE_NUM;
    args[2] = (uint64_t)optBuf;
    args[3] = (uint64_t)resBufD;
    int threadId = hthread_group_create(clusterId, CORE_NUM, "kernel_evaluSin", 2, 2, args);
    #else
    unsigned long int args[3];
    args[0] = (uint64_t)evaNum;
    args[1] = (uint64_t)optBuf;
    args[2] = (uint64_t)resBufD;
    int threadId = hthread_group_create(clusterId, CORE_NUM, "kernel_evaluSinOnSingleCore", 1, 2, args);
    #endif

    hthread_group_wait(threadId);
    comBuf_f64(optBuf, resBufH, resBufD, evaNum, 2);
    hthread_dev_close(clusterId);
}

int main(int argc, char **argv){
    char *program = "helloSin.dev.dat";
    size_t dataNum = (16 * 1000 + 345 ) * 24 + 13;
    int clusterId = 0;
    if(argc > 2){
    	program = argv[1];
    }
    if(argc > 2){
    	dataNum = (size_t)atoi(argv[2]);
    }
    if(argc > 3){
    	clusterId = atoi(argv[3]);
    }
    fprintf(stdout, "dat : %s, clusterId : %d, dataNum : %lu\n", program, clusterId, dataNum);
    testEvaluSin(dataNum, clusterId, program);
    return 0;
}
```

## 结果
![image-20240124012613871.png](https://s2.loli.net/2024/02/23/8Pj1iEpbdnBIG2S.png)