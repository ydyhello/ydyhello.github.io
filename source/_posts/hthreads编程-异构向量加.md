---
title: hthreads编程-异构向量加
date: 2024-02-02 22:21:01
tags:
categories:
    - HPC
---
## 熟悉mt3000编程环境目录结构

```
hnu_ydy@ln6:/vol8/appsoftware/mt3000_programming_env$ ls
bios          kernel                                   mt3000.update
bug.xlsx      libmt                                    mt_gdb
ChangeLog     link2m3000.sh                            README
driver        M3000-gcc                                test_bench
dsp_compiler  M3000-makedat                            third-party-lib
hthreads      mt3000_programming_env_update_temple.sh  tools
install.info  mt3000.setup
```

- ChangeLog文件：记录了更新日志。最新日志的时间作为版本号。
- tools目录
- third-party-lib目录

## 熟悉dsp端makefile

### host_code

```makefile
ENV_ROOT=/vol8/appsoftware/mt3000_programming_env/hthreads

EXE=copy_host
ALL:
	gcc -O2 ${EXE}.c -std=c99 -I./ -I$(ENV_ROOT)/include $(ENV_ROOT)/lib/libhthread_host.a -lpthread -fopenmp -o ${EXE}
	cp ${EXE} ../bin/

clean:
	rm ${EXE}
```

- 头文件路径`-I./ -I$(ENV_ROOT)/include`
- 库的路径`$(ENV_ROOT)/lib/libhthread_host.a`
- 依赖pthread库`-lpthread`

### device_code

```makefile
GCCROOT=/vol8/appsoftware/mt3000_programming_env/dsp_compiler
ENV_ROOT=/vol8/appsoftware/mt3000_programming_env/hthreads
CC=MT-3000-gcc
AR=MT-3000-ar
LD=MT-3000-ld

export LD_LIBRARY_PATH=/vol8/appsoftware/mt3000_programming_env/third-party-lib:$LD_LIBRARY_PATH
CFLAGS=-c -O2 -g -gdwarf-2 -fenable-m3000 -ffunction-sections -flax-vector-conversions -I./ -I$(ENV_ROOT)/include -I$(GCCROOT)/include/
LDFLAGS= -L$(ENV_ROOT)/lib --gc-sections -Tdsp.lds
SRC=copy_kernel.c
OBJ=copy_kernel.o
EXE=copy_kernel.out
DAT=copy_kernel.dat

ALL: $(EXE)
	$(GCCROOT)/bin/MT-3000-makedat -J $(EXE)
	cp $(DAT) ../bin

$(OBJ): $(SRC)
	$(GCCROOT)/bin/$(CC) $(CFLAGS) $(SRC) -o $(OBJ) 

$(EXE): $(OBJ)
	$(GCCROOT)/bin/$(LD) $(LDFLAGS) $(OBJ) $(ENV_ROOT)/lib/libhthread_device.a $(GCCROOT)/lib/vlib3000.a $(GCCROOT)/lib/slib3000.a -o $(EXE) 

clean:
	rm $(EXE) $(OBJ) $(DAT)
```

- gcc编译器依赖库`export LD_LIBRARY_PATH=/vol8/appsoftware/mt3000_programming_env/third-party-lib:$LD_LIBRARY_PATH`

## 使用查看mt模块状态

> 查看可用结点

```
hnu_ydy@ln6:/vol8/appsoftware/mt3000_programming_env$ yhi
PARTITION AVAIL  TIMELIMIT  NODES  STATE NODELIST
TNG          up   infinite      3  drain cn[7319-7321]
TNG          up   infinite      6  alloc cn[7322,7324-7327,7460]
TNG          up   infinite    291   idle cn[7168-7318,7323,7328-7459,7461-7467]
```

> 查看加载的模块

- `-p`：指定分区名
- `-N`：1个结点
- `-n`：1个进程
- `-w`：指定结点

```
hnu_ydy@ln6:/vol8/appsoftware/mt3000_programming_env$ yhrun -p TNG -N 1 -n 1 -w cn7329 lsmod
Module                  Size  Used by
mgc                   118784  1
lustre               1110016  241
lmv                   233472  2 lustre
mdc                   286720  5 lustre
fid                    40960  1 mdc
lov                   368640  163 mdc,lustre
fld                    53248  2 lov,lmv
osc                   471040  341 mdc
ksocklnd              196608  1
ptlrpc               1585152  8 fld,osc,fid,mgc,lov,mdc,lmv,lustre
obdclass             1249280  174 fld,osc,fid,ptlrpc,mgc,lov,mdc,lmv,lustre
lnet                  790528  7 osc,obdclass,ptlrpc,mgc,ksocklnd,lmv,lustre
libcfs                294912  12 fld,lnet,osc,fid,obdclass,ptlrpc,mgc,ksocklnd,lov,mdc,lmv,lustre
zni_net                40960  0
zni_dev                98304  1 zni_net
crc32_generic          16384  0
eccintr                20480  0
xpmem                  45056  0
knem                   49152  0
mt                     40960  0
sunrpc                425984  2 lnet
ip_tables              32768  0
x_tables               53248  1 ip_tables
```

> mt模块

```
mt                     40960  0
```

0表示目前模块没有被任何进程占用，上面所有设备都是可用的

## 使用free查看可用内存

`free -h`

释放1GB mt驱动预留内存

> mt驱动预留了多大内存

```
hnu_ydy@ln6:/vol8/appsoftware/mt3000_programming_env$ yhrun -p TNG -N 1 -n 1 -w cn7329 free -h
              total        used        free      shared  buff/cache   available
Mem:           61Gi        52Gi       8.2Gi       525Mi       542Mi       8.0Gi
```

四个簇占用52G，每个dsp簇最大可用13G，申请dsp容量不要超过13G

后端可用8.2G

## 使用工具查看dsp状态

查看dsp 当前PC值

```
/vol8/appsoftware/mt3000_programming_env/tools/get_dsp_pc
```

```
hnu_ydy@ln6:/vol8/appsoftware/mt3000_programming_env$ yhrun -p TNG -N 1 -n 1 -w cn7329 /vol8/appsoftware/mt3000_programming_env/tools/get_dsp_pc
get_dsp_pc <id> [<core>], returns 64-bit pc, if core not specified, print 24 cores' pc
yhrun: error: cn7329: task 0: Exited with exit code 255
```

\<id>：dsp簇号

[\<core>]：查看哪个核的pc值

```
hnu_ydy@ln6:/vol8/appsoftware/mt3000_programming_env$ yhrun -p TNG -N 1 -n 1 -w cn7329 /vol8/appsoftware/mt3000_programming_env/tools/get_dsp_pc 0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
```

## 编写异构向量加程序

目的：熟悉hthreads编程

修改`GCCROOT`，`ENV_ROOT`，`LD_LIBRARY_PATH`

```makefile
GCCROOT=/vol8/appsoftware/mt3000_programming_env/dsp_compiler
ENV_ROOT=/vol8/appsoftware/mt3000_programming_env/hthreads
export LD_LIBRARY_PATH=/vol8/appsoftware/mt3000_programming_env/third-party-lib:$LD_LIBRARY_PATH
```

device端

```
hnu_ydy@ln6:~/copy_test/device_code$ make
/vol8/appsoftware/mt3000_programming_env/dsp_compiler/bin/MT-3000-gcc -c -O2 -g -                  gdwarf-2 -fenable-m3000 -ffunction-sections -flax-vector-conversions -I./ -I/vol8                  /appsoftware/mt3000_programming_env/hthreads/include -I/vol8/appsoftware/mt3000_p                  rogramming_env/dsp_compiler/include/ copy_kernel.c -o copy_kernel.o
/vol8/appsoftware/mt3000_programming_env/dsp_compiler/bin/MT-3000-ld -L/vol8/apps                  oftware/mt3000_programming_env/hthreads/lib --gc-sections -Tdsp.lds copy_kernel.o                   /vol8/appsoftware/mt3000_programming_env/hthreads/lib/libhthread_device.a /vol8/                  appsoftware/mt3000_programming_env/dsp_compiler/lib/vlib3000.a /vol8/appsoftware/                  mt3000_programming_env/dsp_compiler/lib/slib3000.a -o copy_kernel.out
/vol8/appsoftware/mt3000_programming_env/dsp_compiler/bin/MT-3000-makedat -J copy                  _kernel.out
cp copy_kernel.dat ../bin
```

```
hnu_ydy@ln6:~/copy_test/host_code$ cd ../bin/
hnu_ydy@ln6:~/copy_test/bin$ ls
copy_host  copy_kernel.dat  run.sh
hnu_ydy@ln6:~/copy_test/bin$ cat run.sh
#!/bin/bash

./copy_host 0 1024 24
```

运行脚本三个参数

- 簇id
- 拷贝的数组的长度
- 要用多少个dsp核

只用1个核

```
hnu_ydy@ln6:~/copy_test/bin$ yhrun -p TNG -N 1 -n 1 ./copy_host 0 1024 1
[Core 0] Start copy
Success!
```

24核

```
hnu_ydy@ln6:~/copy_test/bin$ yhrun -p TNG -N 1 -n 1 ./copy_host 0 1024 24
[Core 0] Start copy
[Core 1] Start copy
[Core 2] Start copy
[Core 3] Start copy
[Core 4] Start copy
[Core 5] Start copy
[Core 6] Start copy
[Core 7] Start copy
[Core 8] Start copy
[Core 9] Start copy
[Core 10] Start copy
[Core 11] Start copy
[Core 12] Start copy
[Core 13] Start copy
[Core 14] Start copy
[Core 15] Start copy
[Core 16] Start copy
[Core 17] Start copy
[Core 18] Start copy
[Core 19] Start copy
[Core 20] Start copy
[Core 21] Start copy
[Core 22] Start copy
[Core 23] Start copy
Success!
```

插入内联汇编

在`copy_kernel.c`加入

```c
asm("swait");
```

执行到这条语句会停机

### 设备端

将数据从A拷贝到C，`copy_kernel.c`

```c
#include <compiler/m3000.h>
#include "hthread_device.h"
/*
 * copy - native version
 * */
__global__ void  native_copy(unsigned long length,
		unsigned long coreNum,
		long *A,
		long *C)
{
	int tid = get_thread_id();
	long i = 0;
	unsigned long offset = tid * length;
	long *A_fixed = &A[offset];
	long *C_fixed = &C[offset];

	hthread_printf("[Core %d] Start copy\n", tid);
	for (i = 0; i < length; i ++)
		C_fixed[i] = A_fixed[i];
   //asm("swait"); //内联汇编
}
```

### 主机端

`copy_host.c`

```c
#include<stdio.h>
#include<stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>
#include "hthread_host.h"

#define NUM_CORES 24
#define NUM_TESTS 1

int main(int argc, char **argv){

	if (argc < 4) {
		printf("Useage: ~ dev_id{0, 1, 2, 3} length num_threads{1...24}\n");
		return -1;
	}

	int dev_id= atoi(argv[1]);
	long length = ((atol(argv[2]) +15)/ 16) * 16;
	int num_threads  = atoi(argv[3]);

	long size = length * num_threads * sizeof (long);
	long total_len = length * num_threads;

	hthread_dev_open(dev_id);
	hthread_dat_load(dev_id, "copy_kernel.dat");

	long *A = hthread_malloc(dev_id, size, HT_MEM_RO);
	long *C = hthread_malloc(dev_id, size, HT_MEM_RO);
	long *C_ref = malloc(size);

	for(int i=0;i<total_len;i++)
	{
		A[i]= i;
		C[i] = 0;
		C_ref[i] = A[i];
	}

	unsigned long args[4];
	args[0] = length;
	args[1] = num_threads;
	args[2] = (unsigned long)A;
	args[3] = (unsigned long)C;

	int tg_id = hthread_group_create(dev_id, num_threads);
	hthread_group_wait(tg_id);

	for(int i=0; i<NUM_TESTS; i++){
		hthread_group_exec(tg_id, "native_copy", 2, 2, args);
		hthread_group_wait(tg_id);

	}

	int error_num = 0;
#pragma omp parallel for reduction(+:error_num) num_threads(8)
	for(long i=0;i<total_len;i++) 	{
		if (C[i] != C_ref[i]) {
			//printf("C[%ld] = %ld\tC_ref[%ld] = %ld\n", i, C[i], i, C_ref[i]); 
			error_num ++;
		}
	}
	if (error_num > 0)
		printf("Result Error!\n");
	else 
		printf("Success!\n");


	hthread_group_destroy(tg_id);
	hthread_free(A);
	hthread_free(C);
	hthread_dev_close(dev_id);
	free(C_ref);

	return 0;
	}
```



### 改成向量加运算

`add_host.c`

```c
#include<stdio.h>
#include<stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>
#include "hthread_host.h"

#define NUM_CORES 24
#define NUM_TESTS 1

int main(int argc, char **argv){

	if (argc < 4) {
		printf("Useage: ~ dev_id{0, 1, 2, 3} length num_threads{1...24}\n");
		return -1;
	}

	int dev_id= atoi(argv[1]);
	long length = ((atol(argv[2]) +15)/ 16) * 16;
	int num_threads  = atoi(argv[3]);

	long size = length * num_threads * sizeof (long);
	long total_len = length * num_threads;

	hthread_dev_open(dev_id);
	hthread_dat_load(dev_id, "copy_kernel.dat");

	long *A = hthread_malloc(dev_id, size, HT_MEM_RO);
    long *B = hthread_malloc(dev_id, size, HT_MEM_RO);
	long *C = hthread_malloc(dev_id, size, HT_MEM_RO);
	long *C_ref = malloc(size);

	for(int i=0;i<total_len;i++)
	{
		A[i]= i;
    B[i] = i;
		C[i] = 0;
		C_ref[i] = A[i]+B[i];
	}

	unsigned long args[5];
	args[0] = length;
	args[1] = num_threads;
	args[2] = (unsigned long)A;
    args[3] = (unsigned long)B;
	args[4] = (unsigned long)C;

	int tg_id = hthread_group_create(dev_id, num_threads);
	hthread_group_wait(tg_id);

	for(int i=0; i<NUM_TESTS; i++){
		hthread_group_exec(tg_id, "native_copy", 2, 3, args);
		hthread_group_wait(tg_id);

	}

	int error_num = 0;
#pragma omp parallel for reduction(+:error_num) num_threads(8)
	for(long i=0;i<total_len;i++) 	{
		if (C[i] != C_ref[i]) {
			//printf("C[%ld] = %ld\tC_ref[%ld] = %ld\n", i, C[i], i, C_ref[i]); 
			error_num ++;
		}
	}
	if (error_num > 0)
		printf("Result Error!\n");
	else 
		printf("Success!\n");


	hthread_group_destroy(tg_id);
	hthread_free(A);
    hthread_free(B);
	hthread_free(C);
	hthread_dev_close(dev_id);
	free(C_ref);

	return 0;
	}
```

`add_kernel.c`

```c
#include <compiler/m3000.h>
#include "hthread_device.h"
/*
 * copy - native version
 * */
__global__ void  native_copy(unsigned long length,
		unsigned long coreNum,
		long *A,
        long *B,
		long *C)
{
	int tid = get_thread_id();
	long i = 0;
	unsigned long offset = tid * length;
	long *A_fixed = &A[offset];
    long *B_fixed = &B[offset];
	long *C_fixed = &C[offset];

	hthread_printf("[Core %d] Start add\n", tid);
	for (i = 0; i < length; i ++)
     //C_fixed[i] = A_fixed[i];
		C_fixed[i] = A_fixed[i] + B_fixed[i];
}

```

### 分析向量加运算中时间开销

使用`get clk()`函数，分析向量加运算中读取ddr数据的时间开销以及加法运算时间开销

```c
   unsigned long t1=0,t2=0,t3=0,t4=0,t5 =0;
  int thidx= get_thread_id();
  t1 = get_clk();
  
  

	hthread_printf("[Core %d] Start add\n", tid);
	for (i = 0; i < length; i ++)
     //C_fixed[i] = A_fixed[i];
		C_fixed[i] = A_fixed[i] + B_fixed[i];
   
   t2 = get_clk();
   
   hthread_printf("[Core %d] t1 = %lu\n", tid, t1);
   hthread_printf("[Core %d] t2 = %lu\n", tid, t2);
```

![image-20240123200930435.png](https://s2.loli.net/2024/02/02/EsKSTXlvWoDZfec.png)

