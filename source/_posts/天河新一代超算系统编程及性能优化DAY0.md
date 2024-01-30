---
title: 天河新一代超算系统编程及性能优化DAY0
date: 2024-01-21 15:20:05
tags:
categories:
    - HPC
---

>培训前熟悉一下工具

## Linux常用命令

### 机器信息

- 用户名称：当前使用的账户

```sh
whoami
```

- 服务器名称：当前所使用的节点

```shell
hostname
```

- 机器信息：基本硬件配置

```shell
lscpu
```

- 内存信息

```shell
free
```

- 监视进程和Linux整体性能

```shell
top
```

### 文件和目录

- 查看指定目录下所有文件

```shell
ls
```

- 更改文件权限

```shell
chmod [para] [filename]
```

- 目录的创建，目录以及文件的删除

```shell
mkdir [directoryName]
rm [-rf] [filename or directoryName]
```

- 查看当前目录 (路径)

```shell
pwd
```

- 目录切换

```shell
cd [absolutePath]
cd .. 上级目录
cd .  当前目录
cd -  上次访问目录
cd ~  用户根目录
```

- 文件的创建 (空文件)

```shell
touch [filename]
```

- 文件/目录的移动

```shell
mv [filename] [object_directiryName]
```

- 文件/目录的重命名

```shell
mv [old_filename] [new_filename]
```

- 文件/目录的拷贝

```shell
cp [source_filename] [dest_filename]
```

- 文件字符搜索：grep (在文件中找字符串)

```shell
grep "[string]" -r [filename]
```

- 文件和目录查找：find (在目录中找文件/文件夹)

```shell
find [directory] -name [filename]
```

- 显示文件全部内容

```shell
cat [file_name] 将文件整个内容从上到下显示在屏幕
```

- 显示文件开头

```shell
head [file_name]
head -n [line_num] [file_name] 指定行数
```

- 显示文件结尾

```shell
tail [file_name]
tail -n [line_num] [file_name]
tail -f [file_name] 动态刷新文件末尾
```

- 显示文件结尾

```shell
more [file_name]
```

### 环境变量

- 环境变量的查看、设置

```shell
env       #用来显示环境变量, 显示当前用户的环境变量
export    #用来设置环境变量
echo      #用来查看指定变量内容
```

- 将/home/user/bin路径正确加入到PATH环境变量中

```shell
export PATH=/home/user/bin:$PATH
export PATH=$PATH:/home/user/bin
export PATH=${PATH}:/home/user/bin
```

- 可执行共享库(动态库)的目录路径

```sh
LD_LIBRARY_PATH=/usr/local/lib:
```

## 文件编辑器 Vim

- 命令行式编辑器：Vim（通过终端对文件进行查看、编写和保存）

```shell
vim [filename]
```

- Vim 编辑环境下的四种常用模式
  - 正常模式 (Normal-mode)
  - 插入模式 (Insert-mode)
  - 命令模式 (Command-mode)
  - 可视模式 (Visual-mode)

[菜鸟教材](https://www.runoob.com/linux/linux-vim.html)

## 编译器 GCC

### 程序编译流程

编译过程：预处理、编译、链接

- 预处理：处理 .c 文件的 #define，#include 等预处理指令
- 编译：把高级语言 (.c) 翻译成汇编指令 (.s)，再翻译成机器码 (.o)
- 链接：将所有目标文件 (.o) 和第三方库文件，链接成二进制文件 (.exe)

### gcc命令的建议编译规则

```shell
gcc -o [binaryfile] [sourcefile1] [sourefile2]…

gcc -o HelloWorld main.c # 编译处可执行文件
./HelloWorld # 执行
```

### gcc 编译链接命令规则及常用编译选项

编译与链接

示例：需要编译 `main.c`、`kernel.c` 成 exe，其使用了 `/home/opt/` 路径下的第三方库 math，其中 opt 目录下包含：`/include/mymath.h`、`/lib/mymath.so`

- 分步进行编译和链接

  - 编译 (将源文件逐个编译成目标文件)

  ```shell
  gcc -o [obj_file] -c [src_file] -I [include_path]
  
  gcc -c main.c -o main.o
  gcc -c kernel.c -o kernel.o -I /home/opt/include
  ```

  - 链接 (将所有目标文件，以及第三方库，链接成二进制文件)

  ```shell
  gcc -o [bin] [all_obj_files] -L [library_path] -l [library_file_name]
  
  gcc -o exe main.o kernel.o -L /home/opt/lib -l mymath
  ```

- 直接编译出二进制文件

  ```shell
  gcc -o [bin] [all_src_files] -I [include_path] -L [library_path] -l [library_file_name]
  
  gcc -o exe main.c kernel.c -I /home/opt/include -L /home/opt/lib -l mymath
  ```

编译选项:

- 优化等级选项：

`-O0`，`-O1`，`-O2`，`-O3` 由编译器自动进行代码性能优化，等级越大，程序运行速度越快

- 警告选项：

`-Wall` 显示所有警告

- 调试选项：

`-g` 通常结合 -O0 优化等级编译，后期可使用 gdb 工具对二进制文件进行调试。会降低程序计算速度

- 性能分析选项：

`-pg` 后期可用于 gprof 工具对二进制文件进行性能分析。会降低程序计算速度

- 宏定义选项

`-D` 对应着程序代码中的宏定义。如 -DUSE_MPI

## 工程构建工具 Make

GNU Make 是最基本的项目构建工具，Make 通过 Makefile 文件，获取如何构建执行文件信息，并按照 Makefile 里面设定的规则，根据源文件的依赖关系，调用编译器自动执行编译

### 编写 makefile

```makefile
BIN=hello                     #"="定义变量BIN: 要生成的二进制文件名称为"hello"
SRC=hello.cpp                 
all:$(BIN)                    #all为make默认执行目标, 这个目标所依赖文件: 二进制文件BIN  
$(BIN):$(SRC)                 #二进制文件BIN依赖于源码文件SRC 
	g++ -o $(BIN) $(SRC)      #建立SRC到BIN的规则
clean:                        #clean 目标不依赖任何其他, 执行时将执行文件删除命令
	rm -rf $(BIN) $(OBJ)

```

## 作业管理系统

### 常用用户命令

- `yhinfo / yhi`：资源信息查询
- `yhalloc`：资源申请 (强占)
- `yhrun`：作业提交 (自动申请资源)
- `yhqueen / yhq`：作业队列查询
- `yhcancel`：作业取消
- `yhbatch`：批处理作业
- `yhacct`：作业历史查询

### 查看系统信息

```
hnu_ydy@ln6:~/test$ yhinfo
PARTITION AVAIL  TIMELIMIT  NODES  STATE NODELIST
TNG          up   infinite      1  drain cn7319
TNG          up   infinite    299   idle cn[7168-7318,7320-7467]
```

- PARTITION 分区
- TIMELIMIT 时间限制
- NODES 节点数量
- STATE运行状态

### 作业提交

- `yhrun`：交互式命令行直接提交作业，**在提交节点上运行，关闭当前登录终端会导致程序退出。**仅用于测试，避免以此方式提交正式作业任务。

```shell
yhrun -p hpc_test -N 2 -n 32 --mpi=pmix vasp_std
```

- `yhalloc`：先获取可用节点资源，再在分配节点上提交任务。用户的作业脚本直接**在提交节点上运行，关闭当前登录终端会导致程序退出。**

- `yhbatch`：批处理方式提交用户作业脚本，在分配的第一个节点上运行，**关闭当前登录终端不会影响作业的运行**。建议用户使用 `yhbatch` 提交作业

首先编写 `slurm` 脚本，设置用于作业的**局部变量**和**环境变量**；添加提交作业的 `yhrun` 指令，并指定**节点数**、**核数**、**分区**以及**作业的可执行文件**等。

`submit_cp2k.sh`

```shell
#!/bin/bash
#SBATCH --job-name=cp2k_submit_test	#作业名
#SBATCH --exclusive	#不共享

route=$PWD
cp2k_exe=/vol8/appsoftware/cp2k/2022.2/exe/local/cp2k.popt

INPUTFILE=$route
```

最后使用 `yhbatch` 提交前述的 `slurm` 脚本，并指定节点数、核数、 分区等信息。

```shell
yhbatch -N 2 -n 32 -p 653 ./submit_cp2k.sh
```

> 提交作业常用参数说明：

`-N,--nodes`：指定作业要使用的节点数。

`-n,--ntasks`：指定作业要加载的任务数。

`-p, --partition`：指定作业要使用的分区。作业将从指定的分区中分配资源，同时使用指定分区的配置进行访问控制检查、资源限制等。如未指定，作业将被提交到系统的默认分区。一个作业必定位于一个分区中，不能跨分区。

`--mpi=pmix`：指定 mpi 运行方式。

`-c,--cpus-per-task`：指定每个任务使用的处理器数，默认每个计算任务使用一个处理器。对于某些多线程的计算任务，如 openMP 程序，使用多个处理器可以得到最佳性能。

### 查看任务

- 提交作业后，使用命令 `yhinfo [-p PARTITION]查看节点信息[指定分区]`。 

- 使用命令 `yhqueue [-u USERNAME]`查看用户当前排队作业[指定用户]

- 使用命令 `yhcontrol`查询仍运行未结束作业；格式为：`yhcontrol show jobs [jobid]`;`jobid` 为单个作业编号，缺省时输出` yhqueue` 命令能查询到的所有作业。

- 使用命令 `yhacct` 查询已提交作业；

常用主要参数有为：

`-u`:指定用户名，非 root 用户可缺省；

`-S`:指定查询开始时间；

`-E`:指定查询结束时间； 

`-j`，`--jobs`:查询作业号列表；

`--format`:指定作业号输出的列表名。

### 结束任务

若任务正常运行结束：`slurm-jobid.out` 文件中不会有任务的报错信息。

若任务提前取消或终止：使用 `yhq` 命令获取 `jobid`；然后 `yhcancel jobid` 即取消指定作业 id 的任务；或使用`yhcancel -u username` 命令即取消指定用户提交的所有任务（**慎用**）。 