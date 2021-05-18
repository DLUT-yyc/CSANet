# Research Logs

* NVIDIA-SMI has failed because it couldn’t communicate with the NVIDIA driver. Make sure that the latest NVIDIA driver is installed and running

```
 1981  ls
 1982  sudo init 3
 1983  ls
 1984  nvidia-smi
 1985  cd bak/
 1986  ls
 1987  sudo chmod a+x NVIDIA-Linux-x86_64-450.119.03.run 
 1988  ls
 1989  ./NVIDIA-Linux-x86_64-450.119.03.run --no-opengl-files
 1990  ./NVIDIA-Linux-x86_64-450.119.03.run --uninstall
 1991  sudo ./NVIDIA-Linux-x86_64-450.119.03.run --uninstall
 1992  sudo ./NVIDIA-Linux-x86_64-450.119.03.run --no-opengl-files
 1993  nvidia-smi
 1994  nvcc -V
 1995  nvidia-smi
 1996  sudo init 5
 1997  ls
 1998  nvidia-smi
 1999  cd
 2000  quit
### 售后人员安装驱动全部过程
```

* DDP显存泄漏

```
kill $(ps aux | grep train.py | grep -v grep | awk '{print $2}')
kill $(ps aux | grep eval.py | grep -v grep | awk '{print $2}')
```

* 命令行快捷输入，在~/.bashrc中

```
alias titan3='ssh -p 2414 omnisky@10.7.47.85'
```

* 1e-2-corase-40000+1e-3-fine-40000 = 76.19miou
* 并行化脚本程序

```
  1 python -m torch.distributed.launch --nproc_per_node=4 --master_port 10000 train.py \                                                                      
  2     --gpu '0,1,2,3' \
  3     --start-iters 4500 \
  4     --learning-rate 2e-3 \
  5     --snapshot-dir './checkpoint/snap2e-3/' \
  6     --tensorboard './checkpoint/tensorboard_2e-3/' \
  7     --restore-from './snap/CS_scenes_4500.pth' &\
  8  
  9 python -m torch.distributed.launch --nproc_per_node=4 --master_port 20000 train.py \
 10     --gpu '4,5,6,7' \
 11     --start-iters 25700 \
 12     --learning-rate 1e-3 \
 13     --snapshot-dir './checkpoint/snap1e-3/' \
 14     --tensorboard './checkpoint/tensorboard_1e-3/' \
 15     --restore-from './snap/CS_scenes_25700.pth'

```

* **tmux**

1. 新建会话`tmux new -s my_session`。
2. 在 Tmux 窗口运行所需的程序。
3. 按下快捷键`Ctrl+b d`将会话分离。
4. 下次使用时，重新连接到会话`tmux attach-session -t my_session`。



### 实验结果

OCNet 77.83%

VSS lr-decay-99000 78.5%

VSS lr-decay-99000---1e-4-finetune-121000 78.83%

VSS 121000-ms 79.97%

* +ohem

VSS lr-decay-99600 79.16%

VSS lr-decay-99200---1e-4-finetune-111600 79.61%

VSS 111600-ms 80.76%
