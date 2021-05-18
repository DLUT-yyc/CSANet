python -m torch.distributed.launch --nproc_per_node=1 --master_port 1995 eval.py \
   --gpu '0' \
   --restore-from './checkpoint/ohem-ft/111600.pth'           &\
python -m torch.distributed.launch --nproc_per_node=1 --master_port 20996 eval.py \
   --gpu '1' \
   --restore-from './checkpoint/ohem-ft/111600.pth'           &\
python -m torch.distributed.launch --nproc_per_node=1 --master_port 30995 eval.py \
   --gpu '2' \
   --restore-from './checkpoint/ohem-ft/111600.pth'           &\
python -m torch.distributed.launch --nproc_per_node=1 --master_port 40995 eval.py \
   --gpu '3' \
   --restore-from './checkpoint/ohem-ft/111600.pth'           &\
python -m torch.distributed.launch --nproc_per_node=1 --master_port 50995 eval.py \
   --gpu '4' \
   --restore-from './checkpoint/ohem-ft/111600.pth'           &\
python -m torch.distributed.launch --nproc_per_node=1 --master_port 60995 eval.py \
   --gpu '5' \
   --restore-from './checkpoint/ohem-ft/111600.pth'           &\
python -m torch.distributed.launch --nproc_per_node=1 --master_port 70995 eval.py \
   --gpu '6' \
   --restore-from './checkpoint/ohem-ft/111600.pth'           &\
python -m torch.distributed.launch --nproc_per_node=1 --master_port 80995 eval.py \
   --gpu '7' \
   --restore-from './checkpoint/ohem-ft/111600.pth'           &\
