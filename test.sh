python -m torch.distributed.launch --nproc_per_node=1 --master_port 7000 inference.py \
   --gpu '0' \
   --output-path "./dlut_output/" \
   --use-flip 'False' \
   --use-ms 'False' \
   --restore-from './checkpoint/ohem-ft/111600.pth'           
