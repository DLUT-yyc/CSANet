
python -m torch.distributed.launch --nproc_per_node=8 --master_port 20000 train.py \
    --gpu '0,1,2,3,4,5,6,7' \
    --start-iters 99200 \
    --num-steps 200000 \
    --learning-rate 1e-4 \
    --snapshot-dir './checkpoint/ohem-ft/' \
    --restore-from './checkpoint/ohem/99200.pth' \
    --optimizer 'SGD' \
    --ohem True \
    --ohem-single \
    --fix-lr \
    --warp_loss 'L2'

# python -m torch.distributed.launch --nproc_per_node=1 --master_port 40000 train.py \
#     --gpu '0' \
#     --start-iters 0 \
#     --num-steps 80000 \
#     --learning-rate 1e-2 \
#     --snapshot-dir './checkpoint/snap0/' \
#     --restore-from './pretrained_model/resnet101-imagenet.pth' \
#     --optimizer 'SGD' \
#     --warp_loss 'KL' &\
# 
# python -m torch.distributed.launch --nproc_per_node=1 --master_port 30000 train.py \
#     --gpu '1' \
#     --start-iters 0 \
#     --num-steps 80000 \
#     --learning-rate 1e-2 \
#     --snapshot-dir './checkpoint/snap1/' \
#     --restore-from './pretrained_model/resnet101-imagenet.pth' \
#     --optimizer 'SGD' \
#     --fix-lr \
#     --warp_loss 'KL' &\
# 
# python -m torch.distributed.launch --nproc_per_node=1 --master_port 20000 train.py \
#     --gpu '2' \
#     --start-iters 0 \
#     --num-steps 80000 \
#     --learning-rate 1e-2 \
#     --snapshot-dir './checkpoint/snap2/' \
#     --restore-from './pretrained_model/resnet101-imagenet.pth' \
#     --optimizer 'SGD' \
#     --warp_loss 'L2' &\

# python -m torch.distributed.launch --nproc_per_node=1 --master_port 10000 train.py \
#     --gpu '3' \
#     --start-iters 0 \
#     --num-steps 80000 \
#     --learning-rate 1e-4 \
#     --snapshot-dir './checkpoint/snap3/' \
#     --restore-from './pretrained_model/resnet101-imagenet.pth' \
#     --optimizer 'Adam' \
#     --fix-lr \
#     --warp_loss 'KL' \

# python -m torch.distributed.launch --nproc_per_node=1 --master_port 5000 train.py \
#     --gpu '4' \
#     --start-iters 0 \
#     --num-steps 80000 \
#     --learning-rate 1e-4 \
#     --snapshot-dir './checkpoint/snap4/' \
#     --restore-from './pretrained_model/resnet101-imagenet.pth' \
#     --optimizer 'Adam' \
#     --fix-lr \
#     --ohem True \
#     --ohem-single \
#     --warp_loss 'KL' 

# [-1, 1]
# python -m torch.distributed.launch --nproc_per_node=1 --master_port 2000 train.py \
#     --gpu '5' \
#     --start-iters 0 \
#     --num-steps 80000 \
#     --learning-rate 1e-4 \
#     --snapshot-dir './checkpoint/snap5/' \
#     --restore-from './pretrained_model/resnet101-imagenet.pth' \
#     --optimizer 'Adam' \
#     --fix-lr \
#     --warp_loss 'KL' 

# # [-5, 5]
# python -m torch.distributed.launch --nproc_per_node=1 --master_port 23400 train.py \
#     --gpu '6' \
#     --start-iters 0 \
#     --num-steps 80000 \
#     --learning-rate 1e-4 \
#     --snapshot-dir './checkpoint/snap6/' \
#     --restore-from './pretrained_model/resnet101-imagenet.pth' \
#     --optimizer 'Adam' \
#     --fix-lr \
#     --warp_loss 'KL' 

##########################################################

# python -m torch.distributed.launch --nproc_per_node=1 --master_port 40000 train.py \
#     --gpu '0' \
#     --start-iters 96000 \
#     --num-steps 200000 \
#     --learning-rate 1e-4 \
#     --snapshot-dir './checkpoint/snap0/' \
#     --restore-from './checkpoint/snap0/96000.pth' \
#     --fix-lr \
#     --optimizer 'SGD' \
#     --warp_loss 'KL' &\
# 
# python -m torch.distributed.launch --nproc_per_node=1 --master_port 30000 train.py \
#     --gpu '1' \
#     --start-iters 96000 \
#     --num-steps 200000 \
#     --learning-rate 1e-2 \
#     --snapshot-dir './checkpoint/snap1/' \
#     --restore-from './checkpoint/snap1/96000.pth' \
#     --optimizer 'SGD' \
#     --fix-lr \
#     --warp_loss 'KL' &\
# 
# python -m torch.distributed.launch --nproc_per_node=1 --master_port 20000 train.py \
#     --gpu '2' \
#     --start-iters 96000 \
#     --num-steps 200000 \
#     --learning-rate 1e-4 \
#     --snapshot-dir './checkpoint/snap2/' \
#     --restore-from './checkpoint/snap2/96000.pth' \
#     --optimizer 'SGD' \
#     --fix-lr \
#     --warp_loss 'L2' &\
# 
# python -m torch.distributed.launch --nproc_per_node=1 --master_port 10000 train.py \
#     --gpu '3' \
#     --start-iters 52000 \
#     --num-steps 200000 \
#     --learning-rate 1e-4 \
#     --snapshot-dir './checkpoint/snap3/' \
#     --restore-from './checkpoint/snap3/52000.pth' \
#     --optimizer 'Adam' \
#     --fix-lr \
#     --warp_loss 'KL' &\
# 
# python -m torch.distributed.launch --nproc_per_node=1 --master_port 5000 train.py \
#     --gpu '4' \
#     --start-iters 6000 \
#     --num-steps 200000 \
#     --learning-rate 1e-4 \
#     --snapshot-dir './checkpoint/snap4/' \
#     --restore-from './checkpoint/snap4/6000.pth' \
#     --optimizer 'Adam' \
#     --fix-lr \
#     --ohem True \
#     --ohem-single \
#     --warp_loss 'KL' 

# [-1, 1]
# python -m torch.distributed.launch --nproc_per_node=1 --master_port 2000 train.py \
#     --gpu '5' \
#     --start-iters 6000 \
#     --num-steps 200000 \
#     --learning-rate 1e-4 \
#     --snapshot-dir './checkpoint/snap5/' \
#     --restore-from './checkpoint/snap5/6000.pth' \
#     --optimizer 'Adam' \
#     --fix-lr \
#     --warp_loss 'KL' 

# [-5, 5]
# python -m torch.distributed.launch --nproc_per_node=1 --master_port 23400 train.py \
#     --gpu '6' \
#     --start-iters 6000 \
#     --num-steps 200000 \
#     --learning-rate 1e-4 \
#     --snapshot-dir './checkpoint/snap6/' \
#     --restore-from './checkpoint/snap6/6000.pth' \
#     --optimizer 'Adam' \
#     --fix-lr \
#     --warp_loss 'KL' 
