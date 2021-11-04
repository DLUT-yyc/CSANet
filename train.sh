
python -m torch.distributed.launch --nproc_per_node=8 --master_port 20000 train.py \
    --gpu '0,1,2,3,4,5,6,7' \
    --start-iters 99200 \
    --num-steps 200000 \
    --learning-rate 1e-4 \
    --snapshot-dir './checkpoint/snap0/' \
    --restore-from './pretrained_model/resnet101-imagenet.pth' \
    --optimizer 'SGD' \
    --ohem True \
    --ohem-single \
    --fix-lr \
    --warp_loss 'L2'
