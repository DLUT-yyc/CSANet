# CSANet for Video Semantic Segmentation With Inter-Frame Mutual Learning (IEEE Signal Processing Letters 2021)

## Enviroment

* Titan RTX(24G) * 8
* pytorch(1.7.1+)

## Models

ResNet101 Pretrained model：https://pan.baidu.com/s/1Lwzd8dDN12OCQbif64TIrg (IIAU)

Best.pth is available: https://pan.baidu.com/s/1Mx_rn1k_jEhNOOnhDQqKtQ (IIAU)

## Training

```bash
sh train.sh
```

## Validation

```bash
sh val.sh
```

## Testing(submit)

```bash
sh test.sh
```

## Results on Cityscapes val:

CSANet lr-decay-99600 79.16%

CSANet lr-decay-99200---1e-4-finetune-111600 79.61%

CSANet 111600-ms 80.76%

## Acknowledgment

[OCNet](https://github.com/openseg-group/OCNet.pytorch#ocnet-object-context-network-for-scene-parsing-pytorch)

