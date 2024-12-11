# Difficulty-aware Balancing Margin Loss for Long-Tailed Recognition (AAAI 2025)

--- 

## Abstract
When trained with severely imbalanced data, deep neural networks often struggle to accurately recognize classes with only a few samples.
Previous studies in long-tailed recognition have attempted to rebalance biased learning using known sample distributions, primarily addressing different classification difficulties at the class level.
However, these approaches often overlook the instance difficulty variation within each class.
In this paper, we propose a difficulty-aware balancing margin (DBM) loss, which considers both class imbalance and instance difficulty.
DBM loss comprises two components: a class-wise margin to mitigate learning bias caused by imbalanced class frequencies, and an instance-wise margin assigned to hard positive samples based on their individual difficulty.
DBM loss improves class discriminativity by assigning larger margins to more difficult samples.
Our method seamlessly combines with existing approaches and consistently improves performance across various long-tailed recognition benchmarks.

## Requirements
 - pytorch
 - torchvision
 - progress

## Running Examples

### Long-tailed CIFAR

Running DBM variants (except GML) on CIFAR-100-LT (IF 100):
```shell 
cd cifar

## DBM-CE
python main.py --gpu 0 --dataset cifar100 --name dbm_ce_m01 --loss_fn ce --imb_ratio 100 --use_norm --use_dbm --max_margin 0.1 --cos --cutout --aug_type autoaug_cifar

## DBM-DRW
python main.py --gpu 0 --dataset cifar100 --name dbm_drw_m01 --loss_fn ce_drw --imb_ratio 100 --use_norm --use_dbm --max_margin 0.1 --cos --cutout --aug_type autoaug_cifar

## DBM-BS
python main.py --gpu 0 --dataset cifar100 --name dbm_bs_m01 --loss_fn bs --imb_ratio 100 --use_norm --use_dbm --max_margin 0.1 --cos --cutout --aug_type autoaug_cifar

## DBM-BCL
python main.py --gpu 0 --dataset cifar100 --name dbm_bcl_m01 --loss_fn bcl --imb_ratio 100 --use_norm --use_dbm --max_margin 0.1 --cos --cutout --aug_type autoaug_cifar --wd 5e-4 --lr_decay 0.1 --lr 0.15

## DBM-NCL
python main.py --gpu 0 --dataset cifar100 --name dbm_ncl_m01 --loss_fn ncl --imb_ratio 100 --use_norm --use_dbm --max_margin 0.1 --cos --cutout --aug_type autoaug_cifar --epochs 400
```


You can change dataset to CIFAR-10-LT by changing the dataset argument to ```--dataset cifar10```.

Similarly, you can also change the imbalance ratio by adjusting ```--imb_ratio```.

### Large datasets

At least 4 GPUs are used in the following experiments.

Running DBM-BCL on ImageNet-LT and iNaturalists:

```shell 
cd large_scale

## ImageNet-LT
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --dataset imgnet --num_classes 1000 --root datasets/imagenet-LT --epochs 90 --arch resnet50 --use_norm --use_dbm --wd 5e-4 --cos --lr 0.1 --batch_size 256 --loss_type BCL --exp_str dbm_bcl_resnet_m01 --max_margin 0.1 


## iNaturalist2018
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --dataset inat --num_classes 8142 --root datasets/iNaturalist --epochs 100 --arch resnet50 --use_norm --use_dbm --wd 1e-4 --cos --lr 0.2 --batch_size 256 --loss_type BCL --exp_str dbm_bcl_resnet_m01 --max_margin 0.1 
```

### DBM-GML

Running Experiments combining GML (ICML 2023) with DBM:

```shell 
cd GML


## CIFAR100-LT
./sh/CIFAR100_train_teacher.sh
./sh/CIFAR100_train_student.sh

## ImageNet-LT
./sh/ImageNetLT_train_teacher.sh
./sh/ImageNetLT_train_student.sh

## iNaturalist2018
./sh/iNaturalist_train_teacher.sh
./sh/iNaturalist_train_student.sh
```

## References
- https://github.com/sumyeongahn/CUDA_LTR/
- https://github.com/naver-ai/cmo
- https://github.com/FlamieZhu/Balanced-Contrastive-Learning
- https://github.com/Bazinga699/NCL
- https://github.com/bluecdm/Long-tailed-recognition