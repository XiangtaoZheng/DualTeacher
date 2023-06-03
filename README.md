# Dual Teacher: A Semi-Supervised Co-Training Framework for Cross-Domain Ship Detection


## Introduction
This is the reserch code of the IEEE Transactions on Geoscience and Remote Sensing 2023 paper.

X. Zheng, H. Cui, C. Xu and X. Lu, "Dual Teacher: A Semi-Supervised Co-Training Framework for Cross-Domain Ship Detection," IEEE Transactions Geoscience and Remote Sensing, 2023.

In this code, we explored the Semi-Supervised Cross-Domain Ship Detection (SCSD) task to improve the cross-domain ship detection performance with a few labeled SAR images. We proposed Dual Teacher framework to integrate cross-domain object detection and semi-supervised object detection for different knowledge fusion.

## Usage

### Requirements
- `Ubuntu 20.04`
- `Anaconda3` with `python=3.6`
- `Pytorch=1.7.0`
- `mmdetection=2.16.0+fe46ffe`
- `mmcv=1.3.9`
- `wandb=0.10.31`


### Installation
```
make install
```

### Data 
- Download DIOR, HRSID and SSDD datasets and put them as follows:
- Execute the following command to generate data set splits:
```shell script
# YOUR_DATA/
#   dior/
#     dior_annotations.json    # only ship instances
#     images/
#   hrsid/
#     annotations/
#     images/
#   ssdd/
#     annotations/
#     JPEGImages/
#   dior_hrsid/
#     annotations/             # labeled optical images and few labeled SAR images
#     images/
ln -s ${YOUR_DATA} data
bash tools/dataset/semi_hrsid.sh
bash tools/dataset/semi_ssdd.sh
```
- ADD HRSIDDataset to MMDetection, similar to COCODataset

### Training
```shell script
# num_SAR_images: number of labeled SAR images for training
# num_gpus: number of gpus for training
bash tools/dist_train_ship_pretrain.sh dior 0 100 ${num_gpus}
for fold in 1,2,3,4,5;
do
    bash tools/dist_train_ship_pretrain.sh dior_hrsid ${fold} ${num_SAR_images} ${num_gpus}
    bash tools/dist_train_dual_teacher_partially_hrsid.sh semi ${fold} ${num_SAR_images} ${num_gpus}
done 
```
### Evaluation
```shell script
python tools/test.py <config_file_path> <model_file_path> --eval bbox --work-dir <save_dir>
```

## Cite
```
@article{zheng2023dual,
 title={Dual Teacher: A Semi-Supervised Co-Training Framework for Cross-Domain Ship Detection},
 author={Zheng, Xiangtao and Cui, Haowen and Xu, Chujie and Lu, Xiaoqiang},
 journal={IEEE Transactions on Geoscience and Remote Sensing},
 year={2023}
 }
```

## Acknowledgement
A large part of the codes are borrowed from [SoftTeacher](https://github.com/microsoft/SoftTeacher). Thanks for the excellent work!