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
#   DIOR/
#     dior_annotations/
#     images/
#   HRSID/
#     annotations/
#     images/
#   SSDD/
#     annotations/
#     JPEGImages/
ln -s ${YOUR_DATA} data
bash tools/dataset/semi_hrsid.sh
bash tools/dataset/semi_ssdd.sh
```

### Training
```shell script
# num_SAR_images: number of labeled SAR images for training
# num_gpus: number of gpus for training
bash tools/dist_train_dual_teacher_partially_hrsid.sh semi ${fold} ${num_SAR_images} ${num_gpus}
```
### Evaluation
```shell script
python tools/test.py <config_file_path> <model_file_path> --eval bbox --work-dir <save_dir>
```

### Acknowledgement
A large part of the codes are borrowed from [SoftTeacher](https://github.com/microsoft/SoftTeacher). Thanks for the excellent work!