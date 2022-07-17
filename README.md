
# Abstract
This repo contains the code used in Context Unaware Knowledge Distillation for Image Retrieval paper.

In this work we experimented on a new approch to knowledge distillation, which is context unaware knowledge distillation where the knowledge distillation of student is done from context un-aware teacher. We also propose a new efficient student model architecture for knowledge distillation. The proposed approach follows a two-step process. The first step involves pre-training the student model with the help of context unaware knowledge distillation from the teacher model followed by fine-tuning the student model on the context of image retrieval. We compare the retrieval results, parameters and operations of the student models with the teacher models under different retrieval frameworks, including deep cauchy hashing [DCH](http://ise.thss.tsinghua.edu.cn/~mlong/doc/deep-cauchy-hashing-cvpr18.pdf) and central similarity quantization [CSQ](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yuan_Central_Similarity_Quantization_for_Efficient_Image_and_Video_Retrieval_CVPR_2020_paper.pdf). The experimental results confirm that the proposed approach provides a promising trade-off between the retrieval results and efficiency.

# How to run
To train student on Knowledge distillation you can use 
```
python KD.py
```
To train the pretrained student models on Image retrival you can use
```
pyhon DCH.py  
pyhon CSQ.py   
```
Weights of student model on knowledge distilled student are provided in ```KD_Checkpoints``` folder

# Datasets
- You can download   NUS-WIDE [here](https://drive.google.com/file/d/0B7IzDz-4yH_HMFdiSE44R1lselE/view?usp=sharing&resourcekey=0-w5zM4GH9liG3rtoZoWzXag).  After downloading, you need to move the nus_wide.tar.gz to ./dataset/nus_wide_m and extract the file there.
- For Cifar 10 it will be downloaded from torchvision.

# References
Code used from [DeepHash-Pytorch](https://github.com/swuxyj/DeepHash-pytorch)