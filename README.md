# DiffHDRsyn
This repository contains the official python implementation for

full paper at AAAI 2021 "End-to-End Differentiable Learning to HDR Image Synthesis for Multi-exposure Images" 
short paper "Differentiable HDR Image Synthesis using Multi-exposure Images" at DiffCVGP NeurIPSW 2020

If you find our paper or code useful, please cite our papers.


### Requirements

VDS dataset can be downloaded from [here](https://drive.google.com/drive/folders/1i7iTC6t6e_ZhyCq178V3-nN-IS5-5WOe?usp=sharing)
Pretrained weights can be downloaded from [here](https://drive.google.com/drive/folders/1inzZWbBTlOJTuqJODHvOhNSg-o60LyWs?usp=sharing)

1. pytorch >= 1.2.0
2. torchvision >= 0.4.0
3. scipy == 1.2.1
4. pyaml == 19.4.1
5. opencv == 4.2.0
6. pillow == 6.1.0
7. scikit-learn == 0.20.4
8. matplotlib = 3.2.1

### Training

### Testing
Download VDS dataset (test_set, test_hdr) from upper link and edit 'test_dir' of default_config.yaml to the desired path
Download pretrained weights and place the weights in 'model_dir' of default_config.yaml


## Acknowledgement

Our cobi loss code is developed based on the PyTorch implementation of Contextual Loss (CX) and Contextual Bilateral Loss (CoBi) provided by [contextual_loss_pytorch](https://github.com/S-aiueo32/contextual_loss_pytorch)
