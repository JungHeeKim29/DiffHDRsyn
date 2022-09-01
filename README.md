# DiffHDRsyn
This repository contains the python implementation for  
full paper "End-to-End Differentiable Learning to HDR Image Synthesis for Multi-exposure Images" at AAAI 2021  
short paper "Differentiable HDR Image Synthesis using Multi-exposure Images" at DiffCVGP NeurIPSW 2020

If you find our [paper](https://arxiv.org/abs/2006.15833) or code useful, please cite our papers.

### Requirements
The dataset can be downloaded from [here](https://drive.google.com/file/d/1t9jmy4IbesieE5r6D6IXuR-t98xOi9oY/view?usp=sharing)  
Pretrained weights can be downloaded from [here](https://drive.google.com/file/d/1KTm7HHisea-bWhdGjhbgBFE8uPQnGytF/view?usp=sharing)

The code was tested under the following setting:
  1. pytorch >= 1.2.0
  2. torchvision >= 0.4.0
  3. scipy == 1.2.1
  4. pyaml == 19.4.1, yaml == 0.1.7
  5. opencv == 4.2.0
  6. pillow == 6.1.0
  7. scikit-learn == 0.20.4
  8. matplotlib == 3.2.1

### Training (Available soon)
1. Download VDS dataset and edit 'data_dir', 'validate_dir' in [default_config.yaml](https://github.com/JungHeeKim29/DiffHDRsyn/blob/main/default_config.yaml) to the desired path  
2. Change the 'mode' to 'train' in [default_config.yaml](https://github.com/JungHeeKim29/DiffHDRsyn/blob/main/default_config.yaml)
3. Run **python main.py**

### Testing
1. Download VDS dataset sample (test_set, test_hdr) from upper link and edit 'test_dir' in [default_config.yaml](https://github.com/JungHeeKim29/DiffHDRsyn/blob/main/default_config.yaml) to the desired path  
2. Download pretrained weights from upper link and place the weights in the desired path and edit 'model_dir' in [default_config.yaml](https://github.com/JungHeeKim29/DiffHDRsyn/blob/main/default_config.yaml)  
3. Change the 'mode' to 'test' in [default_config.yaml](https://github.com/JungHeeKim29/DiffHDRsyn/blob/main/default_config.yaml)
4. Run **python main.py**

### Acknowledgement

The code for the coBi loss is folked from the [Contextual_loss_pytorch](https://github.com/S-aiueo32/contextual_loss_pytorch)  
The code for the adam cent is folked from the [Gradient Centralization](https://github.com/Yonghongwei/Gradient-Centralization)
