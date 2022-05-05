## pytorch-unet
- 来源于[usuyama/pytorch-unet](https://github.com/usuyama/pytorch-unet)
- 训练字幕图像的背景擦除模型

#### 目录结构
```text
.
├── convert.py
├── datasets
│   └── gen_datasets
│            ├── train
│            │   ├── images
│            │   └── masks
│            └── val
│                ├── images
|                └── masks
├── eval_onnxruntime.py
├── loss.py
├── pytorch_unet.py
├── requirements.txt
└── train_self.py
```