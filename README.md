# Connectivity-aware salience demystifies inconspicuous liver vessel segmentation

- Implementation of our work: [Connectivity-aware salience demystifies inconspicuous liver vessel segmentation](https://arxiv.org/abs/xxxx.xxxxx).

![](./imgs/Figure2.png)

## 1.  Prepare data

1. Sign up in the [official 3D-IRCADb website](https://www.ircad.fr/research/data-sets/liver-segmentation-3d-ircadb-01/) and download the dataset. Convert them to numpy format, clip the images within [0, 400], normalize each 3D image to [0, 1], and transform the 3D numpy array to h5 format for training and testing.
3. You can also send an Email directly to xiaoxu_u@qq.com to request the preprocessed data for reproduction.
4. The directory structure of the whole project is as follows:

```bash
.
├── data
|   └──3Dircadb1
|       └── data_h5
|           ├── case001.h5
|           └── *.h5
├── datasets
│   ├──liver_vessel.py
│   └──...
├── supplement
│   └── Supplementary.pdf
├── imgs
│   └──Figure2.png
├── list
│   └──3Dircadb1
|       ├── train.txt
|       ├── test.txt
|       └── val.txt
│   ├── LiVS
│   │   ├── test.txt
│   │   ├── train.txt
│   │   └── val.txt
│   └── MSD
│       ├── test.txt
│       ├── train.txt
│       └── val.txt
├── ── networks
│   ├── AttenUNet.py
│   ├── AttenUNet_with_salience.py
│   ├── salience_generator.py
│   ├── TransUNet.py
│   ├── TransUNet_with_salience.py
│   ├── unet3d.py
│   ├── unet3d_with_salience.py
│   ├── unetPlusPlus.py
│   ├── unetPlusPlus_with_salience.py
│   ├── unet.py
│   ├── unet_with_salience.py
│   ├── UNeXt.py
│   ├── UNeXt_with_salience.py
│   ├── utils.py
│   ├── YNet.py
│   └── YNet_with_salience.py
├── main.py
├── metrics.py
├── README.md
├── requirements.txt
├── trainer.py
└── utils.py
```

## 2. Environment

Please prepare an environment with python=3.7, and then use the command "pip install -r requirements.txt" for the dependencies.

## 3. Training/Testing

* Run the train script on 3Dircadb1 dataset. The backbone could be any segmentation architecture .

  ```bash
  CUDA_VISIBLE_DEVICES=0 python main.py -epo 500 -bs 2 -lr 0.001 --dataset 3Dircadb1 --method unet_3d --withSalience 1 --in_channels 3
  ```
  
Key Arguments

--dataset : Name of dataset (File name of dataset, default: 3Dircadb1)

--method : Backbone model (unet_3d, unet, unetPlusPlus, AttenUNet, UNeXt, TransUNet, YNet, default: unet_3d)

--withSalience : Use salience module (1) or not (0)

--in_channels : Number of input channels (default: 3)

--out_channels : Number of output channels (default: 1)

--max_epochs : Number of training epochs (default: 1000)

--batch_size : Batch size (default: 1)

--base_lr : Initial learning rate (default: 0.001)

--deterministic : Use deterministic training (1) or non-deterministic (0)
## Citations

```tex
@article{xiao2024connectivity-aware,
  title={Connectivity-aware salience demystifies inconspicuous liver vessel segmentation},
  author={Xiao et al.},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2024}
}
```

