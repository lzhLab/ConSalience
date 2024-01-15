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
├── imgs
│   └──Figure2.png
├── list
│   └──3Dircadb1
|       ├── train.txt
|       ├── test.txt
|       └── val.txt
├── networks
|	├── utils.py
|	├── unet3d_with_salience.txt
│   ├── unet3d.txt
│   └── salience_generator.py
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
  CUDA_VISIBLE_DEVICES=0 python main.py -epo 500 -bs 2 -lr 0.001 --method unet_3d --in_channels 3
  ```

## Citations

```tex
@article{xiao2024connectivity-aware,
  title={Connectivity-aware salience demystifies inconspicuous liver vessel segmentation},
  author={Xiao et al.},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2024}
}
```

