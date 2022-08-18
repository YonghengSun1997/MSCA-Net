## MSCA-Net: Multi-Scale Contextual Attention Network for Skin Lesion Segmentation 




### Data preparation

We cropped the ISIC 2018 dataset to 224*320 and saved it in npy format,  which can be downloaded from Baidu web disk. 

```
link: https://pan.baidu.com/s/1bIVUdzYG_7tuwalbI4Y8Ww

password: c36c
```

Place the downloaded npy files in the "data" directory and unzip them. The decompression format is as follows:

```
/data/ISIC2018_npy_all_224_320/image/

​		ISIC_0000000.npy

​		ISIC_0000001.npy

​		...

​		ISIC_0016072.npy

/data/ISIC2018_npy_all_224_320/label/

​		ISIC_0000000_segmentation.npy

​		ISIC_0000001_segmentation.npy

​		......

​		ISIC_0016072_segmentation.npy
```

### Train and Test

Our program is easy to train and test,  just need to run "main_train.py". 

```
python main_train.py
```

### Performance on ISIC 2018

| Method          |  Para(M)   |  Flops (G)  |       JI       |      DSC       |      ACC       |
|-----------------|:----------:|:-----------:|:--------------:|:--------------:|:--------------:|
| FCN             | **15.31**  |    21.98    |   78.66±0.41   |   86.80±0.32   |   95.04±0.32   |
| U-Net           |   34.53    |    71.61    |   81.69±0.50   |   88.81±0.40   |   95.68±0.29   |
| U-Net++         |   36.63    |   151.59    |   81.87±0.47   |   88.93±0.38   |   95.68±0.33   |
| AttU-Net        |   34.88    |    72.81    |   81.99±0.59   |   89.03±0.42   |   95.77±0.26   |
| DeepLabv3+      |   39.76    |    47.34    |   82.32±0.35   |   89.26±0.23   |   95.87±0.23   |
| DenseASPP       |   35.37    |    42.63    |   82.53±0.55   |   89.35±0.37   |   95.89±0.28   |
| BCDU-Net        |    28.8    |   171.50    |   80.84±0.57   |   88.33±0.48   |   95.48±0.40   |
| Focus-Alpha     |   26.36    |    41.92    |   81.92±0.63   |   88.93±0.41   |   95.84±0.44   |
| DO-Net          |   24.75    |   122.45    |   82.61±0.51   |   89.48±0.37   |   95.78±0.36   |
| CE-Net          |   29.00    |    9.75     |   82.82±0.45   |   89.59±0.35   |   95.97±0.30   |
| CPF-Net         |   30.65    |  **8.78**   |   82.92±0.52   |   89.63±0.42   |   96.02±0.34   |
| MSCA-Net (Ours) |   27.09    |    12.88    | **84.18±0.38** | **90.52±0.26** | **96.41±0.29** |

### Reference



