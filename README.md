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

| Networks      | Para(M) |     JA     |    DC    |    AC     |   SE   | SP  |
| ------------- | :-----: | :--------: | :--------: | :--------: | :--------: | :--------: |
| FCN           |  14.6   |   0.7866   |   0.8680   |   0.9504   |   0.8827   |   0.9703   |
| U-Net         |  32.9   |   0.8169   |   0.8881   |   0.9568   |   0.8858   |   0.9781   |
| U-Net++       |  34.9   |   0.8187   |   0.8893   |   0.9568   |   0.8910   |   0.9769   |
| AttU-Net      |  33.3   |   0.8199   |   0.8903   |   0.9577   |   0.8898   |   0.9780   |
| DeepLabv3+    |  37.9   |   0.8232   |   0.8926   |   0.9587   |   0.8974   |   0.9735   |
| DenseASPP     |  33.7   |   0.8253   |   0.8935   |   0.9589   |   0.8950   |   0.9748   |
| BCDU-Net      |  28.8   |   0.8084   |   0.8833   |   0.9548   |   0.8913   |   0.9720   |
| Focus-Alpha   |  26.4   |   0.8192   |   0.8893   |   0.9584   | **0.9157** |   0.9668   |
| DO-Net        |  24.7   |   0.8261   |   0.8948   |   0.9578   |   0.9036   |   0.9704   |
| CE-Net        |  29.0   |   0.8282   |   0.8959   |   0.9597   |   0.9054   |   0.9725   |
| CPF-Net       |  43.3   |   0.8292   |   0.8963   |   0.9602   |   0.9062   |   0.9738   |
| MSCA-Net (our） |    27.09   | **0.8341** | **0.8998** | **0.9625** |   0.9114   | **0.9748** |

### Reference



