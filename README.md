# CrossFusion
##  A Multi-Scale Cross-Attention Convolutional Fusion Model for Cancer Survival Prediction

**Abstract:** Cancer survival prediction from whole slide images (WSIs) is a challenging task in computational pathology due to the large size, irregular shape, and high granularity of the WSIs. These characteristics make it difficult to capture the full spectrum of patterns, from subtle cellular abnormalities to complex tissue interactions, which are crucial for accurate prognosis. To address this, we propose CrossFusion—a novel, multi-scale feature integration framework that extracts and fuses information from patches across different magnification levels. By effectively modeling both scale-specific patterns and their interactions, CrossFusion generates a rich feature set that enhances survival prediction accuracy. We validate our approach across six cancer types from public datasets, demonstrating significant improvements over existing state-of-
the-art methods. Moreover, when coupled with domain-specific feature extraction backbones, our method shows further gains in prognostic performance compared to general-purpose backbones

<p align="center">
<img src=".github/CrossFusion.jpg" style="width:80%; height:auto;" align="center" />
</p>

## Downloading TCGA Data

To download diagnostic WSIs (formatted as .svs files), please refer to the [NIH Genomic Data Commons Data Portal](https://portal.gdc.cancer.gov/). We used WSIs from these studies: *BLCA*, *BRCA*, *COAD*, *LUAD*, *GB&LGG*, and *UCEC*. WSIs for each cancer type can be downloaded using the [GDC Data Transfer Tool](https://docs.gdc.cancer.gov/Data_Transfer_Tool/Users_Guide/Data_Download_and_Upload/).

## WSI Processing and Embedding

We follow [CLAM](https://github.com/mahmoodlab/CLAM) to patch and embed whole-slide images (WSIs). The patches are 256×256 and from three different magnification power levels: 20x, 10x, and 5x. As the WSIs in the TCGA datasets have different maximum magnifications, different magnification levels in the WSI translate to exact magnification power. For example, level 0 for a WSI with a maximum magnification power of 40x translates to a 40x magnification level, but level 0 for a WSI with 20x maximum magnification power will be 20x. For a WSI with a 40x maximum magnification level, 20x patches will be extracted by patching the WSI with a patch size of 512x512 at level 0 and then resizing the patches to 256x256 during the feature extraction part. In the subsequent steps, we follow CLAM's storage format to obtain the patch coordinates and features.

## Training

For training, please take a look at the `train.sh` file. In the file, the proper address of folders for the patches, and features should be set in the `img_dir` and `pt_dir`. You should also modify the `save_dir` variable to match the folder paths on your machine. The other variables are for data that are present in the `data` folder. Finally, to start the training, you can do:
```shell

sh train.sh
```
