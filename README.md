# My Final Year Project(FYP) In National University of Singapore(NUS)
## First, I must thank for this video, he teaches me to build my first network manually, really thanks
salute to: https://www.bilibili.com/video/BV11341127iK?from=search&seid=4525203158161865153&spm_id_from=333.337.0.0
## You need

```
Pytorch(stable 1.9.1) 
```
Both cuda version and cpu version are OK

## File Structure
```
📦FYP-U-Net
 ┣ 📂data
 ┃ ┣ 📂imgs
 ┃ ┃ ┣ 📌···.tif
 ┃ ┃ ┗ ···
 ┃ ┣ 📂masks
 ┃ ┃ ┣ 📌···_mask.tif
 ┃ ┃ ┗ ···
 ┃ ┣ 📂PredictImage 
 ┃ ┃ ┣ 📌0.tif
 ┃ ┃ ┣ 📌1.tif
 ┃ ┃ ┗ ···
 ┃ ┣ 📂SaveImage
 ┃ ┃ ┣ 📌0.tif
 ┃ ┃ ┣ 📌1.tif
 ┃ ┃ ┗ ···
 ┃ ┗ 📂Source
 ┃ ┃ ┣ 📂TCGA_CS_4941_19960909
 ┃ ┃ ┃ ┣ 📌TCGA_CS_4941_19960909_1.tif
 ┃ ┃ ┃ ┣ 📌TCGA_CS_4941_19960909_1_mask.tif 
 ┃ ┃ ┃ ┣ 📌TCGA_CS_4941_19960909_2.tif
 ┃ ┃ ┃ ┣ 📌TCGA_CS_4941_19960909_2_mask.tif 
 ┃ ┃ ┃ ┗ ···
 ┃ ┃ ┣ 📂TCGA_CS_4942_19970222
 ┃ ┃ ┗ ···
 ┣ 📂params
 ┃ ┗ 📜unet.pth
 ┣ 📓README,md
 ┣ 📄data.py
 ┣ 📄net.py
 ┣ 📄utils.py
 ┗ 📄train.py
 ```

* '**data**' dir contains the origin dataset in '**Source**' dir. And the dataset can be download in Kaggle (https://www.kaggle.com/c/rsna-miccai-brain-tumor-radiogenomic-classification/). And also you can use different dataset.
* '**imgs**' contains images and '**masks**' contains corresponding masks to images. Corresponding masks have a `_mask` suffix. More inforamtion you can check in kaggle.
* '**SaveImage**' is meant for store train results and '**PredictImage**' is meant for store test results.
* '**params**' is meant for store model.

## Quick Up
 Run train.py

## Change DataSet
* Delte all images in data dir and its subdir.
* Install dataset from kaggle or anything you like(**PS. Corresponding masks must have a `_mask` suffix**) into '**Source**' dir
* Run data.py

  ```
  python3 data.py
  ```
  Remember change the path.
  After this, you will get images and masks in imgs dir and masks dir.
* Run train.py

  ```
  python3 train.py
  ``` 
  Remember change the path.
  And you can see the results in '**SaveImage**' dir and '**PredictImage**' dir.

## Results
After 70 epochs:
![Segment Image](https://i.ibb.co/rGYCwLz/92.png)
After 1500 epochs:
![Segment Image](https://i.ibb.co/myGKcnq/image.png)

## Pre-trained model
https://drive.google.com/file/d/1yyrITv7BQf9kDnP__g6Qa3_wUPD1c_i_/view?usp=sharing
Put the .pth file into **params** dir. Out net will use it.