### Requirements
- python == 3.6
- pytorch == 1.10.2
- torchvision == 0.11.3
- cuda (tested in cuda11)


### Dataset
- Datasets should be downloaded into 'data' folder.  The folder structure should look like:
```
data/ 
|-- Cataract_Train
|   |-- Train
|   `-- Val
`-- Cataract_Test
|   |-- Oval_index
|   `-- Oval_index_video
|   `-- Rotate_Index_Standard
|   `-- Video
`-- aachen
`-- revisitop1m
```

- Cataract dataset are available at [Google Drive Link](https://drive.google.com/file/d/1zYisVa3rDt2AhpcP8KE5coih-CZ7L2qu/view?usp=share_link);
- For the original CATARACT dataset, please click on the link (https://ieee-dataport.org/documents/cataract-surgery-dataset-eye-positioning-and-alignment)
- Aachen Day-Night dataset, Oxford and Paris dataset can be obtained by the following operations:
  
  First, create a folder that will host all data in a place where you have sufficient disk space (15 GB required).
  ```bash
  DATA_ROOT=/path/to/data
  mkdir -p $DATA_ROOT
  ln -fs $DATA_ROOT data 
  mkdir $DATA_ROOT/aachen
  ```
  Then, manually download the [Aachen dataset here](https://drive.google.com/drive/folders/1fvb5gwqHCV4cr4QPVIEMTWkIhCpwei7n) and save it as `$DATA_ROOT/aachen/database_and_query_images.zip`.
  Finally, execute the download script to complete the installation. It will download the remaining training data and will extract all files properly.
  ```bash 
  ./download_training_data.sh
  ```
  The following datasets are now installed:

  | full name                       |tag|Disk |# imgs|# pairs| python instance                |
  |---------------------------------|---|-----|------|-------|--------------------------------|
  | Random Web images               | W |2.7GB| 3125 |  3125 | `auto_pairs(web_images)`       |
  | Aachen DB images                | A |2.5GB| 4479 |  4479 | `auto_pairs(aachen_db_images)` |
  | Aachen style transfer pairs     | S |0.3GB| 8115 |  3636 | `aachen_style_transfer_pairs`  |
  | Aachen optical flow pairs       | F |2.9GB| 4479 |  4770 | `aachen_flow_pairs`            |


### Train and Test
#### Train 
  ``` $ python train.py ```
#### Test
The trained weights of the R2D2 and our model are in the checkpoints folder and can be used for testing. 

- First, run the code below to calculate the rotation angle, and also calculate the metrics of MMA, MVMP, MMP. The results will be saved in the 'output' folder.

  ``` $ python test.py ```
- Then, calculate the RE of each video sequence through the downlink code, and the results will be saved in the 'output' folder.

  ``` $ python computeRE.py ```
- In addition, the model can be measured for speed.

  ``` $ python test_FPS.py ```

### 本文对比模型：
- SIFT：cv2.xfeatures2d.SIFT_create()   基于opencv-contrib-python-3.4.2.16 库
- SURF：cv2.xfeatures2d.SURF_create()   基于opencv-contrib-python-3.4.2.16 库
- SuperPoint：https://github.com/magicleap/SuperPointPretrainedNetwork
- D2-Net：https://github.com/mihaidusmanu/d2-net
- R2D2：https://github.com/naver/r2d2

