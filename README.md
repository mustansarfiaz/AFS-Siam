# AFS-Siam PyTorch implementation
## Introduction
This is my project in the direction of Visual Object Tracking.

[Paper](https://link.springer.com/chapter/10.1007/978-981-15-4818-5_13): Adaptive Feature Selection Siamese Networks for Visual Tracking

**AFS-Siam architecture**
![img1](https://github.com/mustansarfiaz/AFS-Siam/blob/master/framework/AFS_Framework.png)

## How to Run - Training
1. **Prerequisites:** The project was built using **python 3.7** and tested on Ubuntu 18.04. It was tested on a **NVIDIA GeForce GTX 1080**. Furthermore it requires [PyTorch 1.0 or more](https://pytorch.org/).

2. Download the **GOT-10k** Dataset in http://got-10k.aitestunion.com/downloads and extract it on the folder of your choice, in my case it is `/media/mustansar/data/benchmarks/GOT-10k` (OBS: data reading is done in execution time, so if available extract the dataset in your SSD partition).


3. Download the ImageNet VID Dataset in http://bvisionweb1.cs.unc.edu/ILSVRC2017/download-videos-1p39.php and extract it on the folder of your choice (OBS: data reading is done in execution time, so if available extract the dataset in your SSD partition). You can get rid of the test part of the dataset, since it has no Annotations.

4. In **config.py** script `root_dir_for_GOT_10k`, `root_dir_for_VID and` and `root_dir_for_OTB` change to your directory. 
```
root_dir_for_GOT_10k = '/media/mustansar/data/benchmarks/GOT-10k' <-- change to your directory 
root_dir_for_VID     = '/media/mustansar/data/benchmarks/VID'     <-- change to your directory
root_dir_for_OTB     = '/media/mustansar/data/benchmarks/OTB2015' <-- change to your directory 
```

5. Run the **train.py** script:
```
python3 train.py
```

## How to Run - Testing
1. Download pretrained `model_e41.pth` from [Google-Drive] (https://drive.google.com/file/d/14pc_bhf2lY84n-Xe49-274ewsGp1a9sX/view?usp=sharing), and put the file under `model/model_e41.pth`.

2. Run the **test.py** script:
```
python3 test.py
```

## Results - 
**OTB2015**
![img1](https://github.com/mustansarfiaz/AFS-Siam/blob/master/framework/overall_2015.png)


## Citing
```
@inproceedings{fiaz2020adaptive,
  title={Adaptive Feature Selection Siamese Networks for Visual Tracking},
  author={Fiaz, Mustansar and Rahman, Md Maklachur and Mahmood, Arif and Farooq, Sehar Shahzad and Baek, Ki Yeol and Jung, Soon Ki},
  booktitle={International Workshop on Frontiers of Computer Vision},
  pages={167--179},
  year={2020},
  organization={Springer}
}
```


