<!-- Banner -->
<p align= "center">
    <a href="https://www.uit.edu.vn/" title="Trường Đại học Công nghệ Thông tin" style="border: none;">
    <img src="https://i.imgur.com/GzhcLYE.png" 
width="200"
alt="Trường Đại học Công nghệ Thông tin | University of Information Technology">
  </a>
    <a href="https://uit-together.github.io/" title="UIT Together Research Group" style="border: none;">
    <img src="https://i.imgur.com/NjNLy4i.png" 
width="180"
alt="UIT Together Research Group"> 
     </a>    
</p>
    
<div align="center">
    <b><font size="5">UIT-Together Research Group </font></b>
    <sup>
      <a href="https://uit-together.github.io/">
        <i><font size="4">HOT</font></i>
      </a>
    </sup>
    &nbsp;&nbsp;&nbsp;&nbsp; 
</div>
  
## Introduction
This is an empirical study of the performance of one-stage object detection methods on the [ZeroWaste](https://github.com/dbash/zerowaste) dataset.

<p>
    <img src="https://i.imgur.com/DSmjwjf.jpg" alt="ZeroWate sample image" style="height: 70%; width: 70%;">
</p>
<p>
    <em>Sample image from ZeroWaste-f[6] dataset.</em>
</p>

[ZeroWaste](https://github.com/dbash/zerowaste) was introduced for industrial-grade waste detection and segmentation. *ZeroWaste-f* subset for fully supervised detection was chosen, containing 4661 frames sampled from 12 processed videos. This benchmark is divided into 3 subsets: 3,002 images for training, 572 images
for validation and 929 images for testing. 

## Implementations
<details open>
<summary>Data Preparation:</summary>

- **Download at:** [zerowaste-f-final.zip](https://zenodo.org/record/6412647/files/zerowaste-f-final.zip?download=1)
Dataset is annotated in COCO format. Whew, such a relief.
Unzip it first, of course.    

- **Organize the directories as follows:**
    ```
    └── work_dir/
        └── data/
            ├── train/
            │   ├── images/
            │   │   └── <train images.PNG>
            │   ├── labels
            │   │   └── <train frames.txt>
            │   └── labels.json
            │   └── sem_seg  
            ├── test/
            │   ├── images/
            │   │   └── <test images.PNG>
            │   ├── labels
            │   └── labels.json
            │   └── sem_seg 
            ├── val/
                ├── images/
                │   └── <val images.PNG>
                ├── labels
                └── labels.json
                └── sem_seg 
    ```
- **Convert annotations to YOLO Darknet format .txt label files (for YOLOv4, YOLOv5 and YOLOv7):**
    
    ```
    python script/convert_format.py
    ```
</details>

<details open>
<summary>Training Models:</summary>
    
<!-- [- **Clone the repository:** 
    ```
    git clone UIT-Together-Research-Group/uit-zerowaste-onestage-detection.git
    ``` -->
- **Training YOLOv3 and YOLOF using MMDetection toolbox:**

    Install dependencies:
    ```
    pip install openmim
    mim install mmdet
    ```

    Clone the MMDetection repository:

    ```
    git clone https://github.com/open-mmlab/mmdetection.git
    cd mmdetection
    ```

    Install more dependencies, cuz why not:
    ```
    pip install -e -v .
    ```
    Modify the following keys in the yolov3.cfg file:
    
    Start training YOLOv3/YOLOF:
    ```
    python tools/train.py path/to/yolov3.cfg
    ```
    
    *Same thing with training YOLOF, using yolof.cfg.*
- **Training YOLOv4:** 
    
    Install **pycocotools**:
    ``` 
    pip install mmpycocotools 
    pip install pycocotools==2.0.1.
    ```
    Download pretrained weights:
    ```
    import gdown
    cd /work_dir
    gdown https://drive.google.com/u/0/uc?id=1TSvLHH48eJJk7Glr5p2lscVet2jCazhi&export=download
    ```
    Install **mish-cuda**:
    ```
    git clone https://github.com/JunnYu/mish-cuda
    cd mish-cuda
    python setup.py build install
    ```
    
    Clone the YOLOv4 Pytorch repo:
    ```
    cd /work_dir
    git clone https://github.com/WongKinYiu/PyTorch_YOLOv4
    cd PyTorch_YOLOv4/
    ```
    
    Install dependencies:
    ```
    pip install -r requirements.txt
    cd /content
    ```
    Start training:
    ```
    python train.py --img 448 448 --batch-size 8 --project "your_save_folder" --data /content/uit-zerowaste-onestage-detection/data/coco.yaml --cfg /content/uit-zerowaste-onestage-detection/config/yolov4.cfg --resume current_epoch --weights /content/yolov4.weights 
    ```

    
- **Training YOLOv5:** 
    Clone the YOLOv5 Pytorch repo:
    ```   
    git clone https://github.com/ultralytics/yolov5 #clone
    cd yolov5
    pip install -r requirements.txt  # install
    ```
    Install dependencies:
    ```
    cd /content
    wget https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5s.pt
    ```
    Start training:
    ```
    cd /content/yolov5
    python train.py --img 640 --batch 16 --epochs 300 --project "your_save_folder" --resume current_epoch --data /content/uit-zerowaste-onestage-detection/data/coco.yaml --weights /content/yolov5s.pt
    ```
    
- **Training YOLOv7** 
    Clone the YOLOv5 Pytorch repo
    ```
    cd /content    # Download YOLOv7 repository and install requirements
    git clone https://github.com/WongKinYiu/yolov7
    cd yolov7
    pip install -r requirements.txt
    ```
    Install dependencies
    ```
    cd /content/yolov7
    wget "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt"
    ```
    Start training
    ```
    cd /content/yolov7
    python train.py --batch 16 --project "your_save_folder" --resume current_epoch.pt --save_period 5 --cfg /content/uit-zerowaste-onestage-detection/config/yolov7.yaml --epochs 300 --data /content/uit-zerowaste-onestage-detection/data/coco.yaml --weights '/content/yolov7/yolov7.pt' --device 0 --hyp /content/yolov7/data/hyp.scratch.p5.yaml 
    ```
    
    </details>
    
    <details open>
    <summary>Evaluation: Describe the evaluation of the trained model results </summary> 

    ```
    # YOLOv5 uses val.py instead test.py
    # Remember to replace the path in coco.yaml
    python test.py --weights your_model --data /content/uit-zerowaste-onestage-detection/data/coco.yaml --img 640 --verbose
    ```  
</details>

<details open>
<summary>Inference: Load the trained model (model.pt) to run prediction</summary>


- Detect for model which is trained by Pytorch
```
# Replace "our_model" to "YOLOv4, YOLOv5, YOLOv7"
python detect.py --weights /uit-zerowaste-onestage-detection/models/our_model.pt --data /content/yolov5/data/coco.yaml --img 640 --conf 0.25 --source '/uit-zerowaste-onestage-detection/data/images'
```
- Detect for model which is trained by mmdetection
```
# Replace "our_model" to "YOLOv3, YOLOF"
# Replace "our_config" to "YOLOv3, YOLOF"
cd /content/mmdetection
python tools/test.py /uit-zerowaste-onestage-detection/data/our_config.py /uit-zerowaste-onestage-detection/models/our_model.pth --show-dir 'save_result_path' 
```
</details>

## Results
### Baseline
| Methods    | AP[%] | AP50[%] | AP75[%] | APs[%] | APm[%] | APl[%] |
|------------|-------|---------|---------|--------|--------|--------|
| RetinaNet  | 21.0  | 33.5    | 22.2    | 4.3    | 9.5    | 22.7   |
| MaskRCNN   | 22.8  | 34.9    | 24.4    | 4.6    | 10.6   | 25.8   |
| TridentNet | 24.2  | 36.3    | 26.6    | 4.8    | 10.7   | 26.1   |
### Ours
| Methods | AP[%] | AP50[%] | AP75[%] | APs[%] | APm[%] | APl[%] | Download |
|---------|-------|---------|---------|--------|--------|--------|----------|
| YOLOv3  | 22.0  | 34.1    | 23.4    | 1.6    | 9.6    | 23.6   |  [model]()   |
| YOLOv4  | ~     | ~       | ~       | ~      | ~      | ~      |  [model](https://drive.google.com/file/d/119_qImmj6rxQhmXz_-7FflBUOazzOrlg/view?usp=sharing)   |
| YOLOv5  | ~     | ~       | ~       | ~      | ~      | ~      |  [model]()   |
| YOLOF   | 26.2  | 41.5    | 28.6    | 1.4    | 10.8   | 28.6   |  [model]()   |
| YOLOv7  | ~     | ~       | ~       | ~      | ~      | ~      |  [model]()   |

## Project Members
| No. | Name              | Github                   | Email                  |
|-----|-------------------|--------------------------|------------------------|
| 1   | Huyen Ngoc N. Van | github.com/huyenngocnvan | 20521424@gm.uit.edu.vn |
| 2   | Khanh B. T. Duong | github.com/KDuongThB     | 20521444@gm.uit.edu.vn |
| 3   | Thinh V. Le       | github.com/levietthinh   | 20520781@gm.uit.edu.vn |
| 4   | Bao N. Tran       | github.com/TNB142        | 20520142@gm.uit.edu.vn |

## Acknowledgement

This project was done under the instruction and support of the [UIT Together Research Group](https://uit-together.github.io/) and the [MMLab - UIT](http://mmlab.uit.edu.vn/).
Implementations were conducted using the [MMDetection Toolbox](https://github.com/open-mmlab/mmdetection), and the official source code and guide provided at: [YOLOv4](https://github.com/WongKinYiu/yolov7), [YOLOv5](https://github.com/ultralytics/yolov5), [YOLOv7](https://github.com/WongKinYiu/yolov7).

## References
1. Redmon, J., & Farhadi, A. (2018). Yolov3: An incremental improvement. arXiv preprint arXiv:1804.02767.
2. Bochkovskiy, A., Wang, C. Y., & Liao, H. Y. M. (2020). Yolov4: Optimal speed and accuracy of object detection. arXiv preprint arXiv:2004.10934.
3. G. Jocher, A. Chaurasia, A. Stoken, et al., ultralytics/yolov5: v6.2 - YOLOv5 Classification Models, Apple M1, Reproducibility, ClearML and Deci.ai integrations, version v6.2, Aug. 2022. [Online]. Available: https://doi.org/10.5281/zenodo.7002879.
4. Chen, Q., Wang, Y., Yang, T., Zhang, X., Cheng, J., & Sun, J. (2021). You only look one-level feature. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 13039-13048).
5. Wang, C. Y., Bochkovskiy, A., & Liao, H. Y. M. (2022). YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors. arXiv preprint arXiv:2207.02696.
6. Bashkirova, D., Abdelfattah, M., Zhu, Z., Akl, J., Alladkani, F., Hu, P., ... & Saenko, K. (2022). ZeroWaste Dataset: Towards Deformable Object Segmentation in Cluttered Scenes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 21147-21157).
