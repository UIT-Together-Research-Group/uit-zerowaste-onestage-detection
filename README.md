<!-- Banner -->
<p align= "center">
    <a href="https://www.uit.edu.vn/" title="Trường Đại học Công nghệ Thông tin" style="border: none;">
    <img src="https://i.imgur.com/GzhcLYE.png" 
width="200"
alt="Trường Đại học Công nghệ Thông tin | University of Information Technology">
  </a>
    <a href="https://uit-together.github.io/" title="UIT Together Research Group" style="border: none;">
    <img src="https://i.imgur.com/NjNLy4i.png" 
width="200"
alt="UIT Together Research Group"> 
     </a>   
    
</p>
    
<div align="center">
    <b><font size="5">UIT Together website</font></b>
    <sup>
      <a href="https://uit-together.github.io/">
<!--         <i><font size="4">HOT</font></i> -->
      </a>
    </sup>
    &nbsp;&nbsp;&nbsp;&nbsp; 
</div>
  
## Introduction
This is an empirical study of the performance of one-stage object detection methods on the [ZeroWaste](https://github.com/dbash/zerowaste) dataset.



## Implementations
<details open>
<summary>Data preparation</summary>

- **Download at:** [zerowaste-f-final.zip](https://zenodo.org/record/6412647/files/zerowaste-f-final.zip?download=1)
Dataset is annotated in COCO format. Whew, such a relief.
Unzip it first, of course.    
- **Convert annotations to YOLO Darknet format (for YOLOv4, YOLOv5 and YOLOv7):**
    
    ```
    thả code vào không khí
    ```
    
- **Organize the directories:**
    

</details>

<details open>
<summary>Training</summary>
    
Mô tả việc training (clone github, parse tham số, chạy training, lưu model,...)
```
cell code
```

> "Hông biết gì hết" - Khanh.

    
</details>

<details open>
<summary>Evaluation</summary>
Mô tả chi việc đánh giá kết quả mô hình train được
    
```
cell code
```


    
</details>

<details open>
<summary>Inference</summary>
Chỗ này là load model mình đã train rồi (model.pt) lên để chạy prediction.
    
```
bỏ code vào cell
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
| Methods    | AP[%] | AP50[%] | AP75[%] | APs[%] | APm[%] | APl[%] |
|------------|-------|---------|---------|--------|--------|--------|
| YOLOv3     | 22.0  | 34.1    | 23.4    | 1.6    | 9.6    | 23.6   |
| YOLOv4     | ~     | ~       | ~       | ~      | ~      | ~      |
| YOLOv5     | ~     | ~       | ~       | ~      | ~      | ~      |
| YOLOF      | 26.2  | 41.5    | 28.6    | 1.4    | 10.8   | 28.6   |
| YOLOv7     | ~     | ~       | ~       | ~      | ~      | ~      |

## Project Members
| No. | Name              | Github                   | Email                  |
|-----|-------------------|--------------------------|------------------------|
| 1   | Huyen Ngoc N. Van | github.com/huyenngocnvan | 20521424@gm.uit.edu.vn |
| 2   | Khanh B. T. Duong |                          | 20521444@gm.uit.edu.vn |
| 3   | Thinh V. Le       | github.com/levietthinh   | 20520781@gm.uit.edu.vn |
| 4   | Bao N. Tran       | github.com/TNB142        | 20520142@gm.uit.edu.vn |

## Acknowledgement

This project was done under the instruction and support of the [UIT Together Research Group](https://uit-together.github.io/) and the [MMLab - UIT](http://mmlab.uit.edu.vn/).
Implementations were conducted using the [MMDetection Toolbox](https://github.com/open-mmlab/mmdetection) and the official source code provided at: [YOLOv4](https://github.com/WongKinYiu/yolov7), [YOLOv5](), [YOLOv7](https://github.com/WongKinYiu/yolov7).

## References
1. Redmon, J., & Farhadi, A. (2018). Yolov3: An incremental improvement. arXiv preprint arXiv:1804.02767.
2. Bochkovskiy, A., Wang, C. Y., & Liao, H. Y. M. (2020). Yolov4: Optimal speed and accuracy of object detection. arXiv preprint arXiv:2004.10934.
3. G. Jocher, A. Chaurasia, A. Stoken, et al., ultralytics/yolov5: v6.2 - YOLOv5 Classification Models, Apple M1, Reproducibility, ClearML and Deci.ai integrations, version v6.2, Aug. 2022. [Online]. Available: https://doi.org/10.5281/zenodo.7002879.
4. Chen, Q., Wang, Y., Yang, T., Zhang, X., Cheng, J., & Sun, J. (2021). You only look one-level feature. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 13039-13048).
5. Wang, C. Y., Bochkovskiy, A., & Liao, H. Y. M. (2022). YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors. arXiv preprint arXiv:2207.02696.
6. Bashkirova, D., Abdelfattah, M., Zhu, Z., Akl, J., Alladkani, F., Hu, P., ... & Saenko, K. (2022). ZeroWaste Dataset: Towards Deformable Object Segmentation in Cluttered Scenes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 21147-21157).
