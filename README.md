# SROD
This project was implemented based on [EDSR](https://github.com/sanghyun-son/EDSR-PyTorch), [ESRGAN](https://github.com/xinntao/ESRGAN), and [Detectorn2](https://github.com/facebookresearch/detectron2).

## Abstract
![SROD](https://user-images.githubusercontent.com/44395361/113983961-871f9b80-9885-11eb-910c-d89ebe729399.png)
<br/>
With the introduction of deep learning, over the past decade, a significant amount of research has been conducted in the field of computer vision. However, despite these advances, there are some limitations that need to be overcome to enable real-world application of deep learning-based OD models. One such limitation is inaccurate OD when the image quality is poor or the target object is small.

To address this issue, we investigated a method to fuse the super-resolution (SR) task of converting a low-resolution image to a high-resolution image with the OD task. First, 32 SR models and 14 deep learning-based OD models were investigated and classified according to the characteristics of predominant common architectures. Among them, seven SR models and five OD models were selected for extensive experiments. The effectiveness of the proposed method according to the application domain was verified by separating general OD and face detection conditions. Experiments were conducted using Microsoft’s Common Objects in Context (MS COCO) dataset and the Wider Face dataset, respectively. To simulate a degraded real-world image, extensive quantitative experiments were performed using three complex image degradation methods. The experimental results demonstrated that the OD performance enhancement tends to increase in proportion to the peak signal-to-noise ratio (PSNR) and structural similarity index measure (SSIM) indices of the SR model. In particular, we noted that, when the SR model was trained using adversarial learning, a higher performance enhancement rate was achieved even if the PSNR and SSIM indices were low. In addition, based on the quantitative experimental results, it was found that the SR model with a pre-upsampling structure is unsuitable for fusion, and the performance enhancement rate of the DETR detector model with transformer architecture is high. The COCO experiment results demonstrated that, on average, the OD enhancement rate for small objects was 16.6 points higher than the enhancement rate for objects of all sizes, and it is expected that adequate fusion of SR and OD models can effectively address the limitations of existing OD models.

## Model Zoo
- Super-resolution
  -  DBPN
  -  DRRN
  -  EDSR
  -  ESRGAN
  -  MSRN
  -  RCAN
  -  RRDB

- Object Detection
  -  [Detectron2](https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md)


## Dependency
- python ≥ 3.6
- detectron2 == 0.4 [(INSTALL.md of Detectron2)](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md)
  - gcc ≥ 5.4
  - g++ ≥ 5.4
  - pyyam == 5.1
  - torch == 1.7
  - cv2 (for visualization)
- gdown


## Quickstart
```{bash}
cd codes/
sh download_weights.sh
python main.py --SR EDSR --OD FasterRCNN
```
