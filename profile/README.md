# 深圳大学计算机视觉研究所 :sunny: 
## 研究领域

  1. 通用深度学习理论
  
     - 新型网络架构、特征提取模块提升CNN、Transformer性能；提出小波卷积层代替下采样池化及上采样模块，解决高低频信号混叠问题。结合手工特征，引导网络早期的快速收敛。研究最新文本-图像跨模态生成模型及其应用。
        - **Wavelet Integrated CNNs for Noise-Robust Image Classification (CVPR 2020)** [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Li_Wavelet_Integrated_CNNs_for_Noise-Robust_Image_Classification_CVPR_2020_paper.pdf)] [[code](https://github.com/CVI-SZU/WaveCNet)]

  2. 弱监督物体检测/语义分割
  
     - 基于图像类别标签的物体检测网络、语义分割网络，减少通用物体检测/分割网络对标注数据的要求。研究如何利用大规模预训练模型减少网络对训练数据的要求，在零样本开放数据上依然能够取得良好性能。
        - **CLIMS: Cross Language Image Matching for Weakly Supervised Semantic Segmentation (CVPR 2022)** [[paper](https://arxiv.org/abs/2203.02668)] [[project](https://sierkinhane.github.io/clims/)] [[code](https://github.com/CVI-SZU/CLIMS)]
        - **CCAM: Contrastive learning of Class-agnostic Activation Map for Weakly Supervised Object Localization and Semantic Segmentation (CVPR 2022)** [[paper](https://arxiv.org/pdf/2203.13505.pdf)] [[project]()] [[code](https://github.com/CVI-SZU/CCAM)]
     
  3. 人脸识别/分析与合成
  
     - 基于深度卷积网络的人脸识别、表情、年龄等属性识别方法及系统，基于GAN、StyleGAN的人脸编辑，文本驱动的人脸生成，以及基于人脸视频的个体性格、抑郁等心理情绪预测。开发的人脸识别系统在深圳高交会、双创周、杭州G20和新疆安防广泛使用。

  4. 医学图像智能分析
  
     - 大脑MRI图像肿瘤区域分割、牙齿CT图像种牙位置预测、多尺度卷积网络的X光图像肺结节识别，眼底图像/视网膜OCT图像疾病检测；
     
     - B超图像甲状腺结节分割与良恶性分类、鼻内窥镜图像腺样体肥大检测、消化道内窥镜图像息肉检测与分析；
     
     - 胃肠组织病理全切片WSI图像肿瘤判读，宫颈TCT细胞病理全切片WSI图像判读
