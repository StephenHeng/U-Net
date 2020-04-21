#### Joint Weakly and Semi-Supervised Deep Learning
一篇关于半监督学习在医疗图像中的应用的论文
Joint Weakly and Semi-Supervised Deep Learning for Localization and Classification of Masses in Breast Ultrasound Images
Author: Shin, Seung Yeon   Lee, Soochahn   Yun, Il Dong   Kim, Sun Mi   Lee, Kyoung Mu    
Journal: IEEE Transactions on Medical Imaging 

在看懂论文之前，需要具备一些基础知识
1.关于faster rcnn的网络架构，损失函数（其中rpn网络是如何生成候选框的也十分重要）
2.关于多实例学习的知识
作者没有提供完整数据集，如需训练可以使用自己的数据集
代码文件：
train.py：将弱标注数据与强标注数据作为一个batch训练
train_alter.py：弱标注数据与强标注数据分别作为一个batch进行交替训练

#### Please watch our videos at
`classmates/vision`: [https://space.bilibili.com/202603446](https://space.bilibili.com/202603446)
