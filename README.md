# ImageClassification_with_VGGAttention
Implement a VGG-Attention model from scratch and implement classification validation on the CIFAR-10 dataset.

本文大部分代码参考：[image-classification-with-attention](https://blog.paperspace.com/image-classification-with-attention/)

原文采用的是pytorch model中自带的预训练的VGG16模型,而这里则是从头实现VGG模型,然后融合Attention机制.

当然VGG的实现也是参考了该博客作者的另一篇博客[Writing VGG from Scratch in PyTorch](https://blog.paperspace.com/vgg-from-scratch-pytorch/)

- 实现架构
  - 基本架构为VGG模型，然后添加了一个attention机制
  - 分别使用第3个poolling层的输出和第4个poolling层 与 
  最后一个poolling层（5）结合（通过CNN 和 elements-wise的 add）得到两个attention weight map
  - 然后在对应poolling层（3、4）上做attention : elements-wise multiplication ，
  再对每个channel 上的 H和W维度求和
  - 得到对应poolling层（3、4）的长度为C的vector，同时在最后一个poolling层上使用AveragePool
  - AveragePool 的kernel size为 7 * 7，因为224下采样32倍后，最后一个pooling的输出为7 * 7 * 512
  - 这样又得到一个vector（长度为C，属于最后一个poolling层，即第5层）
  - 将得到的3个vector Concatenates, 再丢到一个FC中，输出10个类别。
- 数据与输入输出
  - 其中VGG模型的输入要求224 * 224
  - 数据使用的CIFAR-10数据集，代码中实现自动下载数据，具体细节可以看代码
  - CIFAR-10数据尺寸为32 * 32，通过resize变为224 * 224，所以难免降低了精度，其他具体内容可查看代码
  
