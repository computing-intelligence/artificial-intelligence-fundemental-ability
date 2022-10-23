# Transfer Learning

> 很少有人会自己从头到尾的来构建模型

+ tensorflow 还是pytorch，常见、经典的模型结构全部都是已经封装好了,我们在这些基础之上做修改
+ 在Github等这些开源地方，我们找stars多的人
+ Kaggle, 天池比赛这种地方，找一些优秀的案例


+ 科学家/工程师们发现，一个深度神经网络，前边层数的CNN layers的权重是很接近的
+ TaskA: 提取竖着的边沿 -> 提取横着的边沿 -> 提取圆形的部分
+ TaskB: 提取竖着的边沿 -> 提取横着的边沿 -> 提取圆形的部分
+ TaskA, TaskB 的前三层的 Fitlers 的 Paramters 很接近
+ If TaskC: 前边层数的CNN的权重，就不用训练了

> Assuming: TaskA, B, C 用的网络结构是一样
+ TaskA, 需要用很多图片做训练
+ TaskB, 需要用很多图片做训练
+ TaskC: 
    直接使用TaskA 或者 TaskB 已经训练好的权重，不用从头开始计算
    Or: 我们在TaskA， TaskB这些已经好的结果之上，继续训练
    > Why？ 

+ From Random Weights => Converaged Weights: Train Data Driven
+ Dimension Cursity

## Transfer Capability

### Q1
+ TaskA, 1000万图片从零开始训练出来的ResNet
+ TaskB, 10万图片从零开始训练出来的ResNet
+ TaskA的模型结构和TaskB一样
> ? TaskC

+ TaskA, 200万图片从零开始训练出来的ResNet, 动物分类场景
+ TaskB, 50万图片从零开始训练出来的ResNet, 人类分类场景
+ TaskA的模型结构和TaskB一样
> ? TaskC, 人类分类场景

迁移能力 -> 原模型的训练数据越大，迁移能力越强
迁移能力 -> 原模型的问题场景和新问题越相似，迁移能力越强
迁移能力 -> 越强需要冷冻的层数就越靠后，就越少； 越弱，所需要冷冻的层数就越多，就需要往前延伸









