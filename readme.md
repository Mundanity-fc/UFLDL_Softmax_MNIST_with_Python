# UFLDL Tutorial 课程 Softmax 章节 Exercise 1C 的 Python 实现

通过模拟给定的 [Matlab代码](https://github.com/amaas/stanford_dl_ex/tree/master/ex1)，完成课程实验的要求任务，即实现 Softmax 中有关损失函数值的计算与梯度的计算。

包含的自定义函数如下：

 - loadMNISTImages：模拟了 Matlab 中 loadMNISTImages.m 文件的功能，完成对 MNIST 数据集的加载
 - loadMNISTLabels：模拟了 Matlab 中 loadMNISTLabels.m 文件的功能，完成对 MNIST 标签集的加载
 - dataset_load：模拟了 ex1_load_mnist.m 文件的功能，完成了对数据的加载和预处理
 - softmax：模拟并完成了 softmax_regression.m 中的所需的功能，完成了损失函数值的计算与梯度的计算
 - get_softmax_loss：用于在优化器中传递可衡量的损失值
 - get_softmax_gradient：用于在优化器中传递可衡量的梯度值
 - get_accuracy：模拟了 Matlab 中 multi_classifier_accuracy.m 文件的功能，对数据集进行准确度检测
 - main：主函数，使用了 Scipy.optimize.minimize 进行目标优化