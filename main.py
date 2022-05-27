from mnist import MNIST
from numpy import ndarray
import numpy


def dataset_load(dir: str):
    """
    从目标文件夹中加载数据集
    
    :param dir: 数据集所在文件夹 
    :return: 训练集与测试集的样本与标签
    """""
    dataset = MNIST(dir)
    train_x, train_y = dataset.load_training()
    test_x, test_y = dataset.load_testing()
    train_x = numpy.array(train_x).T
    train_y = numpy.array(train_y)
    test_x = numpy.array(test_x).T
    test_y = numpy.array(test_y)
    return train_x, train_y, test_x, test_y


def get_hypothesis(theta,x):
    return 0


def softmax(theta: ndarray, train_x: ndarray, train_y: ndarray, class_num: int, feature_num:int, sample_num:int):
    """
    :param theta: 
    :param train_x: 训练集样本
    :param train_y: 训练集标签
    :return: 
    """""

    print()
    return ndarray


if __name__ == '__main__':
    train_x, train_y, test_x, test_y = dataset_load('./mnist_dataset')
    class_num = 10
    feature_num = len(train_x[:, 0])
    sample_num = len(train_x[0, :])
    theta = numpy.random.random((feature_num, class_num - 1))
    theta_1 = theta[:,0]
    softmax(theta, train_x, train_y, class_num, feature_num, sample_num)
    print(sample_num)
