import numpy as np
from numpy import ndarray


def loadMNISTImages(dir: str):
    """
    从目标文件中加载数据，模拟了Matlab中loadMNISTImages.m文件的功能。
    
    :param dir: 目标文件 
    :return: 数据集
    """""
    f = open(dir, mode='rb')
    magic = int.from_bytes(f.read(4), 'big')
    numImages = int.from_bytes(f.read(4), 'big')
    numRows = int.from_bytes(f.read(4), 'big')
    numCols = int.from_bytes(f.read(4), 'big')
    images = np.fromfile(f, np.uint8)
    f.close()
    images = images.reshape(numCols, numRows, numImages, order='F')
    images = np.transpose(images, [1, 0, 2])
    images = images.reshape(len(images[:, 0, 0]) * len(images[0, :, 0]), len(images[0, 0, :]), order='F')
    images = images.astype(np.float64)
    images = images / 255
    return images


def loadMNISTLabels(dir: str):
    """
    从目标文件夹中加载标签，模拟了Matlab中loadMNISTLabels.m文件的功能。

    :param dir: 目标文件 
    :return: 标签集
    """""
    f = open(dir, mode='rb')
    magic = int.from_bytes(f.read(4), 'big')
    numLabels = int.from_bytes(f.read(4), 'big')
    labels = np.fromfile(f, np.uint8)
    f.close()
    return labels


def dataset_load(dir: str):
    """
    从目标文件夹中加载数据集，模拟了Matlab中ex1_load_mnist.m文件的功能。包含对数据的加载、乱序、标准化过程。
    
    :param dir: 数据集所在文件夹 
    :return: 训练集与测试集的样本与标签
    """""
    train_x = loadMNISTImages(dir + '/train-images-idx3-ubyte')
    train_y = loadMNISTLabels(dir + '/train-labels-idx1-ubyte')
    test_x = loadMNISTImages(dir + '/t10k-images-idx3-ubyte')
    test_y = loadMNISTLabels(dir + '/t10k-labels-idx1-ubyte')
    feature_length = len(train_x[:, 0])

    # 对训练集的数据集和标签集进行拼接，乱序后进行截取
    temp_y = train_y[np.newaxis, :]
    train = np.concatenate((train_x, temp_y), axis=0)
    train = train.T
    np.random.seed()
    np.random.shuffle(train)
    train = train.T
    train_y = train[feature_length, :]
    train_y = train_y.astype(int).T
    train_x = train[0:feature_length:1, :]

    # 对测试集的数据集和标签集进行拼接，乱序后进行截取
    temp_y = test_y[np.newaxis, :]
    test = np.concatenate((test_x, temp_y), axis=0)
    test = test.T
    np.random.seed()
    np.random.shuffle(test)
    test = test.T
    test_y = test[feature_length, :]
    test_y = test_y.astype(int).T
    test_x = test[0:feature_length:1, :]

    # 对训练数据集标准化
    s = np.std(train_x, axis=1)
    s_new = s + 0.1
    m = np.mean(train_x, axis=1)
    train_x = train_x - m[:, np.newaxis]
    train_x = train_x / s_new[:, np.newaxis]

    # 对测试数据集标准化
    test_x = test_x - m[:, np.newaxis]
    test_x = test_x / s_new[:, np.newaxis]
    return train_x, train_y, test_x, test_y


def get_hypothesis(theta, x):
    return 0


def softmax(theta: ndarray, train_x: ndarray, train_y: ndarray, class_num: int, feature_num: int, sample_num: int):
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

    # 将标签从1开始
    train_y = train_y + 1
    test_y = test_y + 1
    class_num = 10
    feature_num = len(train_x[:, 0])
    sample_num = len(train_x[0, :])
    theta = np.random.random((feature_num, class_num - 1))
    theta_1 = theta[:, 0]
    softmax(theta, train_x, train_y, class_num, feature_num, sample_num)
    print(sample_num)
