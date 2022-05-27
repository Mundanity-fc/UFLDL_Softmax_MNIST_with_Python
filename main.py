import math
import numpy as np
from numpy import ndarray
from scipy.optimize import minimize


def loadMNISTImages(dir: str):
    """
    从目标文件中加载数据，模拟了 Matlab 中 loadMNISTImages.m 文件的功能。
    
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
    从目标文件夹中加载标签，模拟了 Matlab 中 loadMNISTLabels.m 文件的功能。

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
    从目标文件夹中加载数据集，模拟了 Matlab 中 ex1_load_mnist.m 文件的功能。包含对数据的加载、乱序、标准化过程。
    
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


def softmax(theta: ndarray, train_x: ndarray, train_y: ndarray):
    """
    对给定的 theta 值与训练集使用 softmax 计算损失函数与梯度值，模拟了 softmax_regression.m 中的功能
    
    :param theta: θ，模型参数
    :param train_x: 训练集样本
    :param train_y: 训练集标签
    :return: 损失函数值与梯度值
    """""
    m = len(train_x[0, :])
    n = len(train_x[:, 0])

    # 修正 theta 形状
    theta = theta.reshape((n, -1), order='C')
    num_class = len(theta[0, :]) + 1
    f = 0
    g = theta * 0

    # h0 为计算 hypothesis [即h_θ(x)] 过程的中间变量
    h0 = np.dot(theta.T, train_x)
    zeros = np.zeros([1, m])
    h0 = np.concatenate((h0, zeros), axis=0)
    h0 = np.exp(h0)
    sum = np.sum(h0, axis=0)
    for i in range(0, m):
        for j in range(0, num_class):
            h0[j, i] = h0[j, i] / sum[i]
    # 得到 h_θ(x)，为 10 x 60000 的向量，表示每一样本在进行了softmax函数处理后对于某一标签的概率
    h_theta = h0

    # 计算损失函数值 f
    for i in range(0, m):
        for j in range(0, num_class):
            if train_y[i] == j + 1:
                f = f + math.log(h_theta[j, i], 2)
    f = f * -1

    # 计算梯度 g
    # g = - [X*(1(y==k)*P(y=k|x,theta))]
    # 令(1(y==k)*P(y=k|x,theta)) = g_1 为，60000 * 10 的向量
    g_1 = np.zeros([m, num_class])
    for i in range(0, m):
        for j in range(0, num_class):
            if train_y[i] == j + 1:
                g_1[i][j] = 1 - h_theta[j][i]
            else:
                g_1[i][j] = 0 - h_theta[j][i]
    g = np.dot(train_x, g_1)
    g = g[:, 0:9:1]
    g = g * -1
    return f, g


def get_softmax_loss(theta: ndarray, train_x: ndarray, train_y: ndarray):
    """
    用于 Scipy.optimize.minimize 的函数，只返回 softmax 的损失函数值
    
    :param theta: θ，模型参数
    :param train_x: 训练集样本
    :param train_y: 训练集标签
    :return: 损失函数值
    """""
    f, g = softmax(theta, train_x, train_y)
    return f


def get_softmax_gradient(theta: ndarray, train_x: ndarray, train_y: ndarray):
    """
    用于 Scipy.optimize.minimize 的函数，只返回 softmax 的梯度值
    
    :param theta: θ，模型参数
    :param train_x: 训练集样本
    :param train_y: 训练集标签
    :return: 梯度值
    """""
    f, g = softmax(theta, train_x, train_y)
    return g.flatten()


def get_accuracy(theta: ndarray, x: ndarray, y: ndarray):
    """
    对数据集进行准确度验证，模拟了 Matlab 中 multi_classifier_accuracy.m 文件的功能

    :param theta: θ，模型参数
    :param x: 数据集的样本
    :param y: 数据集的标签
    :return: 准确度 0~1
    """""
    correct = 0
    result = np.dot(theta.T, x)
    max = np.argmax(result, axis=0)
    max = max + 1
    for i in range(len(x[0, :])):
        if max[i] == y[i]:
            correct += 1
    accuracy = correct / len(x[0, :])
    return accuracy


def main():
    """
    主函数
    
    """""
    train_x, train_y, test_x, test_y = dataset_load('./mnist_dataset')
    zeros = np.zeros([1, len(train_x[0, :])])
    train_x = np.concatenate((train_x, zeros), axis=0)
    zeros = np.zeros([1, len(test_x[0, :])])
    test_x = np.concatenate((test_x, zeros), axis=0)
    # 将标签从1开始
    train_y = train_y + 1
    test_y = test_y + 1
    class_num = 10
    feature_num = len(train_x[:, 0])
    sample_num = len(train_x[0, :])
    # 随机初始化 theta
    theta = np.random.random((feature_num, class_num - 1))
    theta = theta * 0.001
    # 设置优化器参数
    opt = {'maxiter': 200, 'disp': True}
    # 对函数 get_softmax_loss 进行优化，初始参数为 theta ，传递无关参数 train_x 与 train_y，应用 opt 中设置，使用 L-BFGS-B 方法（即 minFunc.m
    # 中的默认方法），通过 get_softmax_gradient 获取梯度
    result = minimize(fun=get_softmax_loss, x0=theta, args=(train_x, train_y), options=opt, method='L-BFGS-B',
                      jac=get_softmax_gradient)
    # 修正 theta 形状
    theta = result.x.reshape((feature_num, -1), order='C')
    zeros = np.zeros([feature_num, 1])
    theta = np.concatenate((theta, zeros), axis=1)
    # 进行验证
    print("在训练集上验证的准确率为：" + str(get_accuracy(theta, train_x, train_y)))
    print("在测试集上验证的准确率为：" + str(get_accuracy(theta, test_x, test_y)))


if __name__ == '__main__':
    main()
