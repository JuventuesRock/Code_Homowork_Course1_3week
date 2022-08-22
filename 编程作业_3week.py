# -*- coding: utf-8 -*-
# @Time : 2022/8/22 15:09
# @Author : zhuyu
# @File : 编程作业_3week.py
# @Project : Python菜鸟教程

import numpy as np
import matplotlib.pyplot as plt
from testCases import *
from planar_utils import plot_decision_boundary,sigmoid,load_planar_dataset,load_extra_datasets

np.random.seed(1)

#加载和查看数据集
X, Y =load_planar_dataset()
print("X.shape=",X.shape) #(2,400) (feature_num,样本数)
print("Y.shape=",Y.shape) #(1,400) (label,样本数)
print("该数据集中数据有：{0}个".format(X.shape[1]))
print("数据集中第一个数据的XY坐标：X-{0}，Y-{1}".format(X.T[0][0],X.T[0][1]))

#使用matplotlib可视化
plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral) #绘制散点图
# plt.show()
# 上一语句如出现问题，请使用下面的语句：
# plt.scatter(X[0, :], X[1, :], c=np.squeeze(Y), s=40, cmap=plt.cm.Spectral) #绘制散点图

#定义神经网络的结构
def layer_size(X,Y):
    """
    确定神经网络的输入层单元个数nx、隐藏层神经元数量nh、输出层神经元ny
    :param X: 数据集 维度(输入的特征数量、训练样本数量)
    :param Y: 标签 维度(输出结果数量，训练样本数量)
    :return:(nx,nh,ny)
        nx-输入层神经元数量
        nh-隐藏层神经元数量
        ny-输出层神经元数量
    """
    nx=X.shape[0]
    nh=4 #设置隐藏层神经元个数为4 硬编码
    ny=Y.shape[0]

    return (nx,nh,ny)

#测试layer_size
print("="*15+"测试layer_size()"+"="*15)
X_asses,Y_asses=layer_sizes_test_case()
(nx,nh,ny)=layer_size(X_asses,Y_asses)
print("【测试】nx = ",nx)
print("【测试】nh = ",nh)
print("【测试】ny = ",ny)

#随机初始化模型参数
def initialize_parameters(nx,nh,ny):
    """
    随机初始化模型参数，该模型是只包含一个隐藏层(4个神经元)的浅层神经网络，一共包含两个权重层W1/b1和W2/b2
    对W1和W2采用np.random.randn进行随机初始化，对偏置单元b初始化为0即可
    :param nx: 输入层节点的数量
    :param nh: 隐藏层节点的数量
    :param ny: 输出层节点的数量
    :return: parameters - 包含初始化后参数的字典
        {
        "W1":W1 - 权重矩阵 维度(nh,nx)
        "b1":b1 - 偏向量 维度(nh,1)
        "W2":W2 - 权重矩阵 维度(ny,nh)
        "b2":b2 - 偏向量 维度(ny,1)
        }
    """
    np.random.seed(2) #指定一个随机数种子
    W1=np.random.randn(nh,nx)*0.01
    b1=np.zeros(shape=(nh,1))
    W2=np.random.randn(ny,nh)*0.01
    b2=np.zeros(shape=(ny,1))

    #使用断言确保数据维度正确
    assert W1.shape==(nh,nx)
    assert b1.shape==(nh,1)
    assert W2.shape==(ny,nh)
    assert b2.shape==(ny,1)

    parameters={
        "W1":W1,
        "b1":b1,
        "W2":W2,
        "b2":b2
    }

    return parameters

#测试initialize_parameters
print("="*15+"测试initialize_parameters()"+"="*15)
(nx_asses,nh_asses,ny_asses)=initialize_parameters_test_case()
param_asses=initialize_parameters(nx_asses,nh_asses,ny_asses)
print("【测试】W1= ",param_asses["W1"])
print("【测试】b1= ",param_asses["b1"])
print("【测试】W2= ",param_asses["W2"])
print("【测试】b2= ",param_asses["b2"])

def forward_propagation(X,parameters):
    """
    前向传播 计算网络输出结果A2
    :param X: 输入数据 维度(nx,m)
    :param parameters: 初始化模型参数的输出 也是后续循环迭代的parameters
    :return: (A2,cache)
        A2:使用sigmoid()函数计算的第二次激活后的输出结果
        cache：包含网络前向计算的结果 “Z1” "A1" "Z2" "A2"
    """
    W1=parameters["W1"]
    b1=parameters["b1"]
    W2=parameters["W2"]
    b2=parameters["b2"]

    Z1=np.dot(W1,X)+b1
    A1=np.tanh(Z1) #第一层激活函数采用比sigmoid效果更好的tanh函数
    Z2=np.dot(W2,A1)+b2
    A2=sigmoid(Z2) #第二层激活函数采用适合二分类的sigmoid激活函数
    # print(Z1.shape,A1.shape,Z2.shape,A2.shape)
    # print(ny)

    cache={
        "Z1":Z1,
        "A1":A1,
        "Z2":Z2,
        "A2":A2
    }

    #assert确保数据维度正确
    # assert Z1.shape==(nh,X.shape[1])
    # assert A1.shape==(nh,X.shape[1])
    # assert Z2.shape==(1,X.shape[1])
    assert A2.shape==(1,X.shape[1])

    return (A2,cache)

#测试forward_propagation
print("="*15+"测试forward_propagation()"+"="*15)
X_asses,param_asses=forward_propagation_test_case()
A2_asses,cache_asses=forward_propagation(X_asses,param_asses)
print("【测试】A2 = ",A2_asses)
print("【测试】cache = ",cache_asses)
print("【测试】mean value=",np.mean(cache_asses["Z1"]), np.mean(cache_asses["A1"]), np.mean(cache_asses["Z2"]), np.mean(cache_asses["A2"]))

def compute_cost(A2,Y):
    """
    计算损失cost 利用交叉熵损失函数
    :param A2: sigmoid()计算得到的模型输出
    :param Y: "True"标签向量，维度(1,样本数量)
    :return: cost - 交叉熵成本
    """

    m=Y.shape[1]

    logprobs =np.multiply(np.log(A2),Y)+np.multiply((1-Y),np.log(1-A2))
    cost=-np.sum(logprobs)/m
    cost=float(np.squeeze(cost))

    assert (isinstance(cost,float))

    return cost

#测试cost
print("="*15+"测试forward_propagation()"+"="*15)
A2_asses,Y_asses,param_asses=compute_cost_test_case()
cost_asses=compute_cost(A2_asses,Y_asses)
print("【测试】cost_asses = ",cost_asses)

def backward_propagation(paramters,cache,X,Y):
    """
    反向传播函数
    :param paramters: 包含参数W1 W2 b1 b2的字典
    :param cache: 包含前向计算变量的字典
    :param X: 输入数据 维度(nx,m)
    :param Y: 标签 维度(ny,m)
    :return: grads - 包含w和b的导数的一个字典类型的变量
    """
    m=X.shape[1]

    W1=paramters["W1"]
    W2=paramters["W2"]
    A1=cache["A1"]
    A2=cache["A2"]

    dZ2=A2 - Y
    dW2=(1/m)*np.dot(dZ2,A1.T)
    db2=(1/m)*np.sum(dZ2,axis=1,keepdims=True)
    dZ1=np.multiply(np.dot(W2.T,dZ2),1-np.power(A1,2))
    dW1=(1/m)*np.dot(dZ1,X.T)
    db1=(1/m)*np.sum(dZ1,axis=1,keepdims=True)

    grads={
        "dW1":dW1,
        "db1":db1,
        "dW2":dW2,
        "db2":db2
    }

    return grads

#测试backward_propagation
print("="*15+"测试backward_propagation()"+"="*15)
param_asses,cache_asses,X_asses,Y_asses=backward_propagation_test_case()
grads_asses=backward_propagation(param_asses,cache_asses,X_asses,Y_asses)
print("【测试】 grads =",grads_asses)

def update_parameters(parameters,grads,learning_rate=1.2):
    """
    使用反向传播计算的梯度下降更新规则更新参数parameters
    :param parameters: 待更新的参数 字典类型
    :param grads:反向传播计算的梯度 字典类型
    :param learning_rate:学习速率
    :return:parameters - 更新后的参数 字典类型
    """
    W1=parameters["W1"]
    b1=parameters["b1"]
    W2=parameters["W2"]
    b2=parameters["b2"]

    dW1=grads["dW1"]
    db1=grads["db1"]
    dW2=grads["dW2"]
    db2=grads["db2"]

    W1=W1-learning_rate*dW1
    b1=b1-learning_rate*db1
    W2=W2-learning_rate*dW2
    b2=b2-learning_rate*db2

    parameters={
        "W1":W1,
        "b1":b1,
        "W2":W2,
        "b2":b2
    }

    return parameters

#测试update_parameters
print("="*15+"测试update_parameters()"+"="*15)
param_asses,grads_asses=update_parameters_test_case()
param_update_asses=update_parameters(param_asses,grads_asses)
print("【测试】param_update = ",param_update_asses)

def nn_model(X,Y,nh,num_iteration,print_cost=False):
    """
    将上述模型构建过程整合到一个函数model中，并进行num_iteration次循环迭代训练权重参数
    :param X:数据集 维度(2,样本数)
    :param Y:标签 维度(1,样本数)
    :param n_h:隐藏层节点个数
    :param num_iteration:梯度下降循环中的迭代次数
    :param print_cost:若为True 每1000次迭代打印一次损失值
    :return:parameters - 模型训练后的参数 可以用来在测试集中预测
            costs - 记录每1000次迭代的损失值 绘制损失函数图
    """
    np.random.seed(3) #指定随机种子
    nx=layer_size(X,Y)[0]
    ny=layer_size(X,Y)[2]

    parameters=initialize_parameters(nx,nh,ny) #初始化模型参数
    W1,b1=parameters["W1"],parameters["b1"]
    W2,b2=parameters["W2"],parameters["b2"]

    costs=[]

    for epoch in range(num_iteration):
        A2,cache=forward_propagation(X,parameters)
        cost=compute_cost(A2,Y)
        grads=backward_propagation(parameters,cache,X,Y)
        parameters=update_parameters(parameters,grads,learning_rate=0.5)

        if print_cost:
            if (epoch+1)%1000==0:
                costs.append(cost)
                print("第 {0} 次循环，loss为：{1}".format(epoch+1,cost))

    return parameters

#测试nn_model
print("="*15+"测试nn_model()"+"="*15)
X_asses,Y_asses=nn_model_test_case()
param_asses=nn_model(X_asses,Y_asses,nh=4,num_iteration=10000,print_cost=True)
print("【测试】W1 = " + str(param_asses["W1"]))
print("【测试】b1 = " + str(param_asses["b1"]))
print("【测试】W2 = " + str(param_asses["W2"]))
print("【测试】b2 = " + str(param_asses["b2"]))
#注意：这里的测试数据出现警告：RuntimeWarning: overflow encountered in exps = 1/(1+np.exp(-x))

def predict(parameters,X):
    """
    使用学习后的参数来对测试集进行预测
    :param parameters: 经过nn_model学习后的参数 包含参数的字典类型变量
    :param X: 测试集输入数据(nx,m)
    :return: predictions - 模型预测的向量(红色:0/蓝色:1)
    """
    A2,cache=forward_propagation(X,parameters)
    predictions=np.round(A2) #四舍五入 大于0.5即可认为是1 蓝色 反之为0 红色

    return predictions

#测试predict
print("="*15+"测测试predict()"+"="*15)
param_asses,X_asses=predict_test_case()
predictions_asses=predict(param_asses,X_asses)
print("【测试】predictitons = ",predictions_asses)

#进行不同学习率的测试函数  复制上面nn_model
def nn_model_learningTest(X,Y,nh,num_iteration,learning_rate,print_cost=False):
    """
    将上述模型构建过程整合到一个函数model中，并进行num_iteration次循环迭代训练权重参数
    :param X:数据集 维度(2,样本数)
    :param Y:标签 维度(1,样本数)
    :param n_h:隐藏层节点个数
    :param num_iteration:梯度下降循环中的迭代次数
    :param print_cost:若为True 每1000次迭代打印一次损失值
    :return:parameters - 模型训练后的参数 可以用来在测试集中预测
            costs - 记录每1000次迭代的损失值 绘制损失函数图
    """
    np.random.seed(3) #指定随机种子
    nx=layer_size(X,Y)[0]
    ny=layer_size(X,Y)[2]

    parameters=initialize_parameters(nx,nh,ny) #初始化模型参数
    W1,b1=parameters["W1"],parameters["b1"]
    W2,b2=parameters["W2"],parameters["b2"]

    costs=[]

    for epoch in range(num_iteration):
        A2,cache=forward_propagation(X,parameters)
        cost=compute_cost(A2,Y)
        grads=backward_propagation(parameters,cache,X,Y)
        parameters=update_parameters(parameters,grads,learning_rate=learning_rate)

        if print_cost:
            if (epoch+1)%1000==0:
                costs.append(cost)
                print("第 {0} 次循环，loss为：{1}".format(epoch+1,cost))

    return parameters

#正式运行
if __name__ == '__main__':
    parameters=nn_model(X,Y,nh=4,num_iteration=10000,print_cost=True)

    #绘制边界
    plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
    plt.title("Decision Boundary for hidden layer size " + str(4))

    predictions=predict(parameters,X)
    print('准确率: %d' % float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100) + '%')

    plt.show()

    print("-------------------------------------拓展测试1----------------------------------")
    #更改隐藏层节点数量
    plt.figure(figsize=(16, 32))
    hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 50]  # 隐藏层数量
    for i, n_h in enumerate(hidden_layer_sizes):
        plt.subplot(5, 2, i + 1)
        plt.title('Hidden Layer of size %d' % n_h)
        parameters = nn_model(X, Y, n_h, num_iteration=5000)
        plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
        predictions = predict(parameters, X)
        accuracy = float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100)
        print("隐藏层的节点数量： {}  ，准确率: {} %".format(n_h, accuracy))

    plt.show()

    """
    ##【可选】探索
    当改变sigmoid激活或ReLU激活的tanh激活时会发生什么？改变的话再反向传播梯度的计算公式需要改变
    改变learning_rate的数值会发生什么  learning_rate_test=[0.001,0.01,0.05,0.1,0.5,1]
    如果我们改变数据集呢？重新训练一套模型参数 决策边界
    """
    print("-------------------------------------拓展测试2----------------------------------")
    #测试不同学习率
    plt.figure(figsize=(16,32))
    learning_rate_test=[0.001,0.01,0.05,0.1,0.5,1,2,5,10,100]
    for i,learning_rate_i in enumerate(learning_rate_test):
        plt.subplot(5,2,i+1) #创建一个3行2列子图，当前绘制的子图位置i+1
        plt.title('learning_rate_{0} =  {1}'.format(i+1,learning_rate_i))
        parameters = nn_model_learningTest(X, Y, nh=4, learning_rate=learning_rate_i,num_iteration=5000)
        plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
        predictions = predict(parameters, X)
        accuracy = float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100)
        print("learning_rate_{0}： {1}  ，准确率: {2} %".format(i+1,learning_rate_i, accuracy))

    plt.show()