# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 17:37:46 2018

@author: Jin Uk, Cho
original source: https://github.com/jgabriellima/backpropagation
"""

import random
import numpy as np
import matplotlib.pyplot as plt

# for 3d plotting
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# for new dataset and plotting decision boundary
# from sklearn.datasets import make_moons
# from preamble import *

# 같은 결과가 나오도록 시드 설정
random.seed(0)


# 난수설정 a ~ b 사이 임의 값 생성
def rand(a, b):
    return (b - a) * random.random() + a


# 시그모이드 함수.
def sigmoid(x):
    sigm = 1. / (1. + np.exp(-x))
    return sigm


# 시그모이드 함수의 미분
def dsigmoid(y):
    deriv = y * (1. - y)
    return deriv


"""
class_뉴럴넷 구성
이 뉴럴넷은 인풋-히든노드/히든노드-아웃풋노드로 구성되어있음. 즉 히든 레이어가 1층
"""

class MLP:
    # 인풋의 feature 수,히든,아웃풋노드의 개수를 정하면서 초기화
    def __init__(self, ni, nh, no):
        '''
        ni = number of input features
        nh = number of hidden nodes
        no = number of output nodes
        '''
        self.ni = ni  # 바이아스 때문에 1 더함
        self.nh = nh
        self.no = no

        # 리스트의 형태로 각 노드의 값들을 저장하기 위해 만듬
        # !주의! 바이어스를 추가하기 위해서 + 1을 함.
        #        입력/출력값 기본을 1로 놓아 바이어스도 자연스레 1
        self.ai = [1.0] * (self.ni + 1)
        self.ah = [1.0] * (self.nh + 1)
        self.ao = [1.0] * self.no

        # 가중치 매트릭스생성
        # !주의! 바이어스의 가중치를 표현하기 위해서 각 행에 가중치 값 하나씩 추가
        self.wi = np.zeros((self.ni + 1, self.nh))
        self.wo = np.zeros((self.nh + 1, self.no))

        # 업데이트할 값을 넣어줄 행렬 만들어둠
        # !주의! 바이어스의 가중치를 표현하기 위해서 각 행에 가중치 값 하나씩 추가
        self.ci = np.zeros((self.ni + 1, self.nh))
        self.co = np.zeros((self.nh + 1, self.no))

        # 각 노드에 연결이 되는 가중치를 초기값 난수로 집어넣어줌
        # !주의! 바이어스의 가중치를 표현하기 위해서 각 행에 가중치 값 하나씩 추가
        for j in range(self.nh):
            for i in range(self.ni + 1):
                self.wi[i][j] = rand(-1.0, 1.0)
        for k in range(self.no):
            for j in range(self.nh + 1):
                self.wo[j][k] = rand(-2.0, 2.0)
        print("Initialization")
        print("--------------")
        print(self.ai)
        print(self.ah)
        print(self.ao)
        print(self.wi)
        print(self.wo)
        print("--------------")

    def update(self, X):
        # forward propagation 정리
        '''
            net1 = X*W1
            h1 = sigmoid(net1)
            net2 = h1*W2
            output = sigmoid(net2)
            loss = 1/2(target - output)^2
        '''
        # 인풋 입력값
        for i in range(self.ni):
            self.ai[i] = X[i]

        # 히든노드 출력값(아웃풋 노드 입력값)
        for j in range(self.nh):
            sum = 0.0
            for i in range(self.ni + 1):
                sum = sum + self.ai[i] * self.wi[i][j]
            self.ah[j] = sigmoid(sum)

        # 아웃풋노드 출력값
        for k in range(self.no):
            sum = 0.0
            for j in range(self.nh + 1):
                sum = sum + self.ah[j] * self.wo[j][k]
            self.ao[k] = sigmoid(sum)
        return self.ao[:]

    # 백프로파게이션 과정
    # 타겟은 주어진 목표값
    # N은 학습률을 의미
    def backPropagate(self, targets, N):
        # 1. 아웃풋의 오류값 output_delta 구하기
        """참고 : 델타 = 미분결과들의 곱
           derror/doutput = -(target - output)
           doutput/dnet2 = output(1-output)
        """
        output_deltas = [0.0] * self.no
        for k in range(self.no):
            error = -(targets[k] - self.ao[k])
            output_deltas[k] = dsigmoid(self.ao[k]) * error

        # 2. 히든노드의 오류값 hidden_delta 구하기
        """참고 : 델타 = 미분결과들의 곱
           dnet2/dh1 = W2
           dh1/dnet1 = h1(1-h1)
        """
        hidden_deltas = [0.0] * self.nh
        for j in range(self.nh):
            error = 0.0
            for k in range(self.no):
                error = error + output_deltas[k] * self.wo[j][k]
            hidden_deltas[j] = dsigmoid(self.ah[j]) * error

        # 3. 아웃풋에서 히든노드 가중치 업데이트
        """참고 : 가중치 변화량 = 학습률 * (뒤)오류값 * (앞)출력값  
            dnet2dw = h 즉 히든노드의 출력값
        """
        for j in range(self.nh + 1):
            for k in range(self.no):
                change = output_deltas[k] * self.ah[j]
                self.wo[j][k] = self.wo[j][k] - N * change
                self.co[j][k] = change
                # print(N*change)
                # print(self.co[j][k])

        # 4. 히든노드에서 인풋 가중치 업데이트
        """참고 : 가중치 변화량 = 학습률 * (뒤)오류값 * (앞)출력값 
           dnet1dw = x 즉 인풋값
        """
        for i in range(self.ni + 1):
            for j in range(self.nh):
                change = hidden_deltas[j] * self.ai[i]
                self.wi[i][j] = self.wi[i][j] - N * change
                self.ci[i][j] = change

        # 현재 오류가 얼마나 되는지 값 계산
        error = 0.0
        for k in range(len(targets)):
            error = error + N * (targets[k] - self.ao[k]) ** 2
        return error

    # 테스트 기본
    def test(self, patterns):
        for p in patterns:
            print(p[0], '->', self.update(p[0]))

    # 0~1 사이 다양한 입력에 대한 prediction을 3d로 보여줌
    def graph3d(self):
        x = np.linspace(0, 1, 100)
        y = np.linspace(0, 1, 100)
        x, y = np.meshgrid(x, y)
        temp = [x, y]
        z = self.update(temp)
        fig = plt.figure()
        ax = fig.gca(projection = '3d')
        surf = ax.plot_surface(x, y, z[0], cmap=cm.viridis, linewidth=2, antialiased=False)
        plt.xlabel("INPUT 1")
        plt.ylabel("INPUT 2")
        fig.colorbar(surf, shrink = 0.3, aspect =6)
        ax.legend()
        plt.show()

    # 가중치 출력

    # 가중치 프린트
    def weights(self):
        print('Input weights:')
        print(self.wi)
        print('Output weights:')
        print(self.wo)

    # 뉴럴넷 훈련
    def train(self, patterns, iterations=1000, N=0.9):
        for i in range(iterations):
            error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.update(inputs)
                error = error + self.backPropagate(targets, N)

            # 100번마다 에러가 얼마인지 출력해보기
            if i % 100 == 0:
                print('error %-.5f' % error)

        print("--------------")
        print("Learning finished")
        print("--------------")

    def makeData(self):
        X, y = make_moons(n_samples=10, noise=0.25, random_state=3)
        Y = y.reshape(10, 1)
        X = X.tolist()
        Y = Y.tolist()
        print(X)
        dataset = []
        for a, b in zip(X, Y):
            temp = []
            temp.append(a)
            temp.append(b)
            dataset.append(temp)
        dataset = np.asarray(dataset)
        return dataset

    def fit(self, patterns, iterations=1000, N=0.9):
        for i in range(iterations):
            error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.update(inputs)
                error = error + self.backPropagate(targets, N)

        print("--------------")
        print("New model trained")
        print("--------------")
        return self

    # plot decision boundary in dataset
    def plotdb(self, model):
        X, y = make_moons(n_samples=10, noise=0.25, random_state=3)
        X = np.asarray(X)
        mglearn.plots.plot_2d_separator(model, X, fill=True, alpha=.2)
        mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
        plt.xlabel("Feature 0")
        plt.ylabel("Feature 1")
        plt.show()


if __name__ == '__main__':
    # xor 문제
    xor = [
        [[0, 0], [0]],
        [[0, 1], [1]],
        [[1, 0], [1]],
        [[1, 1], [0]]]

    NN = MLP(2, 3, 1)
    # NN.train(xor)
    # NN.test(xor)
    # NN.weights()

'''작업 진행중'''
    # newdataset = NN.makeData()
    # print(newdataset)
    # NN.train(newdataset)
    # NN.test(newdataset)



