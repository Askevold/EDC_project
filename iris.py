import numpy as np
import matplotlib.pyplot as plt

klasser = {
    'Iris-setosa': np.array([1, 0, 0]),
    'Iris-versicolor': np.array([0, 1, 0]),
    'Iris-virginica': np.array([0, 0, 1])
}

#deler opp dataen fra filene i et læresett (30) L[] og et treningssett (20) T[]
def hent_data():
    split = 30

    class_1 =open("class_1").read().split('\n')
    class_2= open("class_2").read().split('\n')
    class_3 = open("class_3").read().split('\n')


    L = class_1[0:split] + class_2[0:split] + class_3[0:split]
    T = class_1[split:50] + class_2[split:50] + class_3[split:50]

    LK = [0]*90
    TK = [0]*60

    for i in range(len(LK)):
        if(i<30):
            LK[i] = [1,0,0]
        elif(i<60):
            LK[i] = [0,1,0]
        else:
            LK[i] = [0,0,1]

    for i in range(len(TK)):
        if (i < 20):
            TK[i]= [1, 0, 0]
        elif (i < 40):
            TK[i]=[0, 1, 0]
        else:
            TK[i] = [0, 0, 1]

    for i in range(len(L)):
        L[i] = L[i].split(',')
        for j in range(len(L[i])):
            L[i][j] = float (L[i][j])

    for i in range(len(T)):
        T[i] = T[i].split(',')
        for j in range(len(T[i])):
            T[i][j] = float (T[i][j])

    return(L,T, LK,TK)

def plotting(parameterX, parameterY, C):
    paramX = [0]*(30*C)
    paramY = [0]*(30*C)

    farger = ['bo','rx','g1']

    for i in range(30*C):
        paramX[i] = L[i][parameterX]
        paramY[i] = L[i][parameterY]

    for i in range(C):
        plt.plot(paramX[i*30:30 * (i + 1) - 1], paramY[i*30:30 * (i + 1) - 1], farger[i], label='vv')
    plt.ylabel(parameterY)
    plt.xlabel(parameterX)
    plt.show()

SL = 0
SW = 1
PL = 2
PW = 3

#plotting(SW,SL, 3)

#calculating the sigmoid function
def sigmoid(W, L):
     return (1 / (1 + np.exp(-np.dot(L, W.T))))

#calculating the MSE gradient
def grad_MSE(g, LK ,L):
    ting1 = g - LK
    ting2 = g*(1- g)
    ting3 = np.transpose(L)

    return (np.dot(ting3, ting1*ting2))

def MSE(A,B):
    return ((A - B) ** 2).mean(axis=1)


def trening(alpha, iterations, L ,LK):
# trene del
    W = np.zeros((3, 4))
    mse_verd = []

    for i in range(iterations):
        g = sigmoid(W, L)
        W = W - alpha * grad_MSE(g, LK, L).T

        mse_verd.append(MSE(LK, g).mean())

    return(W, mse_verd)

def predict(W,T):
    P = sigmoid(W,T)
    if P.ndim == 1:
        return(np.argmax(P))
    else:
        return (np.argmax(P,axis=1))

def confusion_mat(pred, true):
    conf = np.zeros((3,3),dtype=int)
    for i in range(len(pred)):
        conf[pred[i]][true[i]] += 1
    return (conf)

def printing_conf(conf):
    print("Class |  1  |  2  |  3  |")
    for i in range(3):
        print(" ",i +1,"  |  ", end = '')
        for j in range(3):
            print(conf[i][j], " | ", end = '')
        print("\n")
def main():
    iterations = 800
    alpha = 0.005
    L, T, LK, TK = hent_data()

    W, mse_verd = trening(alpha,iterations,L, LK)

    pred = predict(W,T)
    true = np.argmax(TK, axis=1)

    conf = confusion_mat(pred, true)
    print(conf)
    printing_conf(conf)



main()










