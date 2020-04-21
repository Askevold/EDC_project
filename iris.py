import numpy as np
import matplotlib.pyplot as plt

klasser = {
    'Iris-setosa': np.array([1, 0, 0]),
    'Iris-versicolor': np.array([0, 1, 0]),
    'Iris-virginica': np.array([0, 0, 1])
}

#deler opp dataen fra filene i et læresett (30) L[] og et treningssett (20) T[]
def hent_data(first):
    class_1 =open("class_1").read().split('\n')
    class_2= open("class_2").read().split('\n')
    class_3 = open("class_3").read().split('\n')

    if (first):
        split = 30
        L = class_1[0:split] + class_2[0:split] + class_3[0:split]
        T = class_1[split:50] + class_2[split:50] + class_3[split:50]
    else:
        split = 20
        L = class_1[split:50] + class_2[split:50] + class_3[split:50]
        T = class_1[0:split] + class_2[0:split] + class_3[0:split]


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

def trening(alpha, iterations, L ,LK, feature_nr):
# trene del
    W = np.zeros((3, feature_nr))
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

def printing_results(W,L,LK,T,TK):
    print("Trainingset:")
    pred = predict(W, L)
    true = np.argmax(LK, axis=1)
    conf = confusion_mat(pred, true)

    print("Confusion Matrix:")
    printing_conf(conf)
    print("Correct:", np.sum(pred == true) / len(pred), '\n')

    print("Testset:")
    pred = predict(W, T)
    true = np.argmax(TK, axis=1)
    conf = confusion_mat(pred, true)

    print("Confusion Matrix:")
    printing_conf(conf)
    print("Correct:", np.sum(pred == true) / len(pred))

#Setosa har klasse nr. 0 ,versicolor har 1, virginica har 2
def feature_splitting(klasse_nr):
    class_1 =open("class_1").read().split('\n')
    class_2= open("class_2").read().split('\n')
    class_3 = open("class_3").read().split('\n')

    data = class_1[0:50] + class_2[0:50] + class_3[0:50]
    for i in range(len(data)):
        data[i] = data[i].split(',')
        for j in range(len(data[i])):
            data[i][j] = float (data[i][j])

    SL = [0]*50
    SW = [0]*50
    PL = [0]*50
    PW = [0]*50

    for i in range(50):
        SL[i] = data[i + 50*klasse_nr][0]
        SW[i] = data[i + 50*klasse_nr][1]
        PL[i] = data[i + 50*klasse_nr][2]
        PW[i] = data[i + 50*klasse_nr][3]

    return (SL, SW, PL, PW)

def histogram_plot():
    set_SL, set_SW, set_PL, set_PW = feature_splitting(0)
    ver_SL, ver_SW, ver_PL, ver_PW = feature_splitting(1)
    vir_SL, vir_SW, vir_PL, vir_PW = feature_splitting(2)

    plt.subplot(2, 2, 1)
    plt.title("Sepal length in cm")
    plt.hist(set_SL, color="lightcoral", label="setosa", edgecolor='black')
    plt.hist(ver_SL, color="darkgreen", label="versicolor",edgecolor='black')
    plt.hist(vir_SL, color="cornflowerblue", label="virginica",edgecolor='black')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.title("Sepal width in cm")
    plt.hist(set_SW, color="lightcoral", label="setosa",edgecolor='black')
    plt.hist(ver_SW, color="darkgreen", label="versicolor",edgecolor='black')
    plt.hist(vir_SW, color="cornflowerblue", label="virginica",edgecolor='black')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.title("Petal length in cm")
    plt.hist(set_PL, color="lightcoral", label="setosa", edgecolor='black')
    plt.hist(ver_PL, color="darkgreen", label="versicolor" , edgecolor='black')
    plt.hist(vir_PL, color="cornflowerblue", label="virginica", edgecolor='black')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.title("Petal width in cm")
    plt.hist(set_PW, color="lightcoral", label="setosa", edgecolor='black')
    plt.hist(ver_PW, color="darkgreen", label="versicolor",edgecolor='black')
    plt.hist(vir_PW, color="cornflowerblue", label="virginica",edgecolor='black')
    plt.legend()
    plt.tight_layout()
    plt.show()

#remove one of the features, SL = 0, SW = 1, PL = 2, PW = 3
def remove_feature(X, feature):
    for i in range(len(X)):
        del X[i][feature]
    return (X)

def main():
    iterations = 5000
    alpha = 0.0005
    feature_nr = 4

    print("------------- 30 first for training and 20 last for testing -------------")
    L, T, LK, TK = hent_data(True)
    W, mse_verd = trening(alpha,iterations,L, LK, feature_nr)
    #printing_results(W,L,LK,T,TK)


    print("------------- 30 last for training and 20 first for testing -------------")
    L, T, LK, TK = hent_data(False)
    #W, mse_verd = trening(alpha,iterations,L, LK)
    #printing_results(W, L, LK, T, TK)

    print("------------- Histograms for features -------------")
    histogram_plot()
    L, T, LK, TK = hent_data(True)
    feature_nr = 3
    L = remove_feature(L,1)
    T = remove_feature(T,1)

    W, mse_verd = trening(alpha,iterations,L,LK,feature_nr)
    printing_results(W, L, LK, T, TK)

main()











