import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable

klasser = {
    'Iris-setosa': np.array([1, 0, 0]),
    'Iris-versicolor': np.array([0, 1, 0]),
    'Iris-virginica': np.array([0, 0, 1])
}


def hent_data(first):
    """
    Deler opp dataen fra filene i et læresett (30) L[] og et treningssett (20) T[].
    Lager LK og TK for å definere den sanne klasses til L og T.
    Dersom first == True, er de første 30 til læring. Dersom den er False er de siste 30 som er til læring, og vica versa med Trening
    """
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

def sigmoid(W, L):
    """
    Regner ut sigmoid funksjonen
    """
    return (1 / (1 + np.exp(-np.dot(L, W.T))))

def grad_MSE(g, LK ,L):
    """
    Regner ut MSE gradienten
    """
    g_min_LK = g - LK
    g_mul_1_min_g = g*(1- g)
    L_trans = np.transpose(L)

    return (np.dot(L_trans, g_min_LK*g_mul_1_min_g))

def MSE(A,B):
    #Finner minimum square error for to ting
    return ((A - B) ** 2).mean(axis=1)

def trening(alpha, iterations, L ,LK, feature_nr):
    """
    Trene del: tar inn steps alpha, antal iteratione, Lære matrise, sanne klassene til lærematrisen og antall featuers som er benyttet.
    Regner ut W (weights) og mse for hver iterasjon
    """
    W = np.zeros((3, feature_nr))
    mse_verd = []

    for i in range(iterations):
        g = sigmoid(W, L)
        W = W - alpha * grad_MSE(g, LK, L).T

        mse_verd.append(MSE(LK, g).mean())

    return(W, mse_verd)

def predict(W,T):
    #Finner predicted classe for et sett T med test data

    P = sigmoid(W,T)
    if P.ndim == 1:
        return(np.argmax(P))
    else:
        return (np.argmax(P,axis=1))

def confusion_mat(pred, true):
    #Finner confusion matrisen for en prediction fra de sanne klassene

    conf = np.zeros((3,3),dtype=int)
    for i in range(len(pred)):
        conf[pred[i]][true[i]] += 1
    return (conf)

def printing_conf(conf):
    #Plotter confusion matrisen sånn at den ser bra ut

    x = PrettyTable()
    x.field_names=["Class nr:", "Iris-setosa" ,"Iris-versicolor","Iris-virginica"]
    conf = np.insert(conf, [0], [[1],[2],[3]], axis=1)
    x.add_row(conf[0])
    x.add_row(conf[1])
    x.add_row(conf[2])
    print(x)

def printing_results(W,L,LK,T,TK):
    #Regner ut prediction, confusion matrisen og error rate for læresettet og treningssettet
    print("Learning set:")
    pred = predict(W, L)
    true = np.argmax(LK, axis=1)
    conf = confusion_mat(pred, true)

    print("Confusion Matrix:")
    printing_conf(conf)
    errorL =  1 - np.sum(pred == true) / len(pred)
    print("Error rate:", "{:.2f}".format(errorL), '\n')

    print("Training set:")
    pred = predict(W, T)
    true = np.argmax(TK, axis=1)
    conf = confusion_mat(pred, true)

    print("Confusion Matrix:")
    printing_conf(conf)
    errorT = 1 - np.sum(pred == true) / len(pred)
    print("Error rate:",  "{:.2f}".format(errorT))

def feature_splitting(klasse_nr):
    #Dele opp dataene i de forskjellige featurene

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
    #plotter histogrammene til de forskjellige featurene

    set_SL, set_SW, set_PL, set_PW = feature_splitting(0)
    ver_SL, ver_SW, ver_PL, ver_PW = feature_splitting(1)
    vir_SL, vir_SW, vir_PL, vir_PW = feature_splitting(2)

    plt.subplot(2, 2, 1)
    plt.title("Sepal length in cm")
    plt.hist(set_SL, color="lightcoral", label="setosa", edgecolor='black')
    plt.hist(ver_SL, color="mediumseagreen", label="versicolor",edgecolor='black')
    plt.hist(vir_SL, color="cornflowerblue", label="virginica",edgecolor='black')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.title("Sepal width in cm")
    plt.hist(set_SW, color="lightcoral", label="setosa",edgecolor='black')
    plt.hist(ver_SW, color="mediumseagreen", label="versicolor",edgecolor='black')
    plt.hist(vir_SW, color="cornflowerblue", label="virginica",edgecolor='black')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.title("Petal length in cm")
    plt.hist(set_PL, color="lightcoral", label="setosa", edgecolor='black')
    plt.hist(ver_PL, color="mediumseagreen", label="versicolor" , edgecolor='black')
    plt.hist(vir_PL, color="cornflowerblue", label="virginica", edgecolor='black')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.title("Petal width in cm")
    plt.hist(set_PW, color="lightcoral", label="setosa", edgecolor='black')
    plt.hist(ver_PW, color="mediumseagreen", label="versicolor",edgecolor='black')
    plt.hist(vir_PW, color="cornflowerblue", label="virginica",edgecolor='black')
    plt.legend()
    plt.tight_layout()
    plt.show()

def remove_feature(X, feature):
    #fjerne en av featurene fra matrisen X. Featurene er gitt som [SL, SW, PL, PW]. NB! kommer til å endres når man har fjernet en
    for i in range(len(X)):
        del X[i][feature]
    return (X)

def main():
    iterations = 2000
    alpha = 0.005
    feature_nr = 4

    print("------------- 30 first for learning and 20 last for training  -------------")
    L, T, LK, TK = hent_data(True)
    W, mse_verd = trening(alpha,iterations,L, LK, feature_nr)
    printing_results(W,L,LK,T,TK)

    print('\n',"------------- 30 last for learning and 20 first for training  -------------")
    L, T, LK, TK = hent_data(False)
    W, mse_verd = trening(alpha,iterations,L, LK,feature_nr)
    printing_results(W, L, LK, T, TK)

    print('\n',"-------------          Histograms for features              -------------")
    histogram_plot()
    L, T, LK, TK = hent_data(True)
    print('\n', "Removing the feature Sepal width")
    feature_nr = 3
    L = remove_feature(L,1)
    T = remove_feature(T,1)

    W, mse_verd = trening(alpha,iterations,L,LK,feature_nr)
    printing_results(W, L, LK, T, TK)

    print('\n', "Removing the feature Sepal lenght")
    feature_nr = 2
    L = remove_feature(L,0)
    T = remove_feature(T,0)

    W, mse_verd = trening(alpha,iterations,L,LK,feature_nr)
    printing_results(W, L, LK, T, TK)

    print('\n', "Removing the feature Petal lenght")
    feature_nr = 1

    L = remove_feature(L,0)
    T = remove_feature(T,0)

    W, mse_verd = trening(alpha,iterations,L,LK,feature_nr)
    printing_results(W, L, LK, T, TK)

main()











