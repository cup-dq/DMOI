import numpy as np
import sys

def getAllDs(dataMat):
    dataMat = np.array(dataMat)
    res = [[0.0 for i in range(dataMat.shape[0])] for i in range(dataMat.shape[0])]
    res = np.array(res)
    for j in range(dataMat.shape[0]):
        for i in range(j+1,dataMat.shape[0]):
            res[i][j] = np.sqrt(np.sum(np.square(dataMat[i]-dataMat[j])))
            res[j][i] = res[i][j]
    return res

def PCCCD(dataMat,labelMat,targetclass,res):
    dataMat = np.array(dataMat)
    labelMat = np.array(labelMat)
    U =[]
    V = []
    for o in range(labelMat.shape[0]):
        if(labelMat[o]==targetclass):#获取目标类的集合
            U.append(o)
        else:#获取非目标类的集合
            V.append(o)
    b=0
    while(len(U)!=0):
        G=[]
        for k in range(dataMat.shape[0]):
            d=getMinD(V,k,res)
            P=getP(U,k,d,res)
            if(len(P)>len(G)):
                G=P
        if(len(G)==0):
            break

        b=b+1
        U=delGandU(G,U)

    return b
def getMinD(V,i,res):#获得不同类之间地最小距离
    Min =sys.maxsize
    for k2 in V:
        dt = res[i][k2]
        if(dt<Min):
            Min = dt
    return Min
def getP(U,i,Min,res):#获得同一类中处于超球
    P = []
    for k1 in U:
        dt = res[i][k1]
        if(dt<Min):
            P.append(k1)
    return P
def delGandU(G,U):#删除超球中的目标类
    for ig in G:
        U.remove(ig)
    return U
def getONBtot(labelMat,dataMat,res):
    ballnum = 0
    labelMat = np.array(labelMat)
    classnum = np.unique(labelMat)
    for i in classnum:
        print(i)
        a = PCCCD(dataMat,labelMat,i,res)
        ballnum  = ballnum + a
    return ballnum/(labelMat.shape[0])
def getONBavg(labelMat,dataMat,res):
    ballratio = []

    labelMat = np.array(labelMat)
    classnum = np.unique(labelMat)
    for i in classnum:
        num = 0
        a = PCCCD(dataMat,labelMat,i,res)
        for j in labelMat:
            if(j==i):
                num=num+1
        ballratio.append(a/num)
    res = np.mean(np.array(ballratio))
    return res
def N2(dataMat,labelMat,res):
    dataMat = np.array(dataMat)
    labelMat = np.array(labelMat)
    classnum = np.unique(labelMat)
    Mindsa = sys.maxsize
    Minina = sys.maxsize
    sumsa = 0
    sumna = 0
    for i in range(dataMat.shape[0]):
        for j in range(dataMat.shape[0]):
            if(i==j):
                continue
            else:
                if(labelMat[j]!=labelMat[i]):
                    dina = res[i][j]
                    if(Minina > dina):
                        Minina = dina
                else:
                    dsa = res[i][j]
                    if(Mindsa > dsa):
                        Mindsa = dsa
        sumna = sumna + Minina
        sumsa = sumsa + Mindsa
        Minina = sys.maxsize
        Mindsa = sys.maxsize
    return (sumsa/sumna)/((sumsa/sumna)+1)
def N3(dataMat,labelMat,res):
    dataMat = np.array(dataMat)
    labelMat = np.array(labelMat)
    cou = 0
    nums = -1
    Minx = sys.maxsize
    for j in range(dataMat.shape[0]):
        for i in range(dataMat.shape[0]):
            if(i==j):
                continue
            else:
                dx = res[i][j]
                if(dx < Minx):
                    Minx = dx
                    nums = i
        Minx=sys.maxsize
        if(labelMat[nums]!=labelMat[j]):
            cou=cou+1
    return cou/dataMat.shape[0]
def T1(dataMat,labelMat,res):
    dataMat = np.array(dataMat)
    labelMat = np.array(labelMat)
    Mindij = sys.maxsize
    dnums = []
    te=[]
    ds = []

    for j in range(dataMat.shape[0]):
        for i in range(dataMat.shape[0]):
            dij = res[i][j]
            te.append(dij)
            if(i==j|labelMat[i]==labelMat[j]):
                continue
            else:
                if(dij < Mindij):
                    Mindij = dij
        ds.append(Mindij)
        Mindij =sys.maxsize
        dnums.append(te)
        te = []
    de = []
    dnums = np.array(dnums)
    for j in range(len(ds)):
        for i in range(len(ds)):
            if(i==j):
                continue
            else:
                if(ds[i] > ds[j]):
                    if(dnums[i][j]+ ds[j] > ds[i]):
                        continue
                    else:
                        de.append(j)
                else:
                    if (dnums[i][j] + ds[i] > ds[j]):
                        continue
                    else:
                        de.append(i)
    de = np.unique(np.array(de))
    hyp = len(ds)-de.shape[0]
    print(hyp)
    return hyp/dataMat.shape[0]
def LSC(dataMat,labelMat,res):
    dataMat = np.array(dataMat)
    labelMat = np.array(labelMat)
    num = 0
    Minnad = sys.maxsize
    ds = []
    da = []
    for j in range(dataMat.shape[0]):
        for i in range(dataMat.shape[0]):
            if (i == j | labelMat[i] == labelMat[j]):
                continue
            else:
                dij = res[i][j]
                if (dij < Minnad):
                    Minnad = dij
        ds.append(Minnad)
        Minnad = sys.maxsize
    for j in range(dataMat.shape[0]):
        for i in range(dataMat.shape[0]):
            if (i == j | labelMat[i] != labelMat[j]):
                continue
            else:
                dij = res[i][j]
                if (dij < Minnad):
                    Minnad = dij
        da.append(Minnad)
        Minnad = sys.maxsize
    for j in range(len(ds)):
        if(da[j]<ds[j]):
            num = num+1
    return 1-(num/(np.square(dataMat.shape[0])))
