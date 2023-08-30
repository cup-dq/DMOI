# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import distance as dist
import init
import complexity as cm
from sklearn import svm
import metric
import os
import random
import PCCCD as pc
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier
import datetime
def getFlist(file_dir):
    for root, dirs, files in os.walk(file_dir):
        pass
        # print('root_dir:', root)  #当前路径
        # print('sub_dirs:', dirs)   #子文件夹
        # print('files:', files)     #文件名称，返回list类型
    return files

def ONBCounter(data,disname,malset,leset):
    data = np.array(data)
    Alldatadis = {}
    for adis in range(data.shape[0]):
        if disname == "马氏距离":
            try:
                Alldatadis[adis] = dist.get_mahalanobis(data,adis,-1)
            except:
                Alldatadis[adis] = 0
        elif(disname == "标准化欧式距离"):
            Alldatadis[adis] = dist.standEclean(data,adis,-1)
        else:
            return False
    Alldatadis_1 = sorted(Alldatadis.items(),key=lambda x:x[1])

    countma = 0
    countle = 0
    keda = [i[0] for i in Alldatadis_1]
    for ke in range(len(keda)-1):
        if((keda[ke] in malset)&(keda[ke+1] in leset)):
            countma = countma + 1
        elif((keda[ke] in leset)&(keda[ke+1] in malset)):
            countle = countle + 1
        else:
            pass
    if(keda[len(keda)-1] in leset):
        countle = countle+1
    elif(keda[len(keda)-1] in malset):
        countma = countma +1
    else:
        pass
    return ((countle/len(leset))+(countma/len(malset)))/((len(malset)/len(leset))+1),((countle/len(leset))+(countma/len(malset)))/2
def GetPartView(train,malset,leset,dis):
    train = np.array(train)
    Partresdismax= []
    Partresdismean = []
    for disi in range(len(dis)):
        Partresmax = []
        for it in range(train.shape[1]):
            train_plus = np.delete(train,it,axis=1)
            OnBPartAdd,OnBPartAddNo=ONBCounter(train_plus,dis[disi],malset,leset)
            Partresmax.append([OnBPartAdd,OnBPartAddNo])
        Partresdismax.append(max(Partresmax))
        Partresdismean.append(np.mean(Partresmax,axis=0))
    return Partresdismax,Partresdismean

def GetAllView(train,malset,leset,dis):
    train = np.array(train)
    Partresdis = []
    for disi in range(len(dis)):
        PartAdd,PartAddNo = ONBCounter(train, dis[disi], malset, leset)
        Partresdis.append([PartAdd,PartAddNo])
    return Partresdis
def Getmean(res):
    res = np.array(res)

    finalres = []
    for iu in range(res.shape[1]):
        teres = []
        for ic in range(res.shape[2]):
            ae = []
            for ie in range(res.shape[0]):
                ae.append(res[ie][iu][ic])
            aemean = np.mean(ae)
            teres.append(aemean)
        finalres.append(teres)

    return finalres
def Othermean(res):
    res = np.array(res)
    return np.mean(res,axis=0)
def get_clf(algorithm):
    if algorithm == "CART":
        clf = DecisionTreeClassifier(criterion='gini')
    elif algorithm == "ID3":
        clf = DecisionTreeClassifier(criterion='entropy')
    elif algorithm == "NB":
        clf = GaussianNB()
    elif algorithm == "SVM":
        clf = svm.SVC(probability=True)
    elif algorithm == "RF":
        clf = RandomForestClassifier()
    elif algorithm == "KNN":
        clf = KNeighborsClassifier()
    elif algorithm == "LDA":
        clf = LinearDiscriminantAnalysis()
    elif algorithm == "GBDT":
        clf = GradientBoostingClassifier()
    elif algorithm == "MLP":
        clf = MLPClassifier(alpha=1, max_iter=1000)
    else:
        raise("无分类器")
    return clf
def Classfication(train,test,classifierlist,ma,le):
    train = np.array(train)
    test = np.array(test)

    clf_model = get_clf(classifierlist)
    clf_model.fit(train[:,:-1],train[:,-1])
    pre = clf_model.predict(test[:,:-1])
    pre_probe = clf_model.predict_proba(test[:,:-1])
    return metric.getResStandard(test[:,-1],pre,pre_probe,ma,le)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

        dis = ["马氏距离","标准化欧式距离"]
        data1 = open("./ONE复杂度量对比实验.txt", "w+")
        data2 = open("./ONBavg遍历的过程.txt","w+")
        data3 = open("./每次遍历选取的下标集合.txt","w+")

        file_dir = './data'  # 你的文件路径
    # file_name1 = getFlist(file_dir)
    # for ie in range(len(file_name1)):
    #     file_name = file_name1[ie]
    #    A = init.GetData(file_dir+'/'+file_name[ie])
        file_name='abalone9-18.csv'#./data文件夹下的数据集名
        A = init.GetData(file_dir+'/'+file_name)
        ma,le, = init.GetLessMore(A.iloc[:,-1])
        #筛选数据量小于1000的数据集参与,如果不想要这个就注释掉。
        # if(A.shape[0]>800):
        #     continue
        # else:
        #     pass
        #
        print(file_name+"的实验结果",file=data1)
        print(file_name+"的实验结果",file=data2)
        # A = init.GetData('./data/zoo-3.csv')
        # ma, le, = init.GetLessMore(A.iloc[:, -1])
        d = datetime.datetime.now().strftime('%d%H%M%S')

        np.random.seed(int(d))
        rseed = np.random.randint(0,2022)
        print(rseed)
        kn = 10
        kf2 = StratifiedKFold(n_splits=kn, shuffle=True, random_state=rseed)
        classlist = ["CART","ID3","NB","SVM","RF","KNN","GBDT","MLP"]
        AllEachDataONB = []
        PartEachDataONB = []
        EachdataONBavg = []
        EachdataONBtot = []
        PartEachDataONBmax = []
        PartEachDataONBmean = []
        ip = 0
        classres = {}
        F1Set,F2Set,F3Set,F4Set,N1Set,N2Set,N3Set,N4Set,T1Set,D3Set = ([] for i in range(10))
        F1Max = []
        F1Mean = []
        ij= 0
        for ic1 in range(len(classlist)):
            classres[classlist[ic1]] = []
        for traindex1, teindex1 in kf2.split(A.iloc[:, :-1], A.iloc[:, -1]):
            print("{0}次的切割选取结果".format(ij),file=data3)
            ij=ij+1
            print("测试集的下标为:",file=data3)
            print(teindex1,file=data3)
            train = A.iloc[traindex1]
            test = A.iloc[teindex1]
            traindata = train.iloc[:,:-1]
            labeldata = train.iloc[:,-1]
            malset, leset = init.Getmalsetandleset(train.iloc[:, -1], ma, le)
            AllEDO= GetAllView(train.iloc[:, :-1], malset, leset, dis)
            AllEachDataONB.append(AllEDO)
            print(file_name+"全局视野的ONB复杂度量计算完毕")
            PartEDOmax,PartEDOmean=GetPartView(train.iloc[:, :-1], malset, leset, dis)
            PartEachDataONBmax.append(PartEDOmax)
            PartEachDataONBmean.append(PartEDOmean)
            print(file_name + "局部视野的ONB复杂度量计算完毕")
            #计算ONBtot,ONBavg
            res = pc.getAllDs(traindata)
            print("矩阵计算完毕")
            ONBtot = pc.getONBtot(labeldata, traindata, res)
            print(file_name +"ONBtot计算完毕")
            ONBavg = pc.getONBavg(labeldata, traindata, res)
            print(file_name + "ONBavg计算完毕")
            ip=ip+1
            print("数据集遍历次数",ip,file=data2)
            print("ONBavg",ONBavg,file=data2)
            print("ONBtot",ONBtot,file=data2)
            EachdataONBavg.append(ONBavg)
            EachdataONBtot.append(ONBtot)
            #ONBtot,ONBavg结束
            for ic in range(len(classlist)):
                classres[classlist[ic]].append(Classfication(train,test,classlist[ic],ma,le))
            cm1 = cm.Complexity(train)
            F1Set.append(cm1.F1())
            F1Max.append(np.max(cm1.F1()))
            F1Mean.append(np.mean(cm1.F1()))
            F2Set.append(cm1.F2())
            F3Set.append(cm1.F3())
            F4Set.append(cm1.F4())
            N1Set.append(cm1.N1())
            N2Set.append(cm1.N2())
            N3Set.append(cm1.N3())
            N4Set.append(cm1.N4())
            T1Set.append(cm1.T1())
            D3Set.append(cm1.D3_value())
        featurename = A.columns.tolist()
        T1res = Othermean(T1Set)
        D3res = Othermean(D3Set)
        F1res = Othermean(F1Set)
        F1maxres = Othermean(F1Max)
        F1meanres = Othermean(F1Mean)
        F2res = Othermean(F2Set)
        F3res = Othermean(F3Set)
        F4res = Othermean(F4Set)
        N1res = Othermean(N1Set)
        N2res = Othermean(N2Set)
        N3res = Othermean(N3Set)
        N4res = Othermean(N4Set)
        dict_F1 = {}
        for ife in range(len(featurename)-1):
            dict_F1[featurename[ife]] = F1res[ife]
        AEO = Getmean(AllEachDataONB)
        PEOMx = Getmean(PartEachDataONBmax)
        PEOMn = Getmean(PartEachDataONBmean)
        EOA = Othermean(EachdataONBavg)#存储十次的ONBavg,然后做平均
        EOT = Othermean(EachdataONBtot)#存储十次的ONBtot,然后做平均
        for ic2 in range(len(classlist)):
            temean = Othermean(classres[classlist[ic2]])
            classres[classlist[ic2]] = temean
        print("ONB链式度量(本实验的主要方法)复杂度为:",file=data1)
        for id in range(len(dis)):
            print(dis[id],file=data1)
            print("全局视野,有平衡比的复杂度为",AEO[id][0],file=data1)
            print("全局视野,无平衡比的复杂度为",AEO[id][1],file=data1)
            print("局部视野求最大值，有平衡比的复杂度为", PEOMx[id][0],file=data1)
            print("局部视野求最大值，无平衡比的复杂度为",PEOMx[id][1],file=data1)
            print("局部视野求平均值，有平衡比的复杂度为", PEOMn[id][0], file=data1)
            print("局部视野求平均值，无平衡比的复杂度为", PEOMn[id][1], file=data1)
        dataa = np.array(A)
        calss = np.unique(dataa[:,-1])
        for key,value in dict_F1.items():
            print("{0}的F1为{1}、".format(key,value),file=data1)
        print("F1最大值是{0}".format(F1maxres),file=data1)
        print("F1均值是{0}".format(F1meanres),file=data1)
        print("F2为:{0}、F3为:{1}、F4为:{2}、N1为:{3}、N2为:{4}、N3为:{5}、N4为:{6}、T1为:{7}".format(F2res,F3res,F4res,N1res,N2res,N3res,N4res,T1res),file=data1)
        for ip in range(len(calss)):
            print("{0}类别的D3值为{1}".format(calss[ip],D3res[ip]),file=data1)
        print("ONBavg为%.8f"%EOA,file=data1)
        print("ONBtot为%.8f"%EOT,file=data1)
        print("性能指标如下：",file=data1)
        for key,value in classres.items():
            print(key+"分类器:",file=data1)
            print("指标为G-mean为 {0} , F1为 {1} , Kappa为 {2} , auc为{3} , Aupr为 {4} ，mcc为 {5}".format(value[0],value[1],value[2],value[3],value[4],value[5]),file=data1)
        data1.close()
        data2.close()
        data3.close()







# See PyCharm help at https://www.jetbrains.com/help/pycharm/
