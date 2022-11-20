import pandas
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, \
    BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from numpy.polynomial import Polynomial
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import recall_score, precision_score
from sklearn.metrics import classification_report
from tensorflow.test import is_gpu_available
from xgboost import XGBClassifier
from xgboost.callback import EarlyStopping
import xgboost as xgb
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical

def W():
    import warnings


def Graphic(Data, Label=None, Cnt=10, Together=True, File=None):
    if Label is None or Label == 0:
        Ind = 0
    elif Label < 0:
        Ind = -Label
    elif Label == 1:
        Ind = 693
    elif Label == 2:
        Ind = 1400
    elif Label == 3:
        Ind = 2071
    elif Label == 4:
        Ind = 2784
    elif Label == 5:
        Ind = 3489

    for i in range(Cnt):
        if not Together:
            plt.figure(figsize=(14, 7))

        # plt.scatter(Data[i + Ind], range(len(Data[i + Ind])))
        plt.plot(Data[i + Ind], marker='.')
        # plt.stem(Data[i + Ind], range(len(Data[i + Ind])))
        if not Together:
            # plt.show()
            if File is not None:
                plt.savefig(f'{File}{i + Ind}.jpg')
                plt.close()
    if Together:
        # plt.show()
        if File is not None:
            plt.savefig(f'{File}{i + Ind}.jpg')


'''
    Вернет маску для позиций, не соответствующих условию, или не данной метки 
    Fun(X, Y) должна вернуть булевый массив масок для позиций в X   
        Например return X > 0.5

'''


def GetLabelMask(X, Y, Fun):
    Mask = Fun(X, Y)
    return np.where(np.tile(Y == 1, [70, 1]).T, np.zeros_like(Mask), Mask)


def Postprocessing(X, Y, Mode=1):
    # return X, Y
    l = len(X)

    X = X.to_numpy()

    X1 = np.diff(X, axis=-1)
    # X2 = np.diff(X1, axis = -1)
    X = np.concatenate([X1, X], axis=1)

    return pandas.DataFrame(X), Y
    return pandas.DataFrame(X), Y
    L = Y[:, 1]
    neigh = NearestNeighbors(n_neighbors=len(X), metric='cosine')
    neigh.fit(X)

    distances, idxs = neigh.kneighbors(X[9:10], len(X), return_distance=True)
    Res = L[idxs]

    '''
    Не сработала попытка обрезать худшие элементы в датасете. Видимо, они соответствуют распределению и в тесте
    Cnt = (Res[0, 4000:]  == 0).sum()
    Mask0 = idxs[Res  == 0][-Cnt:]

    X.drop(Mask0, axis=0, inplace=True)
    Y = np.delete(Y, Mask0, axis = 0)
    '''
    return X, Y


def Preprocessing(df, Y=0):
    return df
    Data = df.to_numpy()

    Y = Y[:, 1]

    Res = [0] * 7
    Res[0] = np.nanpercentile(Data[Y == 0], [5, 95], axis=0).reshape(1, 2, 70)
    Res[1] = np.nanpercentile(Data[Y == 1], [5, 95], axis=0).reshape(1, 2, 70)
    Res[2] = np.nanpercentile(Data[Y == 2], [5, 95], axis=0).reshape(1, 2, 70)
    Res[3] = np.nanpercentile(Data[Y == 3], [5, 95], axis=0).reshape(1, 2, 70)
    Res[4] = np.nanpercentile(Data[Y == 4], [5, 95], axis=0).reshape(1, 2, 70)
    Res[5] = np.nanpercentile(Data[Y == 5], [5, 95], axis=0).reshape(1, 2, 70)
    Res[6] = np.nanpercentile(Data[Y == 6], [5, 95], axis=0).reshape(1, 2, 70)
    Res = np.concatenate(Res)

    Mask = Data > Res[1, 1]
    Mask = np.where(np.tile(Y == 1, [70, 1]).T, np.zeros_like(Mask), Mask)

    Data[Mask] = np.nan

    return df


def Interpolation(df, Y, Train, Columns):
    # Удаляем нули
    #    1. Заменяем их на nan, это надо для запуска полинома
    df[df < 0] = np.nan
    df[df > 1] = np.nan
    df = df.replace(to_replace=0, value=np.nan)

    # df[‘column_name’].mask(df[‘column_name’] == ‘some_value’, value, inplace = True )

    #    2. Также надо пронумеровать колонки. Тоже для полинома
    ColSet = {v: i for i, v in enumerate(Columns[4 if Columns[2] != 'id' else 3:])}
    df1 = df.rename(columns=ColSet)

    Percentiles = []

    if Train:
        df1 = Preprocessing(df1, Y)

    #    3. Интерполяция. Не будут заменены ведущие и заключительные non
    df = df1.interpolate(method='spline', limit_direction='both', order=5, axis=1)
    # df = df.interpolate(method='linear', limit_direction='both', axis=1)

    return df, Y


def ReadCsv(CrPath, DelZeros, SortCtg, Train, Au=None, RetPrc=False, PostProc=False):
    if CrPath[-4:] == '.csv':
        df = pd.read_csv(CrPath)
    else:
        Name = 'train' if Train else 'test'
        df = pd.read_csv(f"{CrPath}Data/{Name}.csv")

    # Ставим все поля в порядке возрастания времени
    Columns = df.columns.to_list()
    Columns.sort()
    # ColNames = {Columns[i]:i for i in range(len(Columns))}
    # Разбиваем на 2 фрейма. Один только временные ряды, второй - доп. информация
    df_labels = df.reindex(columns=Columns[1:3])
    df = df.reindex(columns=Columns[4 if Train else 3:])
    # df.rename(ColNames)
    df_labels = df_labels.to_numpy()

    df, df_labels = Interpolation(df, df_labels, Train, Columns)

    if PostProc > 0:
        df, df_labels = Postprocessing(df, df_labels, PostProc)

    df = df.to_numpy()

    # Увы, стандартизация тоже не зашла
    # Scaler = StandardScaler()
    # Scaler.fit(df)
    # df = Scaler.transform(df)

    if SortCtg:
        # Сортируем массив по категориями. Это удобно для анализа, хотя перед обучением придется переставлять
        SortInd = np.argsort(df_labels[:, 1], axis=0)

        # удаляем нули
        df = df[SortInd]

        df_labels = df_labels[SortInd]

    return df, df_labels


def WriteCsv(File, Labels, arr):
    # ID = np.arange(len(arr))
    arr = np.concatenate((Labels.reshape((len(Labels), 1)), arr.reshape((len(arr), 1))), axis=1)
    DF = pd.DataFrame(arr, columns=("id", "crop"))
    DF.to_csv(File, sep=',', index=False)


# ВНИМАНИЕ! Y должен быть формата [Samples, Размер] Даже если размер равен 1 ( то есть не (2000,), а (2000, 1)  )
def TimeAugmentation(XTest, YTest, K, random=None, LevelK=0.05, UseMain=True, Ver=0, Train=True, SinMode=0):
    # NewX = [0]*(K)
    np.random.seed(random)
    NewX = [XTest * (np.random.random_sample(size=XTest.shape) * LevelK + 0.975) for i in
            range(K - (1 if UseMain else 0))]

    if UseMain:
        NewX.append(XTest)

    XTest = np.concatenate(NewX)

    if YTest is not None:
        YTest = np.tile(YTest, [K, 1])

        return XTest, YTest
    else:
        return XTest


from scipy import signal


def Filter(Data, K=1 / 32):
    b, a = signal.butter(8, K, 'lowpass')  # Конфигурационный фильтр 8 указывает порядок фильтра
    return signal.filtfilt(b, a, Data)


def TestForest(X, Y, X_test=None, y_test=None, N=175, RandomState=0):
    if X_test is None:
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.1)
    else:
        X_train, y_train = X, Y

    Frst = RandomForestClassifier(random_state=RandomState, n_estimators=N)
    R = Frst.fit(X_train, y_train)
    pred = Frst.predict(X_test)

    res = recall_score(y_test, pred, average="macro", zero_division=0)

    print(res)
    print(0, classification_report(pred, y_test))

    return pred, res


'''
    0 - GradientBoostingClassifier
    1 - HistGradientBoostingClassifier
    2 - = BaggingClassifier(KNeighborsClassifier(),
...                            max_samples=0.5, max_features= 0.5)
'''


def TestClassify(X, Y, X_test0=None, y_test0=None, X_test1=None, XTestLen=2071, KAu=0, N=175, RandomState=0, ClsType=0,
                 Hr=[]):
    X_train, y_train = X, Y

    if ClsType == 0:
        Cls = GradientBoostingClassifier(random_state=RandomState)  # , n_estimators=N)
    elif ClsType == 1:
        Cls = HistGradientBoostingClassifier(random_state=RandomState)
    elif ClsType == 2:
        Cls = BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5, verbose=True)
    elif ClsType == 1:
        Cls = 0
    elif ClsType == 1:
        Cls = 0
    elif ClsType == 1:
        Cls = 0
    elif ClsType == 1:
        Cls = 0
    elif ClsType == 1:
        Cls = 0
    elif ClsType == 1:
        Cls = 0

    R = Cls.fit(X_train, y_train)
    pred = Cls.predict(X_test0)
    res = recall_score(y_test0, pred, average="macro", zero_division=0)
    print(res)
    # print(0, classification_report(pred, y_test0))

    if KAu == 0:
        if X_test1 is not None:
            pred1 = Frst.predict(X_test1)
            return pred, res, pred1
        else:
            return pred, res
    else:
        if X_test1 is not None:
            pred1 = Frst.predict(X_test1)
            predict = np.reshape(pred1, (1, KAu, XTestLen, 1))
            Res = [0] * 7
            Res[0] = (predict == 0)
            Res[1] = (predict == 1)
            Res[2] = (predict == 2)
            Res[3] = (predict == 3)
            Res[4] = (predict == 4)
            Res[5] = (predict == 5)
            Res[6] = (predict == 6)
            R = np.concatenate(Res)
            # R = R.sum(axis = 0)
            pred1 = np.argmax(R.T.sum(-2), axis=-1).reshape((R.shape[2], 1))

            return pred, res, pred1
        else:
            return pred, res


def TestForest2(X, Y, X_test0=None, y_test0=None, X_test1=None, XTestLen=2071, KAu=0, N=175, RandomState=0):
    X_train, y_train = X, Y

    Frst = RandomForestClassifier(random_state=RandomState, n_estimators=N)
    R = Frst.fit(X_train, y_train)
    pred = Frst.predict(X_test0)
    res = recall_score(y_test0, pred, average="macro", zero_division=0)
    print(res)
    # print(0, classification_report(pred, y_test0))

    if KAu == 0:
        if X_test1 is not None:
            pred1 = Frst.predict(X_test1)
            return pred, res, pred1
        else:
            return pred, res
    else:
        if X_test1 is not None:
            pred1 = Frst.predict(X_test1)
            predict = np.reshape(pred1, (1, KAu, XTestLen, 1))
            Res = [0] * 7
            Res[0] = (predict == 0)
            Res[1] = (predict == 1)
            Res[2] = (predict == 2)
            Res[3] = (predict == 3)
            Res[4] = (predict == 4)
            Res[5] = (predict == 5)
            Res[6] = (predict == 6)
            R = np.concatenate(Res)
            # R = R.sum(axis = 0)
            pred1 = np.argmax(R.T.sum(-2), axis=-1).reshape((R.shape[2], 1))

            return pred, res, pred1
        else:
            return pred, res


'''
    Создает состояние двоичного леса на основании массива Hr[7].
    Байт 0 - инициализация генератора аугментации 
         1 - начало окна 
         2 - AuK
'''


def ForestHromosom(Hr, X, Y, XTest, WinSize=100, MaxAuK=35, MaxAyLevel=0.3):
    if len(Y.shape) == 1:
        Y = Y.reshape((Y.shape[0], 1))

    np.random.seed(int(Hr[0] * 65535))
    Start = int(Hr[1] * (len(X) - WinSize))
    DelMask = Start + np.arange(100)

    if MaxAyLevel == 0:
        X1, Y1 = (np.delete(X, DelMask, axis=0), np.delete(Y, DelMask, axis=0))

        Xvalid, Yvalid = (X[Start:Start + WinSize], Y[Start:Start + WinSize])

    else:
        X1, Y1 = TimeAugmentation(np.delete(X, DelMask, axis=0),
                                  np.delete(Y, DelMask, axis=0),
                                  K=1 + int(Hr[2] * (MaxAuK - 1)), random=int(Hr[3] * 65535),
                                  LevelK=0.01 + Hr[4] * MaxAyLevel, UseMain=Hr[5] < 0.1)

        Xvalid, Yvalid = TimeAugmentation(X[Start:Start + WinSize], Y[Start:Start + WinSize],
                                          K=10, random=0, LevelK=0.2, UseMain=True)

    if XTest is not None:
        XTestLen = len(XTest)
        # XTestAu = TimeAugmentation(XTest, None, K=33, random=0, LevelK=0.2, UseMain=True)

        CrRes, CrValue, CrTest = TestForest2(X1, Y1, Xvalid, Yvalid, XTest, XTestLen, KAu=0,
                                             RandomState=int(Hr[6] * 65535))
        return CrRes, CrValue, CrTest
    else:
        CrRes, CrValue = TestForest2(X1, Y1, Xvalid, Yvalid, None, 0, KAu=33 if MaxAyLevel > 0 else 0,
                                     RandomState=int(Hr[6] * 65535))
        return CrRes, CrValue


def HGFHromosom(Hr, X, Y, XTest, WinSize=100, MaxAuK=35, MaxAyLevel=0.3):
    if len(Y.shape) == 1:
        Y = Y.reshape((Y.shape[0], 1))

    np.random.seed(int(Hr[0] * 65535))
    Start = int(Hr[1] * (len(X) - WinSize))
    DelMask = Start + np.arange(100)

    if MaxAyLevel == 0:
        X1, Y1 = (np.delete(X, DelMask, axis=0), np.delete(Y, DelMask, axis=0))

        Xvalid, Yvalid = (X[Start:Start + WinSize], Y[Start:Start + WinSize])

    else:
        X1, Y1 = TimeAugmentation(np.delete(X, DelMask, axis=0),
                                  np.delete(Y, DelMask, axis=0),
                                  K=1 + int(Hr[2] * (MaxAuK - 1)), random=int(Hr[3] * 65535),
                                  LevelK=0.01 + Hr[4] * MaxAyLevel, UseMain=Hr[5] < 0.1)

        Xvalid, Yvalid = TimeAugmentation(X[Start:Start + WinSize], Y[Start:Start + WinSize],
                                          K=10, random=0, LevelK=0.2, UseMain=True)

    if XTest is not None:
        XTestLen = len(XTest)
        # XTestAu = TimeAugmentation(XTest, None, K=33, random=0, LevelK=0.2, UseMain=True)

        CrRes, CrValue, CrTest = TestHGB(X1, Y1, Xvalid, Yvalid, XTest, XTestLen, KAu=0, RandomState=int(Hr[6] * 65535))
        return CrRes, CrValue, CrTest
    else:
        CrRes, CrValue = TestHGB(X1, Y1, Xvalid, Yvalid, None, 0, KAu=33 if MaxAyLevel > 0 else 0,
                                 RandomState=int(Hr[6] * 65535))
        return CrRes, CrValue

def ProcessXGboost(Hr, X, Y, ValidX, ValidY, TestX, GPU = None):
    if GPU is None:
        GPU = is_gpu_available()

    RA = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1, 5, 10, 50, 100]
    Methods = ['exact', 'approx', 'hist']
    CrMethod = Methods[int(Hr[10] * 3)]

    if GPU:  # and CrMethod == 'hist':
        CrMethod = 'gpu_hist'

    K = 33  # 1 + int(Hr[1] * 20)
    rnd = int(Hr[4] * 65535)
    LevelK = 0.2  # 0.01 + Hr[2] / 3.3

    eval_set = [(ValidX, ValidY)]

    model = XGBClassifier(learning_rate=0.03,
                          n_estimators=10000,
                          max_depth=6,
                          min_child_weight=6,
                          max_bin=100,
                          gamma=0,
                          subsample=0.6,
                          colsample_bytree=0.6,
                          reg_alpha=0.005,
                          objective='binary:logistic',
                          nthread=6,
                          scale_pos_weight=1,
                          seed=int(65),
                          tree_method=CrMethod,
                          random_state=65,
                          verbose=0)

    UseMetric = "mlogloss"
    es = EarlyStopping(
        rounds=100,
        save_best=True,
        maximize=False,
        data_name="validation_0",
        metric_name=UseMetric
    )

    model.fit(X, Y, eval_metric=UseMetric, eval_set=eval_set, callbacks=[es], verbose=1)
    BestIter = model.best_iteration
    # make predictions for test data
    test_pred = model.predict(ValidX)

    accuracy = accuracy_score(ValidY, test_pred)
    print('Accuracy0: %.5f%%' % (accuracy))

    if TestX is not None:
        y_pred = model.predict(TestX)
        y_pred_cat = to_categorical(y_pred)
        y_pred_cat = np.reshape(y_pred_cat, (1, len(y_pred_cat), 7))

        return test_pred, accuracy, y_pred, y_pred_cat

    return test_pred, accuracy


if __name__ == '__main__':
    ReadCsv("E:/Uinnopolis/", DelZeros=True, SortCtg=True)