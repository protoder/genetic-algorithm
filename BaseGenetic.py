import numpy
import os

import numpy as np
import random
import pickle
import os.path
import math
from sklearn.metrics import accuracy_score
from tensorflow.test import is_gpu_available

Colab = True
try:
    from google.colab import drive
except:
    Colab = False

if Colab:
    from google.colab import drive

    # Подключаем Google drive
    drive.mount('/content/drive')
    CrPath = "/content/drive/MyDrive/Uinnopolis/"

    import sys
    sys.path.append('/content/drive/MyDrive/Uinnopolis')
else:
    Acer = not os.path.exists("E:/Uinnopolis/")
    CrPath = "C:/Uinnopolis/" if Acer else "E:/Uinnopolis/"

import os, glob

from Libs import *

print('bgn v 3.0')

Epoches = 100

activation_list = ['relu', 'elu', 'selu', 'tanh', 'sigmoid']
optimizer_list = ['sgd', 'rmsprop', 'adagrad', 'adadelta', 'adam', 'adamax', 'nadam']

Henetic = CrPath + 'Henetic/'
ResPath = Henetic + 'Res.npy'
StatePath = Henetic + "state.npy"
BestPath = Henetic + "best"
BestNetPath = Henetic + "BestNet"
BestWeightsPath = Henetic + "BestWeights"

class TBaseGenetic:
    '''
        Предусмотрено восстановление работы из архива. Но для востановления надо установить переменную
        self.TryLoadOnStart = True

        HromosomLen - размер хромосомы. Надо иметь ввиду, что оно должно быть на 2 больше необходимого.
                      0-й элемент хромосомы - признак того, что она изменена и ее надо пересчитать
                      1-й - порядковый уникальный номер хромосомы
        StopFlag: 0 - never stop
                  1 - stop after n populations
                  2 - stop when Metric is More or Less then MetricLimit

        GenGroupSize - для удобства, гены можно разбить на группы. Например, относящиеся к одному слою.
                  При этом хромосому можно прорешейпить [Nbit/GenGroupSize, GenGroupSize]
                  Понятно, что количество ген в хромосоме должно делиться на GenGroupSize

        FixedGroupsLeft - сколько групп слева не могут быть разбиты кроссинговером

        TheBestListSize - количество элементов Best- списка. В нем хранятся лучшие хромосомы. По мере появления старых
                 новые удаляются. Хромосомы из этого списка не участвуют в размножении. При длительных отсутствиях
                 улучшений они могут быть добавлены в текущую популяцию
        StartPopulationSize  При запуске сперва создается заданное количество случайных хромосом
        PopulationSize - в данной системе используется популяция фиксированного размера
    '''
    def __init__(self, HromosomLen, GenGroupSize = 1, FixedGroupsLeft = 0, StopFlag = 0, TheBestListSize = 50,
                 StartPopulationSize = 50, PopulationSize = 100):
        self.StartPopulationSize = StartPopulationSize # стартовое количество хромосом
        self.PopulationSize = PopulationSize # постоянное количество хромосом


        # The list of the best result for any time
        self.TheBestListSize = TheBestListSize # the number of storing best hromosoms
        self.TheBestList = [None]*TheBestListSize # The list of the copy of best Hromosoms
        self.TheBestValues = np.zeros(TheBestListSize)  # The list of the best results
        self.BestValueCnt = 0

        # Alive hromosoms and there Ratings
        self.Hromosoms = [0]*PopulationSize #
        self.HromosomRatingValues = []
        self.ArgRating = [] # упорядоченный список индексов рейтинга. Порядок по возрастанию, \
                            # вне зависимости от флага InverseMetric

        self.StopFlag = StopFlag
        self.InverseMetric = True  # метрика убывает, т.е. оптимизация на убывание
        self.Metric = 10000000   # текущее значение метрики
        self.MetricLimit = 1     # При достижении этого зхначения - остановка при StopFlag == 2
        self.FirstStep = True
        self.GenerationsLimit = 100 # Стоп при достижени этого лимита при StopFlag == 1

        self.Generation = 0

        self.PMutation = 0.2 # probability of the mutation for any individuals. Or, if >=1, the number of individuals that will die in each generation
        self.PMultiMutation = 0.1 # the probability of an any bit to be mutated, in the case of the mutation has occurred
        self.PDeath = 0.2 # probability of the death. Or, if >=1, the number of individuals that will die in each generation
        self.PCrossingover = 0.5
        self.PairingAttempts = 8 # попытки скрещивания. Скрещивание неудачно, если в результате рождается хромосома, которая уже была.

        self.GenGroupSize = GenGroupSize
        self.HromosomLen = HromosomLen  # В хромосому добавляем два флага. 1 - признак измененности. 2 - ссылка на доп. данные

        self.FixedGroupsLeft = FixedGroupsLeft + 3

        self.ReportStep = 1

        self.HrList = {}

        self.StoredPath = 'copy/copy.dat'
        self.StorePeriod = 1

        self.TryLoadOnStart = False

    # Методы, которые скорее всего придется перекрывать в своей реализации

    # Возвращает рейтинг хромосомы. Хромосома - только значимая часть ( то есть без левых 2-х байт)
    # Ее номер вынесен в отдельные параметр Id
    def TestHromosom(self, Hr, Id):
        return random.random()

    def PublicResult(self, Hr, FileName): #метод вызывается для вывода результата хромосомы. Например, может быть тест на тестовых данных
                                #и публикация результата
        pass

    # При сохранении метод может передать дополнительный список значений для сохранения
    # Как их интерпретировать, определить надо в LoadFromStorer
    def AddToStored(self):
        return ()

    def LoadFromStorer(self, StoredList):
        pass

    def Save(self):
        StoreList = [
            self.TheBestList,
            self.TheBestValues,  # The list of the best results
            self.Hromosoms,
            self.HromosomRatingValues,
            self.ArgRating,
            self.Metric,
            self.FirstStep,
            self.Generation,
            self.HrList,
            self.BestValueCnt]

        AddList = self.AddToStored()
        StoreList.extend(AddList)

        with open(fr'{self.StoredPath}', 'wb') as FileVar:
            pickle.dump(StoreList, FileVar)

    def Load(self):
        try:
            with open(self.StoredPath, 'rb') as FileVar:
                StoreList = pickle.load(FileVar)

            self.TheBestList = StoreList[0]
            self.TheBestValues = StoreList[1]
            self.Hromosoms = StoreList[2]
            self.HromosomRatingValues = StoreList[3]
            self.ArgRating = StoreList[4]
            self.Metric = StoreList[5]
            self.FirstStep = StoreList[6]
            self.Generation = StoreList[7]
            self.HrList = StoreList[8]
            self.BestValueCnt = StoreList[9]
            self.LoadFromStorer(StoreList[10:])
            return True
        except :
            return False

    # Вызывается после создания хромосомы каким-либо способом
    def InitNewHromosom(self, Res, GetNewID = True):
        Res[0] =  0 # Признак, что хромосома измерена.

        Res[1] = len(self.HrList)

    # Можно как-то обработать удаление хромосомы
    def BeforeDelHromosoms(self, P):
        pass

    # С хромосомой можно ксвязать какие-то данные
    def HromosomInfo(self, Hr):
        pass

    # создаем хромосому. По умолчанию - случайный массив от 0 до 1
    def GenerateHromosom(self, IsMutation):
        return  np.random.randint(252, size = self.HromosomLen)

    # Проверка корректности хромосомы. По умолчанию она бракуется, если ее Хэш уже зарегистрирован
    def TryHr(self, Hr, Info):
        Hash = hash(tuple(Hr[2:]))
        if not Hash in self.HrList:
            self.HrList[Hash] = Info
            return True
        else:
            return False

    # Системные методы. Обычно не перекрываются
    def GenerateHromosomSys2(self, IsMutation):
        return self.GenerateHromosom()

    def GenerateHromosomSys(self, IsMutation = False):
        self.GeneratedValues = []
        while True:
            Res =  self.GenerateHromosomSys2(IsMutation)

            Info = self.HromosomInfo(Res)

            if self.TryHr(Res, Info):
                break

        self.InitNewHromosom(Res)
        return Res

    def MutateHromosom(self, Hr, MutList):
        # process the mutation of the single hromosom.
        # Hr - the reference to the mutated hromosom. MutList - numpy array, the same length as the Hromosom, has a
        # random values from 0 to 1. can be use to can be used to determine which bits of a chromosome must be mutated
        # mutate.
        # returns the mutated Hromosom

        Mutations = self.GenerateHromosomSys(True)
        return np.where(MutList <= self.PMultiMutation, Mutations, Hr)

    # Спаривание 2-х хромосом
    def HromosomsPairing(self, P0, P1):
        for i in range(self.PairingAttempts):
            if random.random() > self.PCrossingover:
                Sel = np.random.rand(self.HromosomLen) > 0.5
                Res = np.where(Sel, P0, P1)
            else:
                CrosingoverPos = random.randint(self.FixedGroupsLeft, self.HromosomLen / self.GenGroupSize - 1) * self.GenGroupSize
                Res =  np.concatenate([P0[:CrosingoverPos], P1[CrosingoverPos:]])

            Info = self.HromosomInfo(Res)

            if self.TryHr(Res, Info):
                break

        if i == self.PairingAttempts:
            while True:
                Res = self.Mutate(Res)

                Info = self.HromosomInfo(Res)

                if self.TryHr(Res, Info):
                    break

        self.InitNewHromosom(Res)

        return Res
    #  скорее всего, остальные методы в перекрытой верии будут неизменны

    # returns True if the end of evalution. The result is Ready
    def Stop(self):
        if self.StopFlag == 1:
            return self.Generation >= self.GenerationsLimit
        elif self.StopFlag == 2:
            if self.InverseMetric:
                return self.Metric <= self.MetricLimit
            else:
                return self.Metric >= self.MetricLimit
        else:
            return False #never stop

    # Просчитывает те хромосомы, что менялись
    def TestHromosoms(self):

        RV = [0]*self.HromosomsCnt

        for i, H in enumerate(self.Hromosoms):
            if H[0] == 1:
                RV[i] = self.HromosomRatingValues[i]
            else:
                while True:
                    R = self.TestHromosom(H[2:], H[1])

                    if R != None:
                        RV[i] = R
                        break

                self.Hromosoms[i][0] = 1


        self.HromosomRatingValues = RV

        self.ArgRating = np.argsort(RV)

        self.Metric = RV[self.ArgRating[0 if self.InverseMetric else -1]]

    # Сохраняет лучшие значения
    def StoreTheBest(self):

        # store the best hromosoms
        HrCnt = len(self.HromosomRatingValues)

        HallHromosomsValues = np.append(self.HromosomRatingValues, self.TheBestValues[:self.BestValueCnt])

        if self.BestValueCnt > 0:
            HallHromosoms = np.concatenate([self.Hromosoms, self.TheBestList[:self.BestValueCnt]])
        else:
            HallHromosoms = self.Hromosoms

        if self.InverseMetric:
            if self.FirstStep:
                CrRating = self.ArgRating
                self.FirstStep = False
            else:
                CrRating = np.argsort(HallHromosomsValues)
        else:
            CrRating = np.argsort(HallHromosomsValues)[::-1]

        BestPos = 0

        InList = set()

        for Ind in CrRating:
            Hr = HallHromosoms[Ind]
            if hash(tuple(Hr[2:])) in InList:
                continue

            Value = HallHromosomsValues[Ind]

            InList.add(hash(tuple(Hr[2:])))

            self.TheBestList[BestPos] = Hr
            self.TheBestValues[BestPos] = Value

            if BestPos == 0:
                self.Metric = Value

            BestPos+= 1

            if BestPos == self.TheBestListSize:
                self.BestValueCnt = BestPos
                return

        self.BestValueCnt = BestPos

    # 2 типа мутаций - кросинговер (скрещивание посередине), и просто случайный обмен значений
    def Mutations(self):
        L = self.HromosomLen
        N = self.PMutation if self.PMutation > 1 else round(random.gauss(self.HromosomsCnt * self.PMutation, 1))

        Muts = np.random.randint(0, self.HromosomsCnt, N) # the list of the mutated hromosoms

        for Hr in Muts:
            self.Hromosoms[Hr] = self.Mutate(self.Hromosoms[Hr])

    def Mutate(self, Hr):
        MutList = np.random.rand(self.HromosomLen)
        MutList[0], MutList[1] = (0, 0) # берем все из мутации
        return self.MutateHromosom(Hr, MutList)

    # Убиваем хромосомы. Вероятность смерти обратно пропорциональна рейтингу
    def Deaths(self):
        MustDie = self.PDeath if self.PDeath > 1 else round(random.gauss(self.HromosomsCnt * self.PDeath, 1))
        Fun = np.abs(np.random.vonmises(0, 0.5, MustDie) * 0.95 * self.HromosomsCnt/math.pi)
        if not self.InverseMetric:
            P = [self.ArgRating[int(x)] for x in Fun]
        else:

            P = [self.ArgRating[int(x)] for x in self.HromosomsCnt - Fun]


        self.BeforeDelHromosoms(P)
        self.Hromosoms = list(np.delete(self.Hromosoms, P, 0))
        self.HromosomRatingValues = list(np.delete(self.HromosomRatingValues, P, 0))

    # Репродукция. вероятность спаривания растет с рейтингом
    def Reproductions(self):
        Cnt = self.HromosomsCnt

        Childs = self.PopulationSize - Cnt # сколько нужно детей

        if Childs <= 0:
            return

        Rating = np.argsort(self.HromosomRatingValues)

        Fun = np.abs(np.random.vonmises(0, 0.5, 2*Childs) * Cnt / math.pi)
        if self.InverseMetric:
            P = Fun.reshape((Childs, 2))
        else:
            P = (Cnt - Fun).reshape((Childs, 2))

        '''if self.InverseMetric:
            P = np.random.triangular(0, 0, Cnt, 2*Childs).reshape((Childs, 2))
        else:
            P = np.random.triangular(0, Cnt, Cnt, 2*Childs).reshape((Childs, 2))'''

        AddList = [0]*Childs
        for i, CrP in enumerate(P):
            P0 = int(CrP[0])
            P1 = int(CrP[1])

            while (P0 == P1):
                P1 = int(np.random.triangular(0, Cnt, Cnt))

            AddList[i] = self.HromosomsPairing(self.Hromosoms[Rating[P0]], self.Hromosoms[Rating[P1]])

        self.Hromosoms.extend(AddList)

        '''
        Sum = 0
        for i, P in enumerate(zip(P0, P1)):
            CrP0 = int(P[0])
            CrP1 = int(P[1])

            while CrP0 == CrP1:
                Cr

            for iR, R in enumerate(Ratings):
                Sum = Sum + R

                if Sum >= P0:
                    Parent0 = iR
                    break

            for iR, R in enumerate(Ratings):
                Sum = Sum + R

                if Sum >= P1:
                    Parent1 = iR
                    break

            Hromosoms[i + self.HromosomsCnt] = HromosomsPairing(Hromosoms[P0], Hromosoms[P1])
            '''
    def Report(self, G, M):
        print(f'Поколение {G:5} : {M}')#, end='\r')

    # Методы придется перекрывать при нелинейной структуре хромосом (ссылки на дополнительные данные и пр.)
    def TryLoad(self):
        if os.path.isfile(self.StoredPath) and self.TryLoadOnStart and self.Load():

            return len(self.Hromosoms)

        return 0

    # Главный метод запуска алгоритма
    def Start(self):
        # Generate Hromosoms List

        # Проверка, можно ли восстановить состояние
        NeedNew = self.StartPopulationSize - self.TryLoad()

        if NeedNew > 0: # при старте создаем случайную популяцию
            self.Hromosoms = [self.GenerateHromosomSys() for i in range(NeedNew)]
            self.HromosomRatingValues = self.GeneratedValues
            IsFirst = True
        else:
            IsFirst = False
            print('Прочитано')
            print()

        Hear = 0  # счетчик поколений без изменений
        VasReplacing = False

        while not self.Stop():
            LastMetric = self.Metric

            # Тестирование, запоминание лучших

            self.TestHromosoms()
            self.StoreTheBest()

            # Если давно не было изменений
            Hear += 1 # счетчик поколений без изменений
            if self.Generation % self.ReportStep == 0 or LastMetric != self.Metric:
                self.Report(self.Generation, self.Metric)

                if LastMetric != self.Metric:
                    Hear = 0 # Признак, что на этом шаге были изменения
                    VasReplacing = False # признак, что срабатывала реакция на долгое неизмеение метрики

            if Hear >= 10:# and not VasReplacing:
                if self.BestValueCnt >= self.HromosomsCnt - 2:
                    if self.InverseMetric:  # снимаем HrLen лучших
                        From = None  # м.б. self.BestValueCnt
                        To = self.HromosomsCnt - 2
                    else:
                        From = -self.HromosomsCnt + 2
                        To = None

                    self.Hromosoms[:self.HromosomsCnt - 2] = self.TheBestList[From:To]
                    self.HromosomRatingValues[:self.HromosomsCnt - 2] = self.TheBestValues[From:To]
                    self.Hromosoms[self.HromosomsCnt - 2:] = [self.GenerateHromosomSys() for i in range(2)]

                    if len(self.GeneratedValues) == 2:
                        self.HromosomRatingValues[self.HromosomsCnt - 2:] = self.GeneratedValues
                else:
                    if self.InverseMetric: # снимаем HrLen лучших
                        From = None # м.б. self.BestValueCnt
                        To = -self.BestValueCnt-2
                    else:
                        From = self.BestValueCnt+2
                        To = None

                        #self.Hromosoms = list(np.array(self.Hromosoms)[self.ArgRating[self.TheBestListSize:]])

                    Slide = self.ArgRating[From:To]
                    self.Hromosoms = list(np.array(self.Hromosoms)[Slide])
                    self.Hromosoms.extend(self.TheBestList) # объединяем лучшие хромосомы с BestList

                    self.Hromosoms.extend([self.GenerateHromosomSys() for i in range(2)]) # свежая кровь


                    self.HromosomRatingValues = list(np.array(self.HromosomRatingValues)[Slide])
                    self.HromosomRatingValues.extend(self.TheBestValues)

                    if len(self.GeneratedValues) == 2:
                        self.HromosomRatingValues.extend(self.GeneratedValues)
                    else:
                        self.HromosomRatingValues.extend([0]*2)

                Hear = 0
                VasReplacing = True
            else:
                if not IsFirst:
                    self.Deaths() # Хромосомы умирают

                IsFirst = False
                self.Reproductions() #Размножение

                self.Mutations() #Мутации

            self.Generation += 1 #Счетчик поколений

            if self.StorePeriod == 1 or self.Generation % self.StorePeriod == self.StorePeriod-1:
                self.Save() # сохраняем список хромосом, BestList и BestValues
    @property
    def HromosomsCnt(self):
        return len(self.Hromosoms)

    '''def RealiseTheBest(self, Limit = None):
        if self.TryLoad() > 0:
           if self.InverseMetric:
               List = zip(self.TheBestValues, self.TheBestList)
           else:
               List = zip(reversed(self.TheBestValues), reversed(self.TheBestList))

           for Value, Hr in List:
               if Limit is not None:
                   if self.InverseMetric:
                       if Value > Limit:
                           break
                   else:
                       if Value < Limit:
                           break
'''

'''
    Хромосомы сперва читает из файла. Имя файла должно содержать Hrs для хромосомы, и Values для значений
    
    Неотлаженный класс, расчитанный на работу алгоритма через Google диск на нескольких Google Colab
'''
class TDistributedGenetic(TBaseGenetic):
    def __init__(self, Paths, Seed = None):
        TBaseGenetic.__init__(self, HromosomLen = 7+2, FixedGroupsLeft=0, StopFlag=2, PopulationSize = 100)
        self.Paths = Paths

    def TestHromosoms(self):

        RV = [0]*self.HromosomsCnt

        HrList = []

        for i, H in enumerate(self.Hromosoms):
            if H[0] == 1:
                RV[i] = self.HromosomRatingValues[i]
            else:
                while True:

                    HrList.append(H)

        HrList = np.concatenate(HrList, axis=0)

        Paths = len(self.Paths) + 1 if self.MySelf else 0
        Sz = len(HrList) // Paths
        Ps = 0

        for i, File in enumerate(self.Paths):
            if i == Paths - 1 and not self.MySelf:
                np.save(File, HrList[Ps:])
            else:
                np.save(File, HrList[Ps:Ps + Sz])
            Ps+= Sz

        if self.MySelf:
            for H in HrList:
                R = self.TestHromosom(H[2:], H[1])

                if R != None:
                    RV[i] = R
                    break

                self.Hromosoms[i][0] = 1


        self.HromosomRatingValues = RV

        self.ArgRating = np.argsort(RV)
        self.Metric = RV[self.ArgRating[0]]

class TDistributedGenetic(TBaseGenetic):
    def __init__(self, Paths, ReadyHrPath = None, HrCount = 0, Seed=None):
        TBaseGenetic.__init__(self, HromosomLen=7 + 2, FixedGroupsLeft=0, StopFlag=2, PopulationSize=100)
        self.InputHr = np.load(ReadyHrPath)[0:HrCount]
        self.InputValues = np.load(Paths.ReadyHrPath('Hrs', 'Values'))
        self.InputHrPos = 0

    def GenerateHromosomSys2(self, IsMutation):
        if self.InputHrPos < len(self.InputHr) and not IsMutation:
            Res = self.InputHr[self.InputHrPos]
            self.GeneratedValues.append(self.InputValues[self.InputHrPos])
            self.InputHrPos+= 1

            return Res
        else:
            return self.GenerateHromosom()

    def TestHromosoms(self):

        RV = [0]*self.HromosomsCnt

        HrList = []

        for i, H in enumerate(self.Hromosoms):
            if H[0] == 1:
                RV[i] = self.HromosomRatingValues[i]
            else:
                while True:

                    HrList.append(H)

        HrList = np.concatenate(HrList, axis=0)

        Paths = len(self.Paths) + 1 if self.MySelf else 0
        Sz = len(HrList) // Paths
        Ps = 0

        for i, File in enumerate(self.Paths):
            if i == Paths - 1 and not self.MySelf:
                np.save(File, HrList[Ps:])
            else:
                np.save(File, HrList[Ps:Ps + Sz])
            Ps+= Sz

        if self.MySelf:
            for H in HrList:
                R = self.TestHromosom(H[2:], H[1])

                if R != None:
                    RV[i] = R
                    break

                self.Hromosoms[i][0] = 1


        self.HromosomRatingValues = RV

        self.ArgRating = np.argsort(RV)
        self.Metric = RV[self.ArgRating[0]]


    def TestHromosoms(self):

        RV = [0]*self.HromosomsCnt

        HrList = []

        for i, H in enumerate(self.Hromosoms):
            if H[0] == 1:
                RV[i] = self.HromosomRatingValues[i]
            else:
                while True:

                    HrList.append(H)

        HrList = np.concatenate(HrList, axis=0)

        Paths = len(self.Paths) + 1 if self.MySelf else 0
        Sz = len(HrList) // Paths
        Ps = 0

        for i, File in enumerate(self.Paths):
            if i == Paths - 1 and not self.MySelf:
                np.save(File, HrList[Ps:])
            else:
                np.save(File, HrList[Ps:Ps + Sz])
            Ps+= Sz

        if self.MySelf:
            for H in HrList:
                R = self.TestHromosom(H[2:], H[1])

                if R != None:
                    RV[i] = R
                    break

                self.Hromosoms[i][0] = 1


        self.HromosomRatingValues = RV

        self.ArgRating = np.argsort(RV)
        self.Metric = RV[self.ArgRating[0]]

# Класс подбора параметров ML классификаторов или регрессоров
class TMLGenetic(TBaseGenetic):
    def __init__(self, HromosomLen, X, Y, XTest = None, Seed = None, Debug = False, Verbose = 0, GenGroupSize = 1,
                 FixedGroupsLeft = 0, StopFlag = 0, TheBestListSize = 30, StartPopulationSize = 15, PopulationSize = 30):
        super().__init__(HromosomLen, GenGroupSize = GenGroupSize,
                 FixedGroupsLeft = FixedGroupsLeft, StopFlag = StopFlag, TheBestListSize = TheBestListSize, StartPopulationSize = StartPopulationSize, PopulationSize = PopulationSize)
        np.random.seed(Seed)
        self.Seed = Seed
        self.X, self.Y = X, Y
        self.XTest = XTest
        self.Debug = Debug
        self.WinSize = int(len(X) * 0.2)
        self.Verbose = Verbose
        self.MLParams = {}
        self.MetricLimitForPublic = 0.94
        self.StoredPath = f'{CrPath}copy/copy.dat'

        self.PredictAfterFit = True

        self.FitParams = {}
        self.FixedWinStart = True
        self.ResFileName = 'RES'

    def BeforeFit(self, X1, Y1, ValidX, ValidY, FreeHr, FitParams, Verbose):
        return FitParams

    def ProcessHistory(self, History):
        pass

    def GetMetric(self, History, model):
        return None

    def Fit(self, model, X, Y, ValidX, ValidY, Verbose, Hr):
        if 'X' in self.FitParams:
            self.FitParams['X'] = X
        elif 'x' in self.FitParams:
            self.FitParams['x'] = X

        if 'Y' in self.FitParams:
            self.FitParams['Y'] = Y
        elif 'y' in self.FitParams:
            self.FitParams['y'] = Y

        if 'validation_data' in self.FitParams:
            self.FitParams['validation_data'] = (ValidX, ValidY)

        if 'verbose' in self.FitParams:
            self.FitParams['verbose'] = Verbose

        History = model.fit(**self.FitParams)

        if History is not None:
            self.ProcessHistory(History)

        return self.GetMetric(History, model)

    def Augmentation(self, X, Y, K, random, LevelK, Hr):
        return TimeAugmentation(X, Y, K, random=random,
                              LevelK=LevelK, UseMain=Hr[0] > 0.9)

    def Predict(self, model, ValidX, Hr):
        return model.predict(ValidX)

    def GetModel(self, Hr):
        return None

    def MLTestHromosom(self, Hr, X, Y, XTest, WinSize, Verbose, FixedWinStart = 0):
        if FixedWinStart is not None:
            Start = FixedWinStart
            FreeHr = Hr[3:]
        else:
            Start = int(Hr[4] * (len(X) - WinSize))
            FreeHr = Hr[3:]

        X1, Y1 = self.Augmentation(np.delete(X, Start + np.arange(WinSize), axis=0),
                                  np.delete(Y, Start + np.arange(WinSize), axis=0),
                                  K=1 + int(Hr[1] * 20), random=int(Hr[0] * 65535),
                                  LevelK=0.01 + Hr[2] / 3.3, Hr = Hr[3:])

        ValidX, ValidY = self.Augmentation(X[Start:Start + WinSize],
                                          Y[Start:Start + WinSize],
                                          K=20, random=24, LevelK=0.1, Hr = Hr[3:])

        model = self.GetModel(FreeHr)

        self.FitParams = self.BeforeFit(X=X1, Y=Y1, ValidX=ValidX, ValidY=ValidY, Hr=FreeHr,
                                        FitParams=self.FitParams, Verbose=Verbose)
        History = self.Fit(model=model, X=X1, Y=Y1, ValidX=ValidX, ValidY=ValidY, Verbose=Verbose, Hr=FreeHr)

        #if self.PredictAfterFit:
            # make predictions for test data
        y_pred = self.Predict(model, ValidX, FreeHr)

        accuracy = accuracy_score(ValidY, y_pred)

        print('Accuracy: %.2f%%' % (accuracy * 100.0))

        if XTest is not None and (self.InverseMetric and accuracy < self.MetricLimitForPublic) or (not self.InverseMetric and accuracy > self.MetricLimitForPublic):
            y_pred = self.Predict(model, XTest, FreeHr)
            return accuracy, y_pred
        else:
            return accuracy, None


    def TestHromosom(self, Hr, Id):
        if not self.Debug:
            CrValue, TestPredict = self.MLTestHromosom(Hr, X=self.X, Y=self.Y, XTest=self.XTest,
                                                                WinSize = self.WinSize, Verbose = self.Verbose,
                                                                FixedWinStart = self.FixedWinStart, **self.MLParams)

            if TestPredict is not None:
                #WriteCsv(fr'{CrPath}XGB{int(Id)}_{CrValue:.4f}.csv', self.YTest, Test)
                np.save(fr'{CrPath}{self.ResFileName}{int(Id)}_{CrValue:.4f}.npy', Hr)
        else:
            CrValue = 1 - random.random() / 100
        return CrValue

    def GenerateHromosom(self, GetNewID=True):
        return np.random.random(size=self.HromosomLen)

    def Start(self):
        self.GPU = is_gpu_available()
        super().Start()
