'''
    Тест генетического алгоритма: подбираем параметры многочлена

'''

import math
import BaseGenetic
import numpy as np
import random

A = np.array((1,2,3))
B = np.array((2,3,4))
С = A % B
def Sigmoid(x):
    return 1/(1 + math.exp(-x))

def Polinom(x, A, x0 = 0):
    xx = x - x0
    NFact = 1
    x = 1
    Sum = 0
    for i, a in enumerate(A):
        if i > 0:
            NFact = NFact * i
            x*= xx

        Sum += a * x / NFact

    return Sum

def Teilor(F, x0, x):
    FX0 = F(x0)
    A = (FX0, FX0**2, FX0**3, FX0**4, FX0**5, FX0**6, FX0**7, FX0**8)

    return F(x), Polinom(x, A, x0)

class TTeilorGenetic(BaseGenetic.TBaseGenetic):
    def __init__(self):
        BaseGenetic.TBaseGenetic.__init__(self, HromosomLen = 8, FixedGroupsLeft=0, StopFlag=2, TheBestListSize = 50, StartPopulationSize = 50, PopulationSize = 100)

    def TestHromosom(self, Hr, Id):
        Error = 0
        for x in range(10):
            X = (x - 5) / 10
            Error += abs(Sigmoid(X) - Polinom(X, Hr))

        return 10000 * Error

    def GenerateHromosom(self, GetNewID = True):
        return (np.random.random(size = self.HromosomLen) - 0.5) * 2


if True:
    Gn = TTeilorGenetic()
    Gn.PDeath = 0.85
    Gn.PMutation = 0.75 # probability of the mutation for any individuals. Or, if >=1, the number of individuals that will die in each generation
    Gn.PMultiMutation = 0.3
    Gn.PCrossingover = 0
    Gn.StopFlag = 0
    Gn.Start()

    print('Поколений ', Gn.Generation)

# Для сравнения пытаемся решить задачу случайным подбором.
LastError = 9999999999
i = 0
Error = 0
while(True):
    A = np.random.random(8)


    Error = 0
    for x in range(10):
        X = (x - 5) / 10
        Error += abs(Sigmoid(X) - Polinom(X, A))

    Error *= 10000

    if Error < LastError:
        print(i, Error)
        LastError = Error

    i+= 1

for x in range(10):
    X = (x - 5) / 10
    print(Sigmoid(X), Polinom(X, Gn.TheBestList[0]))


