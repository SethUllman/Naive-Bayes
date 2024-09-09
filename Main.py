from DataHandler import DataHandler
from Model import Model
import pandas as pd
import numpy as np

class Main:
    def __init__(self, filePath, columnNames, className, ignoreList):
        #initialize data handler, clean data, and separate into 10 folds
        self.handler = DataHandler(filePath, columnNames)
        self.cleanFolds = self.handler.separateSets(self.handler.workingData)
        self.noisyFolds = self.handler.separateSets(self.handler.addNoise())
        self.className = className
        self.ignoreList = ignoreList
        
        #create 10 trained models for both the clean and noisy data
        print("-----------Clean Models------------")
        self.cleanModels = self.train(self.cleanFolds)
        print("-----------Noisy Models------------")
        self.noisyModels = self.train(self.noisyFolds)


    #Uses the 10 folds passed in order to perform cross validation,
    #creating 10 models which each use 1 of the folds as a test set
    def train(self, folds):
        models = []
        matrices = []
        for i in range(10):
            model = Model(folds[0], folds[1:], self.className, self.ignoreList)
            matrices.append(model.train())
            models.append(model)
            folds.append(folds[0])
            folds.pop(0)


        for matrix in matrices:
            print(matrix)
            print("----------------")
        print("")
        return models

    def test(self):
        pass

    def visualizeResults(self):
        pass

