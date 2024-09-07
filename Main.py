from DataHandler import DataHandler
from Model import Model

class Main:
    def __init__(self, filePath, columnNames, className):
        #initialize data handler, clean data, and separate into 10 folds
        self.handler = DataHandler(filePath, columnNames)
        self.cleanFolds = self.handler.separateSets(self.handler.workingData)
        self.noisyFolds = self.handler.separateSets(self.handler.addNoise())
        self.className = className
        
        #create 10 trained models for both the clean and noisy data
        self.cleanModels = self.train(self.cleanFolds)
        self.noisyModels = self.train(self.noisyFolds)


    #Uses the 10 folds passed in order to perform cross validation,
    #creating 10 models which each use 1 of the folds as a test set
    def train(self, folds):
        models = []

        for i in range(10):
            model = Model(folds[0], folds[1:], self.className)
            model.train()
            models.append(model)
            folds.append(folds[0])
            folds.pop(0)

        return models

    def test(self):
        pass

    def visualizeResults(self):
        pass

