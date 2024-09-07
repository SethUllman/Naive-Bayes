import pandas as pd

class Model:
    def __init__(self, testFold, trainingFolds, className):
        self.trainingFolds = trainingFolds
        self.testFold = testFold
        self.className = className

        self.classProbs = {}  #{Class1: prob, Class2: prob}
        self.conditionalProps = {}  #{Class1: {Value1: Prob, Value2: Prob}, Class2: {}}

    #finds the probability of the classes possible values: step 1 in algorithm
    def findClassProbs(self):
        valueCounts = []
        totalLength = 0

        #for each fold, get a count of each class value as well as total length
        for fold in self.trainingFolds:
            count = fold[self.className].value_counts()
            valueCounts.append(count)
            totalLength += len(fold)

        #sum class counts of all training folds, this creates a Series that
        #we convert to a dictionary
        totalCounts = pd.concat(valueCounts, axis=1).fillna(0).sum(axis=1)
        countDict = dict(totalCounts)
        countDict = {k: int(v) for k, v in countDict.items()}

        #using our counts and total training entries, we find probabilities
        #for all Class values
        for key, value in countDict.items():
            self.classProbs[key] = value / totalLength

    #finds the probability of features given a class: main algorithm
    def findConditionalProbs(self):
        pass

    def train(self):
        print("training model")
        self.findClassProbs()
        self.findConditionalProbs()
