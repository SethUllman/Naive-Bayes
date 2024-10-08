import pprint
import pandas as pd
from collections import defaultdict

class Model:
    def __init__(self, testFold, trainingFolds, className, ignoreList, alpha):
        self.trainingFolds = pd.concat(trainingFolds)
        self.testFold = testFold
        self.className = className
        self.ignoreList = ignoreList
        self.alpha = alpha
        self.confusionMatrix = defaultdict(lambda: defaultdict(int)) #{Class1: {Actual1: count, Actual2: count}, Class2: ...}

        self.classCounts = {} #{Class1: count, Class2: count ...}
        self.classProbs = {}  #{Class1: prob, Class2: prob}
        self.conditionalProps = {}  #{Class1: {Value1: Prob, Value2: Prob}, Class2: {}}

    def findClassProbs(self):
        length = len(self.trainingFolds)

        #get a count of each class value
        counts = self.trainingFolds[self.className].value_counts()

        #convert counts dataframe to a dictionary
        countDict = dict(counts)
        countDict = {k: int(v) for k, v in countDict.items()}
        self.classCounts = countDict

        #using our counts and total training entries, find probabilities
        #for all Class values
        for key, value in countDict.items():
            self.classProbs[key] = value / length


    def findConditionalProbs(self):
        #iterate through every value in the data set by row and column
        for index, row in self.trainingFolds.iterrows():
            for col in self.trainingFolds.columns:
                value = row[col]
                classVal = row[self.className]

                #ignore unimportant columns such as synthetic keys
                if col in self.ignoreList:
                    continue

                #keep a count of each feature value occurance and
                #what the class value is for the occurance
                if classVal in self.conditionalProps:
                    if col in self.conditionalProps[classVal]:
                        if value in self.conditionalProps[classVal][col]:
                            self.conditionalProps[classVal][col][value] += 1
                        else:
                            self.conditionalProps[classVal][col][value] = 1
                    else:
                        self.conditionalProps[classVal][col] = {value: 1}
                else:
                    self.conditionalProps[classVal] = {col: {value: 1}}

        #use our dictionary of counts and convert them to probabilities
        for className in self.conditionalProps:
            classCount = self.classCounts[className]

            for col in self.conditionalProps[className]:
                d = len(self.conditionalProps[className][col])
                for value in self.conditionalProps[className][col]:
                    probability = (self.conditionalProps[className][col][value] + self.alpha) / (classCount + d)
                    self.conditionalProps[className][col][value] = probability

    def train(self):
        #driver for the training process, find all probabilities and test
        #the test fold
        self.findClassProbs()
        self.findConditionalProbs()
        matrix = self.test()
        return matrix

    def test(self):
        #classify each row in the test fold and print total correct
        #and incorrect guesses
        for index, row in self.testFold.iterrows():
            self.classify(row)
        
        self.confusionMatrix = dict(self.confusionMatrix)
        dfConfusion = pd.DataFrame(self.confusionMatrix).fillna(0).astype(int)
        return dfConfusion


    def classify(self, row):
        #find the probability of a class given all attributes
        probabilities = {}

        for key in self.classProbs:
            probability = self.classProbs[key]

            for col, value in row.items():
                if col in self.ignoreList:
                    continue
                   
                #multiply current probabilit based on the conditional probability dictionary
                if row[col] in self.conditionalProps[key][col]:
                    probability = probability * self.conditionalProps[key][col][row[col]]

            probabilities[key] = probability

        #find the classification with the largest probability
        predictedClass = max(probabilities, key=probabilities.get)
        actualClass = row[self.className]
        self.confusionMatrix[predictedClass][actualClass] += 1

		
        
		
