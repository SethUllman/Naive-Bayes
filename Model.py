import pandas as pd

class Model:
    def __init__(self, testFold, trainingFolds, className):
        self.trainingFolds = pd.concat(trainingFolds)
        self.testFold = testFold
        self.className = className

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


    def findConditionalProbs(self, d, ignoreList):
        #iterate through every value in the data set by row and column
        for index, row in self.trainingFolds.iterrows():
            for col in self.trainingFolds.columns:
                value = row[col]
                classVal = row[self.className]

                #ignore unimportant columns such as synthetic keys
                if col in ignoreList:
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
                for value in self.conditionalProps[className][col]:
                    probability = (self.conditionalProps[className][col][value] + 1) / (classCount + d)
                    self.conditionalProps[className][col][value] = probability

    def train(self):
        print("training model")
        self.findClassProbs()
        self.findConditionalProbs(1, ["Sample code number"])
        print(self.conditionalProps)
        print("--------------------------------")
