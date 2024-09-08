import pandas as pd

class Model:
    def __init__(self, testFold, trainingFolds, className, ignoreList):
        self.trainingFolds = pd.concat(trainingFolds)
        self.testFold = testFold
        self.className = className
        self.ignoreList = ignoreList
        self.confusionMatrix = {"Correct": 0, "Incorrect": 0}

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


    def findConditionalProbs(self, d):
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
                for value in self.conditionalProps[className][col]:
                    probability = (self.conditionalProps[className][col][value] + 1) / (classCount + d)
                    self.conditionalProps[className][col][value] = probability

    def train(self):
        #driver for the training process, find all probabilities and test
        #the test fold
        print("training model...")
        self.findClassProbs()
        self.findConditionalProbs(1)
        self.test()

    def test(self):
        #classify each row in the test fold and print total correct
        #and incorrect guesses
        for index, row in self.testFold.iterrows():
            self.classify(row)

        print(self.confusionMatrix)


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
        maxClass = max(probabilities, key=probabilities.get)

        #increment correct or incorrect guess counts based on outcome
        if maxClass == row[self.className]:
            self.confusionMatrix["Correct"] += 1
        else:
            self.confusionMatrix["Incorrect"] += 1
