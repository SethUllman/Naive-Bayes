import pandas as pd
import random


class DataHandler:
	def __init__(self, filePath, columnNames):
        #imports file path to data and created data frame to work with
		self.filePath = filePath
		self.workingData = pd.read_csv(self.filePath, delimiter=',', header=None, names=columnNames)

	def clean(self):
		pass

	def addNoise(self):
		#determins how many features need to be shuffled (10%)
		numColumns = self.workingData.shape[1]
		numColumnsToShuffle = int(numColumns * 0.10)

		#sample 10% of features at random and determine their column name
		noisyColumns = []
		for i in range(0, numColumnsToShuffle):
			noisyColumns.append(random.sample(range(0, numColumns), numColumnsToShuffle))

		columnNames = []
		for i in range(0, len(noisyColumns)):
			columnNames.append(self.workingData.columns[i])
		
		#create a copy of the working data to add noise to
		noisyData = self.workingData.copy()

		#shuffle values for 10% of features
		for column in columnNames:
			noisyData[column] = self.workingData[column].sample(frac=1).reset_index(drop=True)

		return noisyData

	def separateSets(self, dataSet):
        #shuffle the entries in the data set
		dataSet = dataSet.sample(frac=1).reset_index(drop=True)
		folds = []
		foldCount = 10
		foldSize = len(dataSet) // foldCount
		
        #creates folds of equal size and returns list of folds
		start = 0
		for i in range(foldCount):
			if i == foldCount - 1:
				end = len(dataSet)
			else:
				end = (i + 1) * foldSize

			fold = dataSet.iloc[start:end]
			folds.append(fold)
			start = end

		return folds
