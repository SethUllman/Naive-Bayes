import pandas as pd
import os


class DataHandler:
	def __init__(self, filePath, columnNames):
		self.filePath = filePath
		self.workingData = pd.read_csv(self.filePath, delimiter=',', header=None, names=columnNames)

	def clean(self):
		pass

	def addNois(self):
		pass

	def separateSets(self):
		self.workingData = self.workingData.sample(frac=1).reset_index(drop=True)
		folds = []
		foldCount = 10
		foldSize = len(self.workingData) // foldCount
		
		start = 0
		for i in range(foldCount):
			if i == foldCount - 1:
				end = len(self.workingData)
			else:
				end = (i + 1) * foldSize

			fold = self.workingData.iloc[start:end]
			folds.append(fold)
			start = end

		return folds
