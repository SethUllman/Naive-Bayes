import pandas as pd
import os


class DataHandler:
	def __init__(self, filePath, columnNames):
		self.filePath = filePath
		self.workingData = pd.read_csv(self.filePath, delimiter=',', header=None, names=columnNames)
		print("df created")
		
		print(self.workingData.head())

	def clean(self):
		pass

	def addNois(self):
		pass

	def separateSets(self):
		pass
