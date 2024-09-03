from DataHandler import DataHandler
from Model import Model

class Main:
	def __init__(self, filePath, columnNames):
		self.handler = DataHandler(filePath, columnNames)
		self.trainingSet, self.testingSet = self.handler.separateSets()
		self.noisyTrainingSet = None
		self.noisyTestingSet = None
		self.testResults = None
		self.cleanModel = Model(trainingSet)
		self.noisyModel = Model(noisyTraingSet)

	def train(self):
		pass

	def test(self):
		pass

	def visualizeResults(self):
		pass

