from DataHandler import DataHandler
from Model import Model

class Main:
	def __init__(self, filePath, columnNames):
		self.handler = DataHandler(filePath, columnNames)
		self.cleanFolds = self.handler.separateSets(self.handler.workingData)
		self.noisyFolds = self.handler.separateSets(self.handler.addNoise())
		print(self.cleanFolds)
		print("--------------")
		print(self.noisyFolds)
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

