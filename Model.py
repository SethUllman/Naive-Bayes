class Model:
	def __init__(self, trainingFolds, testFold):
		self.trainingFolds = trainingFolds
		self.testFold = testFold

	def train(self):
		print("training model")
