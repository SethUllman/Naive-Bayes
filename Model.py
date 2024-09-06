class Model:
	def __init__(self, trainingFolds, testFold):
		self.trainingFolds = trainingFolds
		self.testFold = testFold

		self.classProbs = {}
		self.conditionalProps = {}

	#finds the probability of the classes possible values: step 1 in algorithm
	def findClassProbs(self):
		pass

	#finds the probability of features given a class: main algorithm
	def findConditionalProbs(self):
		pass
	
	def train(self):
		print("training model")
		findClassProbs()
		findConditionalProbs()
