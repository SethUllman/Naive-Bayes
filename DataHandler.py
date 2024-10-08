import pandas as pd
import numpy as np
import random
import math
pd.set_option('future.no_silent_downcasting', True)


class DataHandler:
    def __init__(self, filePath, columnNames, ignoreList, continuousColumns):
        #imports file path to data and created data frame to work with
        self.filePath = filePath
        self.continuousColumns = continuousColumns
        self.workingData = pd.read_csv(self.filePath, delimiter=',', header=None, names=columnNames)
        self.columnNames = columnNames
        self.ignoreList = ignoreList

        if filePath == "./data/Breast Cancer Wisconsin Data.data":
            self.clean_breast_cancer()
        elif filePath == "./data/Glass data ML assignment.data":
            self.clean_glass()
        elif filePath == "./data/House Votes 84 Data.data":
            self.clean_votes()
        elif filePath == "./data/Iris data from ecat.montana.data":
            self.clean_iris()
        elif filePath == "./data/Soybean small data from ecat.montana.data":
            self.clean_soybean()
        self.discretize()

    def clean_breast_cancer(self):
        # Converts ? to NaN
        self.workingData.replace('?', pd.NA, inplace=True)

        # Convert column 6 to int
        self.workingData.iloc[:, 6] = pd.to_numeric(self.workingData.iloc[:, 6], errors='coerce')

        # Calculate mean for column 6 based on Class
        benign_mean = round(self.workingData[self.workingData.iloc[:, 10] == 2].iloc[:, 6].mean(),2)
        malignant_mean = round(self.workingData[self.workingData.iloc[:, 10] == 4].iloc[:, 6].mean(),2)

        # assign missing values in 'Bare Nuclei' column based on Class
        for i in range(len(self.workingData)):
            if pd.isna(self.workingData.iloc[i, 6]):
                if self.workingData.iloc[i, 10] == 2:  # Benign class
                    self.workingData.iloc[i, 6] = benign_mean
                elif self.workingData.iloc[i, 10] == 4:  # Malignant class
                    self.workingData.iloc[i, 6] = malignant_mean

        # Return cleaned data
        return self.workingData

    def clean_glass(self):
        self.filePath = "./data/Glass data ML assignment.data"
        self.workingData = pd.read_csv(self.filePath, delimiter=',', header=None, names=self.columnNames)
        return self.workingData


    def clean_iris(self):
        self.filePath = "./data/Iris data from ecat.montana.data"
        self.workingData = pd.read_csv(self.filePath, delimiter=',', header=None, names=self.columnNames)

        self.workingData.replace({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}, inplace=True)
        return self.workingData

    def clean_soybean(self):
        columnNames = [
            'Class', 'Date', 'Plant stand', 'Precipitation', 'Temperature', 'Hail', 'Crop history',
            'Area damaged', 'Severity', 'Seed treatment', 'Germination', 'Plant growth', 'Leaves',
            'Leaf spots halo', 'Leaf spots margin', 'Leaf spot size', 'Leaf shred', 'Leaf malformation',
            'Leaf mildew', 'Stem', 'Lodging', 'Stem cankers', 'Canker lesion', 'Fruiting bodies', 'External decay'
        ]
        self.filePath = "./data/Soybean small data from ecat.montana.data"
        self.workingData = pd.read_csv(self.filePath, delimiter=',', header=None, names=self.columnNames)

        self.workingData.replace({'D1': 1, 'D2': 2, 'D3': 3, 'D4': 4}, inplace=True)

        return self.workingData

    def clean_votes(self):
        columnNames = [
            'Class Name', 'handicapped-infants', 'water-project-cost-sharing',
            'adoption-of-the-budget-resolution', 'physician-fee-freeze', 'el-salvador-aid',
            'religious-groups-in-schools', 'anti-satellite-test-ban', 'aid-to-nicaraguan-contras',
            'mx-missile', 'immigration', 'synfuels-corporation-cutback', 'education-spending',
            'superfund-right-to-sue', 'crime', 'duty-free-exports', 'export-administration-act-south-africa'
            ]
        self.filePath = "./data/House Votes 84 Data.data"
        self.workingData = pd.read_csv(self.filePath, delimiter=',', header=None, names=self.columnNames)
        # Replace '?' with NaN for easier processing
        self.workingData.replace('?', pd.NA, inplace=True)

        # Convert 'y' and 'n' to binary values 1 (y) and 0 (n)
        self.workingData.replace({'y': 1, 'n': 0}, inplace=True)

        # Handle missing values by class (democrat or republican)
        for column in self.workingData.columns[1:]:  # Skip the 'Class Name' column
            missing_count = self.workingData[column].isnull().sum()

            # assign missing values based on the class (democrat/republican)
            for cls in ['democrat', 'republican']:
                cls_subset = self.workingData[self.workingData['Class Name'] == cls]
                cls_value_prob = cls_subset[column].value_counts(normalize=True)

                # assign missing values based on the conditional probability of the class
                for idx in self.workingData[self.workingData[column].isna()].index:
                    if self.workingData.loc[idx, 'Class Name'] == cls:
                        self.workingData.loc[idx, column] = np.random.choice(
                            cls_value_prob.index, p=cls_value_prob.values
                        )

        # Apply one-hot encoding for categorical variables
        self.workingData = pd.get_dummies(self.workingData, columns=self.workingData.columns[1:], drop_first=True)
        # One last clean to convert all to ints
        self.workingData.replace({'democrat': 1, 'republican': 0}, inplace=True)
        # Convert boolean values to integers
        self.workingData = self.workingData.astype(int)

        return self.workingData

    def discretize(self):
        #for every column containing continuous values, bin them with pd.cut()
        for column in self.continuousColumns:
            dataPoints = self.workingData[column].dropna().shape[0]
            bins = math.sqrt(dataPoints)
            self.workingData[column] = pd.cut(self.workingData[column], bins=int(bins), labels=False)

    def addNoise(self):
        #determins how many features need to be shuffled (10%)

        #sample 10% of features at random and determine their column name
        filteredList = [col for col in self.workingData.columns if col not in self.ignoreList]
        numColumnsToShuffle = max(1, int(len(filteredList) * 0.10))
        noisyColumns = random.sample(filteredList, numColumnsToShuffle)

        #create a copy of the working data to add noise to
        noisyData = self.workingData.copy()

        #shuffle values for 10% of features
        for column in noisyColumns:
            shuffledColumn = noisyData[column].sample(frac=1, random_state=42).values
            noisyData[column] = shuffledColumn
        
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
