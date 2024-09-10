from Main import Main
import numpy as np

print("Training Cancer")


cleanAlphaResults = {}
noisyAlphaResults = {}

for i in range(0, 100):
    cancer = Main("./data/Breast Cancer Wisconsin Data.data", ["Sample code number", "Clump Thickness", "Uniformity of Cell Size", "Uniformity of Cell Shape", "Marginal Adhesion", "Singel Epithelial Cell Size", "Bare Nuclei", "Bland Chromatin", "Normal Nucleoli", "Mitoses", "Class"], "Class", ["Sample code number", "Class"], [], i)

    cleanAlphaResults[i] = np.sum(np.diagonal(cancer.cleanModels.values))
    noisyAlphaResults[i] = np.sum(np.diagonal(cancer.noisyModels.values))

print(max(cleanAlphaResults, key=cleanAlphaResults.get))
print(max(noisyAlphaResults, key=noisyAlphaResults.get))

#print("Training Glass")
#glass = Main("./data/Glass data ML assignment.data", ["Id number", "RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe","Type of glass"], "Type of glass", ["Id number", "Type of glass"], ["RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe"])

#print("")
#print("Training Votes")
#votes = Main("./data/House Votes 84 Data.data", ["Class Name", "handicapped-infants", "water-project-cost-sharing", "adoption-of-the-budget-resolution", "physician-fee-freeze", "el-salvador-aid", "religious-groups-in-schools", "anti-satellite-test-ban", "aid-to-nicaraguan-contras", "mx-missile", "immigration", "synfuels-corporation-cutback", "education-spending", "superfund-right-to-sue", "crime", "duty-free-exports", "export-administration-act-south-africa"], "Class Name", ["Class Name"], [])

#print("")
#print("Training Iris")
#iris = Main("./data/Iris data from ecat.montana.data", ["sepal length in cm", "sepal width in cm", "petal length in cm", "petal width in cm", "class"], "class", ["class"], ["sepal length in cm", "sepal width in cm", "petal length in cm", "petal width in cm"])

#print("")
#print("Training Soybean")
#soybean = Main("./data/Soybean small data from ecat.montana.data", ["Data", "Plant stand", "Precipitation", "Temperature", "Hail", "Crop history", "Area damages", "Severity", "Seed treatment", "Germination", "Plant growth", "Leaves", "Leaf spots halo", "Leaf spots margin", "Leaf spot size", "Leaf shred", "Leaf malformation", "Leaf mildew", "Stem", "Lodging", "Stem cankers", "Canker lesion", "Fruiting bodies", "External decay", "Class"], "Class", ["Class"], [])
