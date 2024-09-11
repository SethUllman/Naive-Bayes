from Main import Main
import numpy as np


cleanCancer = {}
noisyCancer = {}
cleanGlass = {}
noisyGlass = {}
cleanVotes = {}
noisyVotes = {}
cleanIris = {}
noisyIris = {}
cleanSoybean = {}
noisySoybean = {}


for i in range(2, 1000, 2):
    print(str(i/10) + "/100")
    cancer = Main("./data/Breast Cancer Wisconsin Data.data", ["Sample code number", "Clump Thickness", "Uniformity of Cell Size", "Uniformity of Cell Shape", "Marginal Adhesion", "Singel Epithelial Cell Size", "Bare Nuclei", "Bland Chromatin", "Normal Nucleoli", "Mitoses", "Class"], "Class", ["Sample code number", "Class"], [], i/10)

    cleanCancer[i/10] = np.sum(np.diagonal(cancer.cleanModels.values))
    noisyCancer[i/10] = np.sum(np.diagonal(cancer.noisyModels.values))

    glass = Main("./data/Glass data ML assignment.data", ["Id number", "RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe","Type of glass"], "Type of glass", ["Id number", "Type of glass"], ["RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe"], i/10)
    
    cleanGlass[i/10] = np.sum(np.diagonal(glass.cleanModels.values))
    noisyGlass[i/10] = np.sum(np.diagonal(glass.noisyModels.values))

    votes = Main("./data/House Votes 84 Data.data", ["Class Name", "handicapped-infants", "water-project-cost-sharing", "adoption-of-the-budget-resolution", "physician-fee-freeze", "el-salvador-aid", "religious-groups-in-schools", "anti-satellite-test-ban", "aid-to-nicaraguan-contras", "mx-missile", "immigration", "synfuels-corporation-cutback", "education-spending", "superfund-right-to-sue", "crime", "duty-free-exports", "export-administration-act-south-africa"], "Class Name", ["Class Name"], [], i/10)

    cleanVotes[i/10] = np.sum(np.diagonal(votes.cleanModels.values))
    noisyVotes[i/10] = np.sum(np.diagonal(votes.cleanModels.values))

    iris = Main("./data/Iris data from ecat.montana.data", ["sepal length in cm", "sepal width in cm", "petal length in cm", "petal width in cm", "class"], "class", ["class"], ["sepal length in cm", "sepal width in cm", "petal length in cm", "petal width in cm"], i/10)

    cleanIris[i/10] = np.sum(np.diagonal(iris.cleanModels.values))
    noisyIris[i/10] = np.sum(np.diagonal(iris.noisyModels.values))

    soybean = Main("./data/Soybean small data from ecat.montana.data", ["Data", "Plant stand", "Precipitation", "Temperature", "Hail", "Crop history", "Area damages", "Severity", "Seed treatment", "Germination", "Plant growth", "Leaves", "Leaf spots halo", "Leaf spots margin", "Leaf spot size", "Leaf shred", "Leaf malformation", "Leaf mildew", "Stem", "Lodging", "Stem cankers", "Canker lesion", "Fruiting bodies", "External decay", "Class"], "Class", ["Class"], [], i/10)

    cleanSoybean[i/10] = np.sum(np.diagonal(soybean.cleanModels.values))
    noisySoybean[i/10] = np.sum(np.diagonal(soybean.noisyModels.values))


cancer = Main("./data/Breast Cancer Wisconsin Data.data", ["Sample code number", "Clump Thickness", "Uniformity of Cell Size", "Uniformity of Cell Shape", "Marginal Adhesion", "Singel Epithelial Cell Size", "Bare Nuclei", "Bland Chromatin", "Normal Nucleoli", "Mitoses", "Class"], "Class", ["Sample code number", "Class"], [], max(cleanCancer, key=cleanCancer.get))

print("----------Clean Models-----------")
print("")
print("Cancer---------------------------")
print("alpha = " + str(max(cleanCancer, key=cleanCancer.get)))
print(cancer.cleanModels)
print("")

glass = Main("./data/Glass data ML assignment.data", ["Id number", "RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe","Type of glass"], "Type of glass", ["Id number", "Type of glass"], ["RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe"], max(cleanGlass, key=cleanGlass.get))

print("Glass----------------------------")
print("alpha = " + str(max(cleanGlass, key=cleanGlass.get)))
print(glass.cleanModels)
print("")

votes = Main("./data/House Votes 84 Data.data", ["Class Name", "handicapped-infants", "water-project-cost-sharing", "adoption-of-the-budget-resolution", "physician-fee-freeze", "el-salvador-aid", "religious-groups-in-schools", "anti-satellite-test-ban", "aid-to-nicaraguan-contras", "mx-missile", "immigration", "synfuels-corporation-cutback", "education-spending", "superfund-right-to-sue", "crime", "duty-free-exports", "export-administration-act-south-africa"], "Class Name", ["Class Name"], [], max(cleanVotes, key=cleanVotes.get))

print("Votes----------------------------")
print("alpha = " + str(max(cleanVotes, key=cleanVotes.get)))
print(votes.cleanModels)
print("")

iris = Main("./data/Iris data from ecat.montana.data", ["sepal length in cm", "sepal width in cm", "petal length in cm", "petal width in cm", "class"], "class", ["class"], ["sepal length in cm", "sepal width in cm", "petal length in cm", "petal width in cm"], max(cleanIris, key=cleanIris.get))

print("Iris-----------------------------")
print("alpha = " + str(max(cleanIris, key=cleanIris.get)))
print(iris.cleanModels)
print("")

soybean = Main("./data/Soybean small data from ecat.montana.data", ["Data", "Plant stand", "Precipitation", "Temperature", "Hail", "Crop history", "Area damages", "Severity", "Seed treatment", "Germination", "Plant growth", "Leaves", "Leaf spots halo", "Leaf spots margin", "Leaf spot size", "Leaf shred", "Leaf malformation", "Leaf mildew", "Stem", "Lodging", "Stem cankers", "Canker lesion", "Fruiting bodies", "External decay", "Class"], "Class", ["Class"], [], max(cleanSoybean, key=cleanSoybean.get))

print("Soybean--------------------------")
print("alpha = " + str(max(cleanSoybean, key=cleanSoybean.get)))
print(soybean.cleanModels)
print("")

#Noisy Data

cancer = Main("./data/Breast Cancer Wisconsin Data.data", ["Sample code number", "Clump Thickness", "Uniformity of Cell Size", "Uniformity of Cell Shape", "Marginal Adhesion", "Singel Epithelial Cell Size", "Bare Nuclei", "Bland Chromatin", "Normal Nucleoli", "Mitoses", "Class"], "Class", ["Sample code number", "Class"], [], max(noisyCancer, key=noisyCancer.get))

print("----------Noisy Models-----------")
print("")
print("Cancer---------------------------")
print("alpha = " + str(max(noisyCancer, key=noisyCancer.get)))
print(cancer.noisyModels)
print("")

glass = Main("./data/Glass data ML assignment.data", ["Id number", "RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe","Type of glass"], "Type of glass", ["Id number", "Type of glass"], ["RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe"], max(noisyGlass, key=noisyGlass.get))

print("Glass----------------------------")
print("alpha = " + str(max(noisyGlass, key=noisyGlass.get)))
print(glass.noisyModels)
print("")

votes = Main("./data/House Votes 84 Data.data", ["Class Name", "handicapped-infants", "water-project-cost-sharing", "adoption-of-the-budget-resolution", "physician-fee-freeze", "el-salvador-aid", "religious-groups-in-schools", "anti-satellite-test-ban", "aid-to-nicaraguan-contras", "mx-missile", "immigration", "synfuels-corporation-cutback", "education-spending", "superfund-right-to-sue", "crime", "duty-free-exports", "export-administration-act-south-africa"], "Class Name", ["Class Name"], [], max(noisyVotes, key=noisyVotes.get))

print("Votes----------------------------")
print("alpha = " + str(max(noisyVotes, key=noisyVotes.get)))
print(votes.noisyModels)
print("")

iris = Main("./data/Iris data from ecat.montana.data", ["sepal length in cm", "sepal width in cm", "petal length in cm", "petal width in cm", "class"], "class", ["class"], ["sepal length in cm", "sepal width in cm", "petal length in cm", "petal width in cm"], max(noisyIris, key=noisyIris.get))

print("Iris-----------------------------")
print("alpha = " + str(max(noisyIris, key=noisyIris.get)))
print(iris.noisyModels)
print("")

soybean = Main("./data/Soybean small data from ecat.montana.data", ["Data", "Plant stand", "Precipitation", "Temperature", "Hail", "Crop history", "Area damages", "Severity", "Seed treatment", "Germination", "Plant growth", "Leaves", "Leaf spots halo", "Leaf spots margin", "Leaf spot size", "Leaf shred", "Leaf malformation", "Leaf mildew", "Stem", "Lodging", "Stem cankers", "Canker lesion", "Fruiting bodies", "External decay", "Class"], "Class", ["Class"], [], max(noisySoybean, key=noisySoybean.get))

print("Soybean--------------------------")
print("alpha = " + str(max(noisySoybean, key=noisySoybean.get)))
print(soybean.noisyModels)
print("")
