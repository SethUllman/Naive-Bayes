from Main import Main

print("Training Cancer")
cancer = Main("./data/Breast Cancer Wisconsin Data.data", ["Sample code number", "Clump Thickness", "Uniformity of Cell Size", "Uniformity of Cell Shape", "Marginal Adhesion", "Singel Epithelial Cell Size", "Bare Nuclei", "Bland Chromatin", "Normal Nucleoli", "Mitoses", "Class"], "Class", ["Sample code number", "Class"])
print("")
print("Training Glass")
glass = Main("./data/Glass data ML assignment.data", ["Id number", "RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe","Type of glass"], "Type of glass", ["Id number", "Type of glass"])

print("")
print("Training Votes")
votes = Main("./data/House Votes 84 Data.data", ["Class Name", "handicapped-infants", "water-project-cost-sharing", "adoption-of-the-budget-resolution", "physician-fee-freeze", "el-salvador-aid", "religious-groups-in-schools", "anti-satellite-test-ban", "aid-to-nicaraguan-contras", "mx-missile", "immigration", "synfuels-corporation-cutback", "education-spending", "superfund-right-to-sue", "crime", "duty-free-exports", "export-administration-act-south-africa"], "Class Name", ["Class Name"])

print("")
print("Training Iris")
iris = Main("./data/Iris data from ecat.montana.data", ["sepal length in cm", "sepal width in cm", "petal length in cm", "petal width in cm", "class"], "class", ["class"])

print("")
print("Training Soybean")
soybean = Main("./data/Soybean small data from ecat.montana.data", ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "class"], "class", ["class"])
