import pickle
from matplotlib import pyplot as plt


# Coordinate center CC:
# x = 43, y = 70, z = 40
# Coordinates periphery CC:
# x = 36, y = 70, z = 43

sliceCCcenter = (43,70,40)
sliceCCperiphery = (36,70,43)

faArrList = pickle.load(open("faArrList.p", "rb"))


centerList = []
peripheryList = []
for arr in faArrList:
    centerList.append(arr[sliceCCcenter])
    peripheryList.append(arr[sliceCCperiphery])

print("done!")
