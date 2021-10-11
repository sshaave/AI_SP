import numpy as np

#profilesSteel

# print(profilesSteel.listIPE[0].name)
l = 5.0
q = 2.0

np.random.seed(42)
sizeTraining = 10000
sizeTest = 2000
l_min = 1.5
l_max = 12
q_min = 0.7
q_max = 18
x_matrix = np.zeros((sizeTraining, 3))
x_matrix_test = np.zeros((sizeTest, 3))

for i in range(sizeTraining):
    x_matrix[i][0] = np.random.uniform(l_min, l_max)
    x_matrix[i][1] = np.random.uniform(q_min, q_max)
    x_matrix[i][2] = np.random.uniform(q_min, q_max)

for j in range(sizeTest):
    x_matrix_test[j][0] = np.random.uniform(l_min, l_max)
    x_matrix_test[j][1] = np.random.uniform(q_min, q_max)
    x_matrix_test[j][2] = np.random.uniform(q_min, q_max)

print(x_matrix)

def writeTestInstances(x_matrix_test):
    with open('testInstances.txt', 'w') as i1:
        for i in range(sizeTest):
            i1.write("makeTestSet({:.2f}".format(x_matrix_test[i][0]) + ", " + "{:.2f}".format(x_matrix_test[i][1]) + ", " + "{:.2f}".format(x_matrix_test[i][2]) + ", list)\n")

def writeTrainingInstances(x_matrix):
    with open('instances.txt', 'w') as i1:
        for i in range(sizeTraining):
            i1.write("makeTrainingSet({:.2f}".format(x_matrix[i][0]) + ", " + "{:.2f}".format(x_matrix[i][1]) + ", " + "{:.2f}".format(x_matrix[i][2]) + ", list)\n")

def writeTrainingSet(span, q_load, bestProfile):
    with open("trainingData.txt", 'a') as t1:
        t1.write("l=" + str(span)+ " " + "q=" + str(q_load)+ " " + "profil=" + str(bestProfile.name)+ "\n")
        t1.close()


def readTrainingData():
    global l_trainingData
    X = np.array([l_trainingData, 3])

    dataFile = open('trainingData.txt', 'r')
    Lines = dataFile.readlines()

    count = 0
    for line in Lines:
        count += 1
        print(line)

writeTrainingInstances(x_matrix)
writeTestInstances(x_matrix_test)