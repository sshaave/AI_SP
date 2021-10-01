import numpy as np
import pickle
from NNFS.SP.AI_SP import Layer_Dense, Activation_ReLU, Activation_Softmax_Loss_CategoricalCrossentropy

def readLayers():
    global layer1
    global layer2
    with open("saved_Layer1.pk", 'rb') as fti1:
        layer1 = pickle.load(fti1)
    with open("saved_Layer2.pk", 'rb') as fti2:
        layer2 = pickle.load(fti2)
        return layer1, layer2


layer1 = Layer_Dense(2, 64)
layer2 = Layer_Dense(64, 124)
readLayers()

activation1 = Activation_ReLU()
loss_activation1 = Activation_Softmax_Loss_CategoricalCrossentropy()

X_test4 = np.zeros([1, 2])
X_test4[0] = [7, 5.0]
X_test4 = np.append(X_test4, [[7, 5.0]], axis=0)
y_test4 = np.zeros(1, dtype='uint8')
y_test4[0] = np.uint(4)
y_test4 = np.append(y_test4, [np.uint8(4)], axis=0)

layer1.forward(X_test4)
activation1.forward(layer1.output)
layer2.forward(activation1.output)
loss = loss_activation1.forward(layer2.output, y_test4)

predictions = np.argmax(loss_activation1.output, axis=1)
accuracy1 = np.mean(predictions == y_test4)
print(predictions)
print(layer1.weights)

for i in range(64):
    print(layer1.weights[1][i])