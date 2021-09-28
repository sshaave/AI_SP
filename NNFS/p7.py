import numpy as np
import math

#softmax_output = [0.7, 0.1, 0.2]
#target_output = [1, 0, 0]

#loss = -

a = np.array([11, 22, 33, 44])
b = np.array(["A", "B", "C", "D"])
print(a)
print(b)

for index, (so, sd) in \
    enumerate(zip(a, b)):
    print("Index: ", index)
    print("So: ", so)
    print("Sd: ", sd)