import numpy as np

def sigmoid(x, der = False):
    if der == True:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))

x = np.array([[1, 0, 1],
              [1, 0, 1],
              [0, 1, 0],
              [0, 1, 0]])

y = np.array([[0, 0, 1, 1]]).T

np.random.seed(1)
syn0 = 2 * np.random.random((3, 1)) - 1
l1 = []

"for iter in range(100000)"
for iter in range(10000):
    l0 = x
    l1 = sigmoid(np.dot(l0, syn0))
    l1err = y - l1
    l1dt = l1err * sigmoid(l1, True)
    
    syn0 += np.dot(l0.T, l1dt)

print("Выходные данные после тренировки")
print(l1)


newlist = np.array([1, 0, 1])
l1new = sigmoid(np.dot(newlist, syn0))
print("Новые данные")
print(l1new)