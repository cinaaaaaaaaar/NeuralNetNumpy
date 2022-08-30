import numpy as np
from random import random
from Model import MLP

if __name__ == "__main__":
    model = MLP([2, 5, 1])
    inputs = np.array([[random() / 2 for _ in range(2)] for _ in range(1000)])
    targets = np.array([[i[0] * i[1]] for i in inputs])
    model.train(inputs, targets, 50, 10)

    sample = np.array([0.8, 0.4])
    output = model.forward_propagate(sample)
    print(f"Sample: {sample}\nOutput: {round(output[0], 2)}")
