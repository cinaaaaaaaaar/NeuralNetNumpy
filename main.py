import numpy as np
from classes import MLP

if __name__ == "__main__":
    model = MLP([3, 5, 7, 2])
    inputs = np.random.rand(model.layers[0])
    output = model.forward_propagate(inputs)

    print(f"Inputs: {inputs}\nOutput: {output}")
