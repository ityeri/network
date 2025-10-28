from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from network.layer import Layer
from network.types import NDShape


class FullyConnectedLayer(Layer):

    def __init__(
            self,
            input_shape: NDShape,
            layer_shape: NDShape,
            activation_func: Callable[[float], float]
    ):
        super().__init__(input_shape, layer_shape)

        self.activation_func: Callable[[float], float] = activation_func
        self.weights: NDArray = np.zeros(layer_shape + input_shape, dtype=np.float32)


    def front_propagation(self, input_layer: NDArray) -> NDArray:
        pass
