from numpy.typing import NDArray
import numpy as np

from network.exceptions import ShapeException
from network.layer import Layer


class NeuralNetwork:
    def __init__(
            self,
            layers: list[Layer],
    ):
        self.layers: list[Layer] = layers
        self.input_layer: NDArray[np.float32] | None = None

    def set_input(self, input_layer: NDArray):
        if input_layer.shape != self.layers[0].input_shape:
            raise ShapeException('Input array size and first layer input shape must be same')

        self.input_layer = input_layer

    def forward_propagation(self) -> NDArray:
        current_array = self.input_layer

        for layer in self.layers:
            current_array = layer.front_propagation(current_array)

        return current_array

    def back_propagation(self):
        ...