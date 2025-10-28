from abc import ABC, abstractmethod

from numpy.typing import NDArray

from network.types import NDShape


class Layer(ABC):
    def __init__(self, input_shape: NDShape, layer_shape: NDShape,):
        self.input_shape: NDShape = input_shape
        self.layer_shape: NDShape = layer_shape

    @abstractmethod
    def front_propagation(self, input_layer: NDArray) -> NDArray: ...

    # @abstractmethod
    # def back_propagation(self, input_layer): ...