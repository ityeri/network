from abc import ABC, abstractmethod

from numpy.typing import NDArray

from network.types import NDShape


class Layer(ABC):
    def __init__(self, input_shape: NDShape, output_shape: NDShape):
        self.input_shape: NDShape = input_shape
        self.output_shape: NDShape = output_shape
        self.output: NDArray | None = None

    @abstractmethod
    def front_propagation(self, input_layer: NDArray) -> NDArray: ...

    # @abstractmethod
    # def back_propagation(self, input_layer): ...