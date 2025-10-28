from collections.abc import Callable

from network.connect_layer import ConnectLayer
from network.types import NDShape


class NeuralNetwork:
    def __init__(
            self,
            input_layer_size: NDShape,
            layers: list[DimensionalArray],
            bridge_layers: list[ConnectLayer],
            activation_func: Callable[[float], float]
    ):
        self.input_layer_size: NDShape = input_layer_size
        self.layers: list[DimensionalArray] = layers
        self.bridge_layers: list[ConnectLayer] = bridge_layers
        self.activation_func: Callable[[float], float] = activation_func

    def set_input(self, input_layer: DimensionalArray):
        if input_layer.shape != self.layers[0].shape:
            raise ValueError('입력된 레이어와 실제 입력 레이어의 크기가 다릅니다')

        self.input_layer = input_layer

    def forward_propagation(self):
        previous_layer = self.input_layer

        for layer, bridge_layer in zip(self.layers, self.bridge_layers):
            for index in layer.get_all_indexes():
                value = sum(
                    previous_layer[prev_layer_index] * bridge_layer.get_weight(prev_layer_index, index)
                    for prev_layer_index in bridge_layer.get_indexes(index)
                ) + bridge_layer.get_bias(index)

                layer[index] = self.activation_func(value)

            previous_layer = layer

    def back_propagation(self):
        ...