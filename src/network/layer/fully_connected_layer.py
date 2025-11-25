from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from .layer import Layer
from network.types import NDShape
from ..exceptions import ShapeException


class FullyConnectedLayer(Layer):

    def __init__(
            self,
            input_shape: NDShape,
            output_shape: NDShape,
            activation_func: Callable[[float], float],
            weights: NDArray[np.float32] = None,
            bias: NDArray[np.float32] = None
    ):
        super().__init__(input_shape, output_shape)

        self.activation_func: Callable[[float], float] = activation_func

        self.weights: NDArray[np.float32] = weights
        self.bias: NDArray[np.float32] = bias

        if self.weights is None:
            # 출력 레이어의 한 뉴런 단위로 그 하위 차원에 입력 레이어에 대한 가중치가 지정됨
            self.weights = np.zeros(output_shape + input_shape, dtype=np.float32)
        if self.bias is None:
            self.bias = np.zeros(output_shape, dtype=np.float32)

    def front_propagation(self, input_layer: NDArray) -> NDArray:
        if input_layer.shape != self.input_shape:
            raise ShapeException()

        out_ndim = len(self.weights.shape) - len(input_layer.shape)
        in_ndim = len(input_layer.shape)

        in_subs = ''.join(chr(ord('a') + i) for i in range(in_ndim))
        out_subs = ''.join(chr(ord('a') + i + in_ndim) for i in range(out_ndim))
        subs = f"{out_subs}{in_subs},{in_subs}->{out_subs}"

        result = np.einsum(subs, self.weights, input_layer)
        result = result + self.bias
        func = np.vectorize(self.activation_func)

        return func(result)