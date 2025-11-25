import numpy as np

from network.layer import FullyConnectedLayer

layer = FullyConnectedLayer(
    (2, 2),
    (2, 2),
    lambda x: x
)

layer.weights = np.array(
    [
        [
            [
                [1, 0],
                [0, 2]
            ],
            [
                [0, 2],
                [100, 0]
            ]
        ],
        [
            [
                [0, 0],
                [3, 0]
            ],
            [
                [0, 0],
                [0, 4]
            ]
        ]
    ],
    dtype=np.float32
)

output = layer.front_propagation(
    np.array(
        [
            [1, 2],
            [2, 1]
        ],
        dtype=np.float32
    )
)

print(output)