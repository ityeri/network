import random
from typing import Callable

from math import ceil
import pygame
import numpy as np
import time
from tensorflow.keras.datasets import mnist

from numpy.typing import NDArray

from network import NDShape
from network.layer import FullyConnectedLayer
from network.neural_network import NeuralNetwork


input_shape = (28, 28)
relu = lambda x: x if 0 < x else 0

def generate_random_network(
        input_shape: tuple[int, int],
        min_weight: float, max_weight: float,
        min_bias: float, max_bias: float,
        activation_func: Callable[[float], float]
) -> NeuralNetwork:

    layer1_initial_weights = np.random.uniform(min_weight, max_weight, (16,) + input_shape).astype(np.float32)
    layer1_initial_bias = np.random.uniform(min_bias, max_bias, (16,)).astype(np.float32)
    layer1 = FullyConnectedLayer(
        input_shape, (16,), activation_func,
        layer1_initial_weights, layer1_initial_bias
    )

    layer2_initial_weights = np.random.uniform(min_weight, max_weight, (16, 16)).astype(np.float32)
    layer2_initial_bias = np.random.uniform(min_bias, max_bias, (16,)).astype(np.float32)
    layer2 = FullyConnectedLayer(
        (16,), (16,), activation_func,
        layer2_initial_weights, layer2_initial_bias
    )

    output_layer_initial_weights = np.random.uniform(min_weight, max_weight, (10, 16)).astype(np.float32)
    output_layer_initial_bias = np.random.uniform(min_bias, max_bias, (10,)).astype(np.float32)
    output_layer = FullyConnectedLayer(
        (16,), (10,), activation_func,
        output_layer_initial_weights, output_layer_initial_bias
    )

    return NeuralNetwork([layer1, layer2, output_layer])

def generate_random_bool(shape: NDShape) -> NDArray[np.bool_]:
    return np.random.rand(*shape) < 0.5

def mix_fc_layer(
        layer1: FullyConnectedLayer,
        layer2: FullyConnectedLayer,
        activation_func: Callable[[float], float]
) -> FullyConnectedLayer:
    mask = np.random.rand(*(layer1.output_shape + layer1.input_shape)) < 0.5
    weight = np.where(mask, layer1.weights, layer2.weights)

    mask = np.random.rand(*layer1.output_shape) < 0.5
    bias = np.where(mask, layer1.bias, layer2.bias)

    return FullyConnectedLayer(
        layer1.input_shape,
        layer1.output_shape,
        activation_func,
        weight,
        bias
    )


network_nums = 1000
candidate_network_nums = 50
mutation_rate = 0.001
mutation_range = 0.0001

networks = [
    generate_random_network(
        (28, 28),
        -1.0, 1.0, -1.0, 1.0, relu
    ) for _ in range(network_nums)
]

pygame.init()

on = True
screen = pygame.display.set_mode((700, 500), pygame.RESIZABLE)
screen_width, screen_height = screen.get_size()
fps = 60
clk = pygame.time.Clock()
dt = int(1000 / fps)

space_ratio = 0.7

input_array = np.zeros(input_shape, np.float32)

input_surface = pygame.Surface((screen_width * space_ratio, screen_height))
output_surface = pygame.Surface((screen_width * (1 - space_ratio), screen_height))

while on:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            on = False

        elif event.type == pygame.VIDEORESIZE:
            screen_width, screen_height = screen.get_size()
            input_surface = pygame.Surface((screen_width * space_ratio, screen_height))
            output_surface = pygame.Surface((screen_width * (1 - space_ratio), screen_height))

        elif event.type == pygame.KEYDOWN:
            try:
                answer = int(event.unicode)
                answer_array = np.zeros((10,), np.bool_)
                answer_array[answer] = True

                for neural_network in networks:
                    neural_network.set_input(input_array)

                outputs = [neural_network.forward_propagation() for neural_network in networks]
                normalized_outputs = list()

                for output in outputs:
                    max_value = output.max()
                    if max_value == 0:
                        max_value = 1

                    normalized_outputs.append(output / max_value)

                costs = [(i, np.sum((output - answer_array) ** 2)) for i, output in enumerate(normalized_outputs)]
                costs.sort(key=lambda x: x[1])

                candidates = [networks[index] for index, _, in costs[0:candidate_network_nums]]
                new_networks: list[NeuralNetwork] = list()

                for i in range(network_nums):
                    neural_network1: NeuralNetwork = random.choice(candidates)
                    neural_network2: NeuralNetwork = random.choice(candidates)

                    new_networks.append(
                        NeuralNetwork(
                            [
                                mix_fc_layer(neural_network1.layers[0], neural_network2.layers[0], relu),
                                mix_fc_layer(neural_network1.layers[1], neural_network2.layers[1], relu),
                                mix_fc_layer(neural_network1.layers[2], neural_network2.layers[2], relu)
                            ]
                        )
                    )

                networks = new_networks
                input_array = np.zeros(input_shape, np.float32)

                # for neural_network in new_networks:
                #     for fc_layer in neural_network.layers:
                #         fc_layer: FullyConnectedLayer
                #
                #
                #         # fc_layer.weights +=

            except ValueError:
                pass

    mouse = pygame.mouse.get_pressed()


    input_surface.fill((25, 25, 25))

    if input_surface.get_height() < input_surface.get_width():
        size = input_surface.get_height()
        drawing_space = pygame.Rect((input_surface.get_width() - size) / 2, 0, size, size)
    else:
        size = input_surface.get_width()
        drawing_space = pygame.Rect(0, (input_surface.get_height() - size) / 2, size, size)

    input_surface.fill((50, 50, 50), drawing_space)

    if mouse[0]:
        mouse_x, mouse_y = pygame.mouse.get_pos()
        x = int((mouse_x - drawing_space.left) / drawing_space.width * input_shape[1])
        y = int((mouse_y - drawing_space.top) / drawing_space.height * input_shape[0])

        if drawing_space.collidepoint(mouse_x, mouse_y):
            input_array[y, x] = 1.0

    pix_size = drawing_space.width / 28

    for y in range(input_shape[0]):
        for x in range(input_shape[1]):
            value = input_array[y, x]
            pygame.draw.rect(
                input_surface,
                (255 * value, 0, 255 * value),
                (
                    drawing_space.x + pix_size * x, drawing_space.y + pix_size * y,
                    ceil(pix_size), ceil(pix_size)
                )
            )


    output_surface.fill((33, 33, 33))

    for neural_network in networks:
        neural_network.set_input(input_array)

    start_time = time.perf_counter()
    outputs = [neural_network.forward_propagation() for neural_network in networks]
    end_time = time.perf_counter()

    elapsed_time = end_time - start_time
    print(int(elapsed_time * 1_000_000), 'us')

    pix_width = output_surface.get_width() / len(outputs)
    pix_height = output_surface.get_height() / 10

    for i, output in enumerate(outputs):
        max_value = output.max()
        if max_value == 0:
            max_value = 1

        for j in range(10):
            value = output[j] / max_value
            color_value = int(255 * value)
            if 255 < color_value:
                color_value = 255

            pygame.draw.rect(
                output_surface,
                (0, color_value, color_value),
                (pix_width * i, pix_height * j, ceil(pix_width), ceil(pix_height))
            )


    screen.blit(input_surface, (0, 0))
    screen.blit(output_surface, (screen_width * space_ratio, 0))

    pygame.display.flip()

    dt = clk.tick(fps)