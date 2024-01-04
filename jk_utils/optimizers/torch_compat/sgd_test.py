import unittest

import keras
import numpy as np
import tensorflow as tf
import torch
from absl.testing import parameterized

from jk_utils import testing
from jk_utils.optimizers.torch_compat.sgd import TorchSGD

assert keras.backend.backend() == "tensorflow"


def loss_keras(x):
    x0, x1 = keras.ops.unstack(x)
    return x0**2 + 2 * (x1 + 2) ** 2


def loss_torch(x):
    return x[0] ** 2 + 2 * (x[1] + 2) ** 2


class TorchSGDTest(testing.TestCase, parameterized.TestCase):
    @parameterized.product(
        momentum=(0, 0.9),
        l2_factor=(0, 1e-1),
        # dampening=(0, 0.4),
        nesterov=(False, True),
        cosine_annealing=(False, True),
    )
    def test_sgd(
        self,
        learning_rate: float = 0.1,
        momentum: float = 0.0,
        l2_factor: float = 0.0,
        # dampening: float = 0.0,
        num_steps: int = 5,
        nesterov: bool = False,
        cosine_annealing: bool = False,
    ):
        # if nesterov and (momentum == 0 or dampening != 0):
        if nesterov and momentum == 0:
            return

        init = np.ones((2,), dtype=np.float32)

        # torch implementation
        torch_x = torch.nn.Parameter(torch.tensor(init))
        torch_optimizer = torch.optim.SGD(
            [torch_x],
            lr=learning_rate,
            momentum=momentum,
            weight_decay=l2_factor,
            # dampening=dampening,
            nesterov=nesterov,
        )
        if cosine_annealing:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                torch_optimizer, eta_min=0.0, T_max=num_steps
            )
        torch_trajectory = []
        for _ in range(num_steps):
            torch_optimizer.zero_grad()
            loss = loss_torch(torch_x)
            loss.backward()
            torch_optimizer.step()
            if cosine_annealing:
                scheduler.step()
            torch_trajectory.append(torch_x.detach().numpy().copy())

        # keras implementation
        keras_lr = learning_rate
        if cosine_annealing:
            keras_lr = keras.optimizers.schedules.CosineDecay(keras_lr, num_steps)
        keras_optimizer = TorchSGD(
            keras_lr,
            momentum=momentum,
            l2_factor=l2_factor,
            # dampening=dampening,
            nesterov=nesterov,
        )
        x = tf.Variable(initial_value=init)
        keras_optimizer.build([x])
        keras_trajectory = []

        @tf.function()
        def update():
            with tf.GradientTape() as tape:
                loss = loss_keras(x)
            grads = tape.gradient(loss, [x])
            keras_optimizer.apply(grads)

        for _ in range(num_steps):
            update()
            keras_trajectory.append(x.numpy())

        self.assertAllClose(torch_trajectory, keras_trajectory, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
