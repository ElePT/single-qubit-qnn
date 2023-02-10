# This code is a modification from the original Qiskit source code.
#
# (C) Copyright IBM 2020, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from qiskit_machine_learning.neural_networks import NeuralNetwork
from qiskit_machine_learning.utils.loss_functions import Loss
from typing import Optional, Union, Callable
import numpy as np

try:
    from sparse import SparseArray
except ImportError:

    class SparseArray:  # type: ignore
        """Empty SparseArray class
        Replacement if sparse.SparseArray is not present.
        """

        pass

class ObjectiveFunction:
    """An abstract objective function. Provides methods for computing objective value and
    gradients for forward and backward passes."""

    # pylint: disable=invalid-name
    def __init__(
            self, X: np.ndarray, y: np.ndarray, neural_network1: NeuralNetwork, neural_network2: NeuralNetwork,
            loss: Loss, callback_fn: Callable
    ) -> None:
        """
        Args:
            X: The input data.
            y: The target values.
            neural_network: An instance of an quantum neural network to be used by this
                objective function.
            loss: A target loss function to be used in training.
        """
        super().__init__()
        self._X = X
        self._y = y
        self._neural_network1 = neural_network1
        self._neural_network2 = neural_network2
        self._loss = loss
        self._last_forward_weights: Optional[np.ndarray] = None
        self._last_forward: Optional[Union[np.ndarray, SparseArray]] = None
        self.callback = callback_fn

    def objective(self, weights: np.ndarray) -> float:
        """Computes the value of this objective function given weights.

        Args:
            weights: an array of weights to be used in the objective function.

        Returns:
            Value of the function.
        """
        raise NotImplementedError

    def gradient(self, weights: np.ndarray) -> np.ndarray:
        """Computes gradients of this objective function given weights.

        Args:
            weights: an array of weights to be used in the objective function.

        Returns:
            Gradients of the function.
        """
        raise NotImplementedError

    def _neural_network_forward(self, weights: np.ndarray) -> Union[np.ndarray, SparseArray]:
        """
        Computes and caches the results of the forward pass. Cached values may be re-used in
        gradient computation.

        Args:
            weights: an array of weights to be used in the forward pass.

        Returns:
            The result of the neural network.
        """
        # if we get the same weights, we don't compute the forward pass again.
        if self._last_forward_weights is None or (
                not np.all(np.isclose(weights, self._last_forward_weights))
        ):
            fwd = []
            for xi in self._X:
                if xi[-1] == 1:
                    # compute forward and cache the results for re-use in backward
                    fwd.append(self._neural_network1.forward(xi[:-1], weights))
                elif xi[-1] == -1:
                    fwd.append(self._neural_network2.forward(xi[:-1], weights))
                else:
                    raise NotImplementedError
                # a copy avoids keeping a reference to the same array, so we are sure we have
                # different arrays on the next iteration.
            self._last_forward = np.asarray(fwd).reshape(-1, 1)
            self._last_forward_weights = np.copy(weights)
        return self._last_forward

class BinaryObjectiveFunction(ObjectiveFunction):
    """An objective function for binary representation of the output,
    e.g. classes of ``-1`` and ``+1``."""

    def objective(self, weights: np.ndarray) -> float:
        # predict is of shape (N, 1), where N is a number of samples
        predict = self._neural_network_forward(weights)
        target = np.array(self._y).reshape(predict.shape)
        # float(...) is for mypy compliance
        return float(np.sum(self._loss(predict, target)))

    def gradient(self, weights: np.ndarray) -> np.ndarray:
        # check that we have supported output shape
        num_outputs = self._neural_network1.output_shape[0]
        if num_outputs != 1:
            raise ValueError(f"Number of outputs is expected to be 1, got {num_outputs}")

        # output must be of shape (N, 1), where N is a number of samples
        output = self._neural_network_forward(weights)
        #         output = (output + np.ones(output.shape)) * 0.5 # map to 0-1
        weight_grad = []
        for xi in self._X:
            if xi[-1] == 1:
                # weight grad is of shape (N, 1, num_weights)
                _, wg = self._neural_network1.backward(xi[:-1], weights)
            elif xi[-1] == -1:
                # weight grad is of shape (N, 1, num_weights)
                _, wg = self._neural_network2.backward(xi[:-1], weights)
            else:
                raise NotImplementedError
            weight_grad.append(wg[0])
        weight_grad = np.asarray(weight_grad)
        # we reshape _y since the output has the shape (N, 1) and _y has (N,)
        # loss_gradient is of shape (N, 1)
        loss_gradient = self._loss.gradient(output, self._y.reshape(-1, 1))
        loss = self._loss.evaluate(output, self._y.reshape(-1, 1))
        self.callback(np.sum(loss) / len(loss), weights)
        # for the output we compute a dot product(matmul) of loss gradient for this output
        # and weights for this output.
        grad = loss_gradient[:, 0] @ weight_grad[:, 0, :]
        # we keep the shape of (1, num_weights)
        grad = grad.reshape(1, -1)

        return grad