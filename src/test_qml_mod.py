import numpy as np
from qiskit import Aer, QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from qiskit.opflow import StateFn, PauliSumOp, AerPauliExpectation, ListOp, Gradient, OperatorStateFn
from qiskit.utils import QuantumInstance
import matplotlib.pyplot as plt
from qiskit_utils import BinaryObjectiveFunction
from data_utils import circle, plot_data, generate_ds

Xdata, ydata = circle(500)
fig, ax = plt.subplots(1, 1, figsize=(4, 4))
plot_data(Xdata, ydata, fig=fig, ax=ax)
plt.show()

# Generate training and test data
num_training = 200
num_test = 2000

X_train, y_train, new_y_train = generate_ds(num_training)
X_test, y_test, new_y_test= generate_ds(num_test)

# set method to calculcate expected values
expval = AerPauliExpectation()

# define gradient method
gradient = Gradient()

# define quantum instances (statevector and sample based)
qi_sv = QuantumInstance(Aer.get_backend('aer_simulator_statevector'))

from qiskit_machine_learning.neural_networks import OpflowQNN
# construct parametrized circuit
inputs = ParameterVector('input', length=4) # 1 extra for label
# inputs = ParameterVector('input', length=3)
weights = ParameterVector('weight', length=9)

# 1qubit classifier
qc1 = QuantumCircuit(1)
qc1.u(inputs[0],inputs[1],inputs[2], 0)
qc1.u(weights[0],weights[1],weights[2], 0)
qc1.u(inputs[0],inputs[1],inputs[2], 0)
qc1.u(weights[3],weights[4],weights[5], 0)
qc1.u(inputs[0],inputs[1],inputs[2], 0)
qc1.u(weights[6],weights[7],weights[8], 0)
# qc1.rx(inputs[3],0)
qc_sfn1 = StateFn(qc1)

H1 = StateFn(PauliSumOp.from_list([('Z', 1.0)]))
H2 = StateFn(PauliSumOp.from_list([('Z', -1.0)]))

op1 = ~H1 @ (qc_sfn1)
op2 = ~H2 @ (qc_sfn1)
print(op1)

# construct OpflowQNN with the operator, the input parameters, the weight parameters,
# the expected value, gradient, and quantum instance.
qnn1 = OpflowQNN(op1, inputs, weights, expval, gradient, qi_sv, input_gradients=True)
qnn2 = OpflowQNN(op2, inputs, weights, expval, gradient, qi_sv, input_gradients=True)

from qiskit.algorithms.optimizers import ADAM, L_BFGS_B
from qiskit_machine_learning.utils.loss_functions import L2Loss

losses = []
def callback_fn(avg_loss, weights):
    print("weights: ", weights)
    print("loss: ", avg_loss)
    losses.append(avg_loss)

function = BinaryObjectiveFunction(X_train, new_y_train, qnn1, qnn2, L2Loss(), callback_fn)

losses = []
# optimizer = ADAM(maxiter=30, lr=0.8)
optimizer = L_BFGS_B(maxiter=2)
# fit_result = optimizer.minimize(
fit_result = optimizer.optimize(
    num_vars=9,
    objective_function=function.objective,
    # initial_point=algorithm_globals.random.random(qnn1.num_weights),
    initial_point = [0.3699248,  0.9269141,  0.9097975,  0.97121462, 0.70890935, 0.64617032,
    0.72212077, 0.60516505, 0.15746219],
    gradient_function=function.gradient,
)

print(fit_result)
print("losses: ", losses)
