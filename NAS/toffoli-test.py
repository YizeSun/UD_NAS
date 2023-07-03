import pennylane as qml
from pennylane import numpy as np

dev = qml.device("default.qubit", wires=3, shots=1000)

# Toffoli-gate:
# [[1 0 0 0 0 0 0 0]
#  [0 1 0 0 0 0 0 0]
#  [0 0 1 0 0 0 0 0]
#  [0 0 0 1 0 0 0 0]
#  [0 0 0 0 1 0 0 0]
#  [0 0 0 0 0 1 0 0]
#  [0 0 0 0 0 0 0 1]
#  [0 0 0 0 0 0 1 0]]

@qml.qnode(dev)
def circuit(inputs):
    qml.RX(np.pi*inputs[0], wires=[0])
    qml.RX(np.pi*inputs[1], wires=[1])
    qml.RX(np.pi*inputs[2], wires=[2])

    # qml.PauliX(wires=[0])
    # qml.PauliX(wires=[1])
    # qml.PauliX(wires=[2])
    qml.Toffoli(wires=[0, 1, 2])
    # qml.T(wires=[0])
    # qml.Hadamard(wires=[2])
    # qml.T(wires=[2])
    # qml.T(wires=[2])
    # qml.CNOT(wires=[1,2])
    # qml.adjoint(qml.T(wires=2))
    # qml.adjoint(qml.T(wires=2))
    # qml.CNOT(wires=[0,2])
    # return qml.expval(qml.PauliZ(2))
    return [qml.expval(qml.PauliZ(0)@qml.PauliZ(1)@qml.PauliZ(2))]

states = [list(map(int, list(format(i, '03b')))) for i in range(2**3)]

for i in states:
    result = circuit(i)
    print(i, (result+1)/2)
# print(qml.matrix(circuit)())