# Quantum Simulator
A from-scratch quantum circuit simulator built in Python. This project implements introductory quantum information theory via the construction and implementation of tensors, state vector values, quantum gates, etc. Tensor logic is used to avoid building full unitary matrices. Coupled with a working Circuit class, this simulator can now be used to run basic probability experiments and trials, simulating and recording results of multiple shots.

Features:
- Statevector representation: n qubits are represented as a complex vector length 2^n. Internally, it is reshaped       as an n-axis tensor so that gates may be applied to individual qubits in isolation.
- Single qubit gates: Hadamard(H), Pauli Gates(X, Y, Z) and a reset operation
- Parameterized rotation gates: RX(θ), RY(θ), RZ(θ) are the building blocks for variational algorithms (VQE, QAOA),     rotional gates to be applied for a non-trivial theta on each axis
- Entangling Gate: CNOT
- Measurement: Utilizes numpy's random function to determine a definite state and correctly renormalizes and            collapses all other states
- Circuit abstraction: A working Circuit class to assemble sequences of gates and apply to individual statevectors,     retains results in order to keep track of shot data.

How applying a gate works:
  
Rather than build the 2^n x 2^n unitary matrix, the flat statevector is reshaped into an n-axis tensor, with each axis representing a qubit. Applying a gate to a qubit therefore becomes a tensordot contraction on that qubit's axis, leaving all other axes untouched. This is mathematically equivalent to applying 𝐼 ⊗ ⋯ ⊗ 𝑈 ⊗ ⋯ ⊗ 𝐼 to the full state, but avoids ever materializing that large matrix.

Circuit class:

Simulates an assembled circuit as an order of gates. When a circuit is implemented on a specific statevector, the circuit returns a list of the measured qubits, as well as a copy of the statevector as it was affected by the circuit, but does not modify the circuit itself. The circuit also keeps track of how many shots it was used for. This allows for multiple shots on the same statevector to run probability trials and record empirical observations.

API References: 

Statevector(n): Initializes an n-qubit register in the ∣0⟩^n state

h(index): Apply Hadamard to qubit index

x(index) / y(index) / z(index): Apply Pauli Gates

rotate_x(theta, index) / rotate_y(theta, index) / rotate_z(theta, index): Apply parameterized rotations

cnot(control, target): Apply controlled-NOT

reset(index): Reset a qubit to |0>

measure(index): Collapse and measure a qubit, return either 0 or 1

probabilities(): Returns |amplitude|^2 for every basis state

copy(): Returns an independent deep-copy of the statevector

Circuit(n): Creates a circuit for a statevector with n qubits

add_gate(gate, qubits, theta=0): qubits, theta=0)	Append a gate to the circuit. gate is one of 'h', 'x', 'y', 'z', 'cnot', 'reset', 'measure', 'rx', 'ry', 'rz'. qubits is a list of target qubit indices; theta is required for rotation gates.

implement(sv): 	Runs the full gate sequence on a copy of sv, returns (results, final_statevector)

  
