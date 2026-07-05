import numpy as np  

# These are quantum gates, applied to specific qubits, represented as matrices
# The brilliance is that if observed once, no distinguishment can be made 
# When another gate is applied, distinguihsments are made

# The hadamard gate takes a definite qubit and converts it into an equal superposition
h_gate = (1/np.sqrt(2)) * np.array([[1,1],[1,-1]], dtype = complex)

# The x gate is the quantum "NOT": flips 0 and 1
x_gate = np.array([[0,1],[1,0]], dtype = complex)

# The z gate leaves positions alone, but flips the phase
# This helps reveal interference and shows phase changes
z_gate = np.array([[1,0],[0,-1]], dtype = complex)

# The y gate flips the one and zero and applies a phase change
y_gate = np.array([[0, -1j],[1j, 0]], dtype=complex)

# The statevector represents all possibilites of the system
# An important distinction to note is that all qubits are represented in one vector
class Statevector:
    def __init__(self, n):
        self.num_qubits = n
        dimensions = 2 ** n # Our vector is length 2^n, so that it can represent all possible values
        self.data = np.zeros(dimensions, dtype = complex) # Initializing amplitudes of each value
        self.data[0] = 1.0 # Start in the |000...0> state

    def __repr__(self):
        return f"Statevector({self.num_qubits} qubits): {self.data}"
    
    def probabilities(self):
        # Each state has probability abs(amplitude)^2
        # Returns a numpy array of the probabilities of all states
        return np.abs(self.data) ** 2

    # We must use tensor logic to apply a gate to one specific qubit
    # By converting the statevector into a tensor, we can matrix multiply to a specific axis
    # Each axis represents a specific qubit, so no we can isolate qubits without disturbing the others

    def apply(self, matrix, index):
        # This tensor has n axes, each size 2
        tensor = self.data.reshape([2]*self.num_qubits)

        tensor = np.tensordot(matrix, tensor, axes = ([1], [index])) # Applies matrix to target axis

        tensor = np.moveaxis(tensor, 0, index) # Moves axis back to appropriate index

        self.data = tensor.reshape(2**self.num_qubits) # Reflatten tensor back to a statevector
    
    def h(self, index):
        self.apply(h_gate, index)

    def x(self, index):
        self.apply(x_gate, index)

    def y(self, index):
        self.apply(y_gate, index)

    def z(self, index):
        self.apply(z_gate, index)
   
    