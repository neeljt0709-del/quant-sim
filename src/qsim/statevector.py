import numpy as np  

# These are quantum gates, applied to specific qubits, represented as matrices
# The brilliance is that if observed once, no distinguishment can be made 
# When another gate is applied, distinguihsments are made

# The hadamard gate takes a definite qubit and converts it into an equal superposition
h_gate = (1/np.sqrt(2)) * np.array([[1,1],[1,-1]], dtype = complex)

# The x gate is the quantum "NOT": flips 0 and 1
# Rotates halfway about the x-axis in the Bloch Sphere
x_gate = np.array([[0,1],[1,0]], dtype = complex)

# The z gate leaves positions alone, but flips the phase
# This helps reveal interference and shows phase changes
# Rotates halfway about the z-axis in the Bloch Sphere
z_gate = np.array([[1,0],[0,-1]], dtype = complex)

# The y gate flips the one and zero and applies a phase change
# Rotates halfway about the y-axis in the Bloch Sphere
y_gate = np.array([[0, -1j],[1j, 0]], dtype = complex)

# Resets a qubit to 0
reset_gate = np.array([0,0], [0,0], dtype = complex)

# Rotational gates: rotate around specific axis, given an inputted theta
def rz(theta):
    return np.array([[np.exp(-1j*theta/2), 0], [0, np.exp(1j*theta/2)]])

def rx(theta):
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array([[c, -1j * s],[-1j * s, c]])

def ry(theta):
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array([[c, -s],[s, c]])

# The statevector represents all possibilites of the system
# An important distinction to note is that all qubit combinations are represented in one vector
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

    # Allows for angle-specific rotation around axes
    # Can be expanded later to include more axes
    def rotate(self, theta, axis, index):
        mem = {'x': rx, 'y': ry, 'z': rz}
        self.apply(mem[axis](theta), index)
    
    # Convenient functions for x,y,z axes
    def rotate_z(self, theta, index):
        self.rotate(theta, 'z', index)

    def rotate_x(self, theta, index):
        self.rotate(theta, 'x', index)

    def rotate_y(self, theta, index):
        self.rotate(theta, 'y', index)


    # Convenient functions to apply these specific gates
    
    def h(self, index):
        self.apply(h_gate, index)

    def x(self, index):
        self.apply(x_gate, index)

    def y(self, index):
        self.apply(y_gate, index)

    def z(self, index):
        self.apply(z_gate, index)

    def reset(self, index):
        self.apply(reset_gate, index)

    # This function is a way of measuring a specific qubit, gain using tensor logic
    # It is important to note we use numpy's random generator, with seed taken from the Mac OS System
    def measure(self, index):
        tensor = self.data.reshape([2] * self.num_qubits)
        tensor = np.moveaxis(tensor, index, 0)# Move target qubit to front
        prob_0 = np.sum(np.abs(tensor[0])**2)
        prob_1 = np.sum(np.abs(tensor[1])**2)
        # Randomly generates the outcomes given the weighting of the probabilities 
        outcome = np.random.choice([0,1], p = [prob_0, prob_1])
        # Zeroes out everything inconsistent with the outcome
        if outcome == 0:
            tensor[1] = 0
        else:
            tensor[0] = 0
        # Now we must renormalize so that the probabilities sum to 1
        norm = np.sqrt(np.sum(np.abs(tensor)**2))
        tensor = tensor/norm        
        tensor = np.moveaxis(tensor, 0, index)
        self.data = tensor.reshape(2 ** self.num_qubits)
        return outcome

    # The CNOT gate flips everything in a target qubit where the control qubit is |1>
    # This helps represent quantum entanglement and the Bell State
    def cnot(self, control, target):
        tensor = self.data.reshape([2]*self.num_qubits)
        tensor = np.moveaxis(tensor, [control, target], [0,1])
        tensor[1,0], tensor[1,1] = tensor[1,1], np.copy(tensor[1,0])
        tensor = np.moveaxis(tensor, [0,1], [control, target])
        self.data = tensor.reshape(2 ** self.num_qubits)

# This is how we represent circuits
# Ordered sequences of applying gates and measuring qubits
class Circuit:
    memory = {'h': Statevector.h, 'x': Statevector.x, 'y': Statevector.y, 'z': Statevector.z, 'cnot': Statevector.cnot}
    def __init__(self, n):
        self.num_qubits = n
        self.gates = [] # This is static; it keeps track of the gates and their associated qubits
    def add_gate(self, gate, qubits): # qubits should be a list
        self.gates.append([gate, qubits])
    def implement(self, sv): # Implements the circuit on a given statevector
        for gate in self.gates:
            func = self.memory[gate[0]]
            args = gate[1]
            for i in args:
                assert i < sv.num_qubits - 1
            func(sv, *args)