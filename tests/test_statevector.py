# pytest runs all functions in this file beginning with 'test_'
import numpy as np
from src.qsim.statevector import Statevector

# Manual tests to check all gates
# np.allclose checks two arrays to within a small floating-point tolerance

def test_initial(): # Checks that the initial state is definite at |000...0>
    test = Statevector(2)
    expected = np.array([1,0,0,0], dtype = complex)
    assert np.allclose(test.data, expected)

def test_hadamard(): # Checking hadamard gate creates equal superposition
    test = Statevector(1)
    test.h(0)
    expected = np.array([0.5, 0.5])
    assert np.allclose(test.probabilities(), expected)

def test_hadamard_twice(): # Tests that using hadamard twice gives the identity
    test = Statevector(1)
    test.h(0)
    test.h(0)
    expected = np.array([1, 0], dtype=complex)
    assert np.allclose(test.data, expected)


def test_h_one(): # Tests hadamard on one qubit in two qubit system
    test = Statevector(2)
    test.h(0)
    expected = np.array([1/np.sqrt(2), 0, 1/np.sqrt(2), 0], dtype=complex)
    assert np.allclose(test.data, expected)


def test_x(): # Tests x gate works as 'NOT'
    test = Statevector(1)
    test.x(0)
    expected = np.array([0, 1], dtype=complex)
    assert np.allclose(test.data, expected)


def test_z(): # Tests that the z gate leaves positions alone
    test = Statevector(1)
    test.z(0)
    expected = np.array([1, 0], dtype=complex)
    assert np.allclose(test.data, expected)


def test_z_phase(): # Tests that the z gate flips phase
    test = Statevector(1)
    test.x(0)
    test.z(0)
    expected = np.array([0, -1], dtype=complex)
    assert np.allclose(test.data, expected)


def test_y(): # Tests that y gate flips the zero to imaginary phase one
    test = Statevector(1)
    test.y(0)
    expected = np.array([0, 1j], dtype=complex)
    assert np.allclose(test.data, expected)


# Tests for measurement
    
def test_collapse(): # Tests that the resulting data collapses
    test = Statevector(1)
    test.h(0)
    outcome = test.measure(0)
    expected = np.zeros(2)
    expected[outcome] = 1
    assert np.allclose(test.data, expected)

def test_variance(): # Tests for variance in measurement results
    lst = []
    for i in range(1000):
        test = Statevector(1)
        test.h(0)
        lst.append(test.measure(0))
    assert  430 < sum(lst) < 570 # Approximately +/- 4 standard deviations

def test_definite(): # Checks measure for a definite qubit
    test = Statevector(1)
    test.x(0)
    outcome = test.measure(0)
    expected = np.array([0,1])
    assert np.allclose(expected, test.data)
    assert outcome == 1

def test_cnot(): # Checking cnot gate
    test = Statevector(2)
    test.h(0)
    test.cnot(0,1)
    expected = np.array([1/np.sqrt(2)+0j, 0, 0, 1/np.sqrt(2)+0j])
    assert np.allclose(expected, test.data)

def test_cnot_inverse(): # Making sure control and target can be whatever axes they want
    test = Statevector(2)
    test.h(1)
    test.cnot(1,0)
    expected = np.array([1/np.sqrt(2)+0j, 0, 0, 1/np.sqrt(2)+0j])
    assert np.allclose(expected, test.data)

def test_cnot_asymmetric_control_zero():
    test = Statevector(2)
    test.x(1)
    initial_state = np.copy(test.data)
    test.cnot(0,1)
    assert np.allclose(test.data, initial_state)

def test_cnot_asymmetric_control_one():
    test = Statevector(2)
    test.x(1)
    test.cnot(1,0)
    expected = np.array([0,0,0,1], dtype = complex)
    assert np.allclose(expected, test.data)

def test_spectator(): # Ensures a qubit not part of cnot is unaffected
    test = Statevector(3)
    test.x(2) 
    test.h(0)
    test.cnot(0,1)
    expected = np.zeros(8, dtype=complex)
    expected[1] = 1/np.sqrt(2) + 0j
    expected[7] = 1/np.sqrt(2) + 0j   
    assert np.allclose(expected, test.data)

def test_cnot_phase_kickback():
    test = Statevector(2)
    test.h(0)
    test.x(1)
    test.h(1)
    test.cnot(0,1)
    # Phase kickback leaves target unchanged, but control changes signs
    expected = np.array([0.5, -0.5, -0.5, 0.5], dtype=complex)
    assert np.allclose(expected, test.data)

# Retrieves (amplitude)^2 for all basis states
def get_probs(sv):
    return np.abs(sv.data) ** 2

