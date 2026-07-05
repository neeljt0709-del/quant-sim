# pytest runs all functions in this file beginning with 'test_'
import numpy as np
from src.qsim.statevector import Statevector

# Manual tests to check all gates
# np.allclose checks two arrays to within a small floating-point tolerance

def test_initial(): # Checks that the initial state is definite at |000...0>
    sv = Statevector(2)
    expected = np.array([1,0,0,0], dtype = complex)
    assert np.allclose(sv.data, expected)

def test_hadamard(): # Checking hadamard gate creates equal superposition
    sv = Statevector(1)
    sv.h(0)
    expected = np.array([0.5, 0.5])
    assert np.allclose(sv.probabilities(), expected)

def test_hadamard_twice(): # Tests that using hadamard twice gives the identity
    sv = Statevector(1)
    sv.h(0)
    sv.h(0)
    expected = np.array([1, 0], dtype=complex)
    assert np.allclose(sv.data, expected)


def test_h_one(): # Tests hadamard on one qubit in two qubit system
    sv = Statevector(2)
    sv.h(0)
    expected = np.array([1/np.sqrt(2), 0, 1/np.sqrt(2), 0], dtype=complex)
    assert np.allclose(sv.data, expected)


def test_x(): # Tests x gate works as 'NOT'
    sv = Statevector(1)
    sv.x(0)
    expected = np.array([0, 1], dtype=complex)
    assert np.allclose(sv.data, expected)


def test_z(): # Tests that the z gate leaves positions alone
    sv = Statevector(1)
    sv.z(0)
    expected = np.array([1, 0], dtype=complex)
    assert np.allclose(sv.data, expected)


def test_z(): # Tests that the z gate flips phase
    sv = Statevector(1)
    sv.x(0)
    sv.z(0)
    expected = np.array([0, -1], dtype=complex)
    assert np.allclose(sv.data, expected)


def test_y(): # Tests that y gate flips the zero to imaginary phase one
    sv = Statevector(1)
    sv.y(0)
    expected = np.array([0, 1j], dtype=complex)
    assert np.allclose(sv.data, expected)