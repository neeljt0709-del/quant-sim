# pytest runs all functions in this file beginning with 'test_'
import numpy as np
from src.qsim.statevector import Statevector, Circuit
import pytest

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

def test_zero_angle(): # Theta = 0 should not alter the state
    for func in ['rotate_x', 'rotate_y', 'rotate_z']:
        test = Statevector(1)
        original = test.data.copy()
        getattr(test, func)(0.0,0)
        assert np.allclose(test.data, original) # Checks each axis independently

def test_full_rotation():  # Full rotation should return to -original (global phase flip)
    for func in ['rotate_x', 'rotate_y', 'rotate_z']:
        test = Statevector(1)
        original = test.data.copy()
        getattr(test, func)(np.pi * 2, 0)
        assert np.allclose(test.data, -original)

def test_inverses(): # Negative angles are inverses of each other
    test = Statevector(1)
    og = test.data.copy()
    test.rotate_x(np.pi/4,0)
    test.rotate_x(-np.pi/4,0)
    assert np.allclose(test.data, og)

def test_rx_pi_exact_amplitude(): # Rx(pi) on |0> should give exactly [0, -1j]
        test = Statevector(1)
        test.rotate_x(np.pi, 0)
        assert np.allclose(test.data, [0, -1j])

def test_ry_pi_exact_amplitude(): # Ry(pi) on |0> should give exactly [0, 1], fully real
        test = Statevector(1)
        test.rotate_y(np.pi, 0)
        assert np.allclose(test.data, [0, 1])

def test_rz_pi_exact_amplitude(): # Rz(pi) on |0> should give exactly [-1j, 0]
        test = Statevector(1)
        test.rotate_z(np.pi, 0)
        assert np.allclose(test.data, [-1j, 0])

# I am too lazy to write these tests, so they were provided by Claude
class TestApplyStructural:

    def test_normalization_preserved_for_non_rotation_gate(self):  # apply() should preserve normalization for any unitary
        hadamard = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
        test = Statevector(1)
        test.apply(hadamard, 0)
        assert np.isclose(np.sum(np.abs(test.data) ** 2), 1.0)

    def test_unrelated_qubits_untouched_three_qubit_product_state(self):  # Rotating one qubit shouldn't affect the others
        test = Statevector(3)
        test.rotate_x(np.pi, 1)  # flip only the middle qubit
        probs = np.abs(test.data) ** 2
        expected = np.zeros(8)
        expected[2] = 1.0  # |010> -- adjust index if your ordering differs
        assert np.allclose(probs, expected)


class TestIndexing:

    @pytest.mark.parametrize("index,expected_state_index", [
        (0, 8),   # |1000>
        (1, 4),   # |0100>
        (2, 2),   # |0010>
        (3, 1),   # |0001>
    ])
    def test_flip_each_qubit_in_four_qubit_register(self, index, expected_state_index):  # rotate_x(pi) should flip only its target qubit
        test = Statevector(4)
        test.rotate_x(np.pi, index)
        probs = np.abs(test.data) ** 2
        expected = np.zeros(16)
        expected[expected_state_index] = 1.0
        assert np.allclose(probs, expected)

    def test_state_size_consistent_after_gate(self):  # Statevector length should always equal 2**num_qubits
        test = Statevector(3)
        test.rotate_y(0.9, 1)
        assert len(test.data) == 2 ** test.num_qubits


class TestEntangledState:

    def test_rotation_on_bell_pair_preserves_normalization(self):  # Rotating one half of an entangled pair shouldn't break normalization
        test = Statevector(2)
        hadamard = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
        test.apply(hadamard, 0)
        test.cnot(0, 1)  # adjust to your actual entangling gate method name
        test.rotate_x(0.5, 0)
        assert np.isclose(np.sum(np.abs(test.data) ** 2), 1.0)


class TestParameterEdgeCases:

    def test_negative_angle_inverts_rotation(self):  # Positive rotation followed by its negative should return to original state
        for func in ['rotate_x', 'rotate_y', 'rotate_z']:
            test = Statevector(1)
            original = test.data.copy()
            getattr(test, func)(0.7, 0)
            getattr(test, func)(-0.7, 0)
            assert np.allclose(test.data, original)


class TestInvalidInput:

    def test_out_of_range_index_raises(self):  # Out-of-range index should raise, not silently misbehave
        test = Statevector(2)
        with pytest.raises(Exception):
            test.rotate_x(np.pi, 5)

def test_add_gate_rejects_unknown_gate_name():
    circuit = Circuit(1)
    with pytest.raises(AssertionError):
        circuit.add_gate('not_a_real_gate', [0])
 
 
def test_add_gate_stores_simple_gate():
    circuit = Circuit(1)
    circuit.add_gate('h', [0])
    assert circuit.gates == [['h', [0]]]
 
 
def test_add_gate_stores_rotation_gate_with_theta():
    circuit = Circuit(1)
    circuit.add_gate('rx', [0], theta=np.pi / 2)
    assert circuit.gates == [['rx', np.pi / 2, [0]]]
 
 
# ---------------------------------------------------------------------------
# Circuit: execution
# ---------------------------------------------------------------------------
 
def test_implement_does_not_mutate_caller_statevector():
    template = Statevector(1)
    circuit = Circuit(1)
    circuit.add_gate('x', [0])
    circuit.implement(template)
    # caller's original object must remain untouched
    assert template.probabilities()[0] == pytest.approx(1.0, abs=1e-9)
 
 
def test_implement_returns_final_collapsed_data():
    template = Statevector(1)
    circuit = Circuit(1)
    circuit.add_gate('x', [0])
    _, final_sv = circuit.implement(template)
    probs = final_sv.probabilities()
    assert probs[1] == pytest.approx(1.0, abs=1e-9)
 
 
def test_implement_records_measurement_with_shot_number():
    template = Statevector(1)
    circuit = Circuit(1)
    circuit.add_gate('x', [0])
    circuit.add_gate('measure', [0])
    circuit.implement(template)
    assert len(circuit.results) == 1
    qubit, outcome, shot = circuit.results[0]
    assert qubit == 0
    assert outcome == 1
    assert shot == 1
 
 
def test_implement_accumulates_results_and_increments_shot():
    template = Statevector(1)
    circuit = Circuit(1)
    circuit.add_gate('x', [0])
    circuit.add_gate('measure', [0])
 
    circuit.implement(template)
    circuit.implement(template)
    circuit.implement(template)
 
    assert len(circuit.results) == 3
    shots_seen = [entry[2] for entry in circuit.results]
    assert shots_seen == [1, 2, 3]
 
 
def test_implement_repeated_shots_do_not_leak_collapse_into_template():
    # Regression test for the "does the template stay pristine" property
    template = Statevector(1)
    circuit = Circuit(1)
    circuit.add_gate('h', [0])
    circuit.add_gate('measure', [0])
 
    for _ in range(20):
        circuit.implement(template)
        # template must be exactly |0> before every single shot
        assert template.probabilities()[0] == pytest.approx(1.0, abs=1e-9)
 
 
def test_bell_circuit_measurements_always_agree():
    np.random.seed(2)
    template = Statevector(2)
    circuit = Circuit(2)
    circuit.add_gate('h', [0])
    circuit.add_gate('cnot', [0, 1])
    circuit.add_gate('measure', [0])
    circuit.add_gate('measure', [1])
 
    for _ in range(50):
        circuit.implement(template)
 
    # group outcomes by shot and check q0 == q1 within each shot
    by_shot = {}
    for qubit, outcome, shot in circuit.results:
        by_shot.setdefault(shot, {})[qubit] = outcome
 
    for shot, outcomes in by_shot.items():
        assert outcomes[0] == outcomes[1]
 
 
# ---------------------------------------------------------------------------
# Circuit: reset
# ---------------------------------------------------------------------------
 
def test_reset_clears_results_and_shot_counter():
    template = Statevector(1)
    circuit = Circuit(1)
    circuit.add_gate('x', [0])
    circuit.add_gate('measure', [0])
    circuit.implement(template)
    circuit.implement(template)
 
    circuit.reset()
 
    assert circuit.results == []
    assert circuit.shot == 0
 
 
def test_reset_does_not_clear_the_gate_sequence():
    circuit = Circuit(1)
    circuit.add_gate('h', [0])
    circuit.add_gate('measure', [0])
    circuit.reset()
    # the circuit definition itself should survive a reset
    assert circuit.gates == [['h', [0]], ['measure', [0]]]
