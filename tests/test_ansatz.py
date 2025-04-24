import pytest
from qiskit.circuit import Parameter
from easy_vqe import create_custom_ansatz

def test_simple_ansatz_creation():
    """Test creating a basic ansatz."""
    structure = [('h', [0, 1]), ('cx', [0, 1]), ('ry', [0, 1])]
    ansatz, params = create_custom_ansatz(num_qubits=2, ansatz_structure=structure)

    assert ansatz.num_qubits == 2
    assert len(params) == 2 
    assert isinstance(params[0], Parameter)
    assert params[0].name == "p_0" 
    assert params[1].name == "p_1"
    assert ansatz.depth() > 2 

def test_block_structure():
    """Test nested list block structure."""
    block = [('ry', [0]), ('rz', [0])]
    structure = [('h', [0]), block, ('rx', [0])]
    ansatz, params = create_custom_ansatz(num_qubits=1, ansatz_structure=structure)

    assert len(params) == 3 
    assert params[0].name == "p_0" 
    assert params[1].name == "p_1" 
    assert params[2].name == "p_2" 

def test_invalid_qubit_index():
    """Test error handling for out-of-bounds qubit index."""
    structure = [('h', [0, 2])] 
    with pytest.raises(ValueError, match=r"Qubit index 2.*out of bounds.*"):
        create_custom_ansatz(num_qubits=2, ansatz_structure=structure)

def test_invalid_gate_name():
    """Test error handling for unknown gate name."""
    structure = [('hadamard', [0])] 
    with pytest.raises(ValueError, match=r"Gate 'hadamard' is not a valid method.*"):
        create_custom_ansatz(num_qubits=1, ansatz_structure=structure)

def test_gate_argument_mismatch():
    """Test error handling for wrong number of qubits for a gate."""
    structure = [('cx', [0])] 
    with pytest.raises(ValueError, match=r"Error applying gate 'cx'.*expects a different number.*"):
         create_custom_ansatz(num_qubits=2, ansatz_structure=structure)

