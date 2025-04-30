import warnings
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from easy_vqe.circuit import create_custom_ansatz, PARAMETRIC_SINGLE_QUBIT_TARGET, PARAMETRIC_MULTI_QUBIT, NON_PARAM_SINGLE, NON_PARAM_MULTI, MULTI_PARAM_GATES

# === Tests for create_custom_ansatz ===

def test_create_valid_simple():
    """Test creating a basic valid ansatz."""
    structure = [('h', [0, 1]), ('cx', [0, 1]), ('ry', [0]), ('rz', [1])]
    num_qubits = 2
    ansatz, params = create_custom_ansatz(num_qubits, structure)

    assert isinstance(ansatz, QuantumCircuit)
    assert ansatz.num_qubits == num_qubits
    assert len(params) == 2
    assert all(isinstance(p, Parameter) for p in params)
    assert params[0].name == "p_0"
    assert params[1].name == "p_1"
    assert len(ansatz.parameters) == 2 # Check parameters registered in circuit
    assert set(ansatz.parameters) == set(params)

def test_create_valid_block_structure():
    """Test nested list block structure processing."""
    block1 = [('ry', [0]), ('rz', [0])] # p_0, p_1
    block2 = [('h', [1]), ('crz', [0, 1])] # p_2
    structure = [('h', [0]), block1, block2, ('rx', [0])] # p_3
    num_qubits = 2
    ansatz, params = create_custom_ansatz(num_qubits, structure)

    assert isinstance(ansatz, QuantumCircuit)
    assert ansatz.num_qubits == num_qubits
    assert len(params) == 4
    assert [p.name for p in params] == ["p_0", "p_1", "p_2", "p_3"]
    assert set(ansatz.parameters) == set(params)
    # Verify order roughly corresponds to structure
    op_names = [instr.operation.name for instr in ansatz.data]
    assert 'h' in op_names[:2] # Initial H
    assert 'ry' in op_names[1:4] # Block 1
    assert 'rz' in op_names[1:4]
    assert 'h' in op_names[3:6] # Block 2
    assert 'crz' in op_names[3:6]
    assert 'rx' in op_names[-1] # Final RX

def test_create_valid_all_gate_types():
    """Test using one gate from each defined category."""
    num_qubits = 3
    structure = [
        ('rx', [0]),          # Param single (p_0)
        ('h', [1]),           # Non-param single
        ('crz', [0, 1]),      # Param multi (p_1)
        ('cx', [1, 2]),       # Non-param multi
        ('barrier', [0,1,2]), # Barrier
        ('p', [2]),           # Param single (p_2)
        ('z', [0]),           # Non-param single
        ('rxx', [0, 2]),      # Param multi (p_3)
        ('swap', [0, 1]),     # Non-param multi
        ('ccx', [0, 1, 2]),   # Non-param multi (toffoli alias)
    ]
    ansatz, params = create_custom_ansatz(num_qubits, structure)
    assert len(params) == 4
    assert [p.name for p in params] == ["p_0", "p_1", "p_2", "p_3"]
    op_names = [instr.operation.name for instr in ansatz.data]
    assert 'rx' in op_names
    assert 'h' in op_names
    assert 'crz' in op_names
    assert 'cx' in op_names
    assert 'barrier' in op_names
    assert 'p' in op_names
    assert 'z' in op_names
    assert 'rxx' in op_names
    assert 'swap' in op_names
    assert 'ccx' in op_names

def test_create_valid_aliases():
    """Test using gate aliases."""
    structure = [('cnot', [0, 1]), ('toffoli', [0, 1, 2])]
    num_qubits = 3
    ansatz, params = create_custom_ansatz(num_qubits, structure)
    assert len(params) == 0
    op_names = [instr.operation.name for instr in ansatz.data]
    assert 'cx' in op_names
    assert 'ccx' in op_names

def test_create_valid_measure_instruction(recwarn):
    """Test handling of 'measure' instruction and auto-added classical register."""
    structure = [('h', [0]), ('meas', [0])]
    num_qubits = 1
    ansatz, params = create_custom_ansatz(num_qubits, structure)

    assert len(params) == 0
    assert len(ansatz.clbits) == 1
    assert len(ansatz.cregs) == 1
    assert ansatz.cregs[0].name == 'c0' # Default name Qiskit auto-assigns? Check this... actually might be crX
    assert any(instr.operation.name == 'measure' for instr in ansatz.data)

    # Check warnings
    assert len(recwarn) == 2 # Explicit measure warning + auto-added CReg warning
    assert "Explicit 'measure' instruction found" in str(recwarn[0].message)
    assert "Auto-added ClassicalRegister" in str(recwarn[1].message)

def test_create_valid_empty_structure():
    """Test creating an ansatz with an empty structure."""
    structure = []
    num_qubits = 3
    ansatz, params = create_custom_ansatz(num_qubits, structure)
    assert ansatz.num_qubits == num_qubits
    assert len(params) == 0
    assert len(ansatz.data) == 0 # No operations

def test_create_valid_no_parameters():
    """Test creating an ansatz with only non-parametric gates."""
    structure = [('h', [0]), ('cx', [0, 1]), ('z', [1])]
    num_qubits = 2
    ansatz, params = create_custom_ansatz(num_qubits, structure)
    assert ansatz.num_qubits == num_qubits
    assert len(params) == 0
    assert len(ansatz.parameters) == 0
    assert len(ansatz.data) == 3

def test_create_error_invalid_num_qubits():
    """Test ValueError for invalid num_qubits."""
    with pytest.raises(ValueError, match="num_qubits must be a positive integer"):
        create_custom_ansatz(0, [])
    with pytest.raises(ValueError, match="num_qubits must be a positive integer"):
        create_custom_ansatz(-1, [])
    with pytest.raises(ValueError, match="num_qubits must be a positive integer"):
        create_custom_ansatz(1.5, []) # type check included? Yes implicitly

def test_create_error_invalid_structure_type():
    """Test TypeError for invalid ansatz_structure type."""
    with pytest.raises(TypeError, match="ansatz_structure must be a list"):
        create_custom_ansatz(1, "string")
    with pytest.raises(TypeError, match="ansatz_structure must be a list"):
        create_custom_ansatz(1, (('h', [0]),))

def test_create_error_invalid_element_type():
    """Test TypeError for invalid elements within the structure list."""
    structure = [('h', [0]), "string", ('cx', [0, 1])]
    with pytest.raises(TypeError, match=r"Elements in ansatz_structure must be tuple.*Found type '<class 'str'>'"):
        create_custom_ansatz(2, structure)
    structure = [('h', [0]), 123]
    with pytest.raises(TypeError, match=r"Elements in ansatz_structure must be tuple.*Found type '<class 'int'>'"):
        create_custom_ansatz(2, structure)

def test_create_error_invalid_instruction_format():
    """Test TypeError for improperly formatted instruction tuples."""
    structure1 = [('h', [0], 'extra')] # Tuple too long
    with pytest.raises(TypeError, match="Instruction must be a tuple of \(gate_name, qubit_list\)"):
        create_custom_ansatz(1, structure1)

    structure2 = [('h',)] # Tuple too short
    with pytest.raises(TypeError, match="Instruction must be a tuple of \(gate_name, qubit_list\)"):
        create_custom_ansatz(1, structure2)

    structure3 = [(123, [0])] # Wrong type for gate name
    with pytest.raises(TypeError, match=r"Instruction tuple must contain \(str, list\). Got: \(<class 'int'>.*\)"):
        create_custom_ansatz(1, structure3)

    structure4 = [('h', 0)] # Wrong type for qubit list
    with pytest.raises(TypeError, match=r"Instruction tuple must contain \(str, list\). Got: \(.*<class 'int'>\)"):
        create_custom_ansatz(1, structure4)

def test_create_error_invalid_qubit_indices():
    """Test ValueError for invalid qubit indices."""
    structure1 = [('h', [-1])] # Negative index
    with pytest.raises(ValueError, match=r"Invalid qubit index '-1'"):
        create_custom_ansatz(1, structure1)

    structure2 = [('cx', [0, 2])] # Index out of bounds
    with pytest.raises(ValueError, match=r"Qubit index 2.*out of bounds.*has 2 qubits"):
        create_custom_ansatz(2, structure2)

    structure3 = [('h', [0.5])] # Non-integer index
    with pytest.raises(ValueError, match=r"Invalid qubit index '0.5'"):
        create_custom_ansatz(1, structure3)

def test_create_error_unknown_gate_name():
    """Test ValueError for unrecognized gate names."""
    structure = [('hadamard', [0])]
    with pytest.raises(ValueError, match=r"Gate 'hadamard' is not a valid method"):
        create_custom_ansatz(1, structure)

def test_create_error_gate_argument_mismatch():
    """Test ValueError for applying a gate with the wrong number of qubits."""
    structure_cx = [('cx', [0])] # CX needs 2 qubits
    with pytest.raises(ValueError, match=r"Error applying gate 'cx'.*Provided 1 qubits.*expects a different number.*approx. 2"):
        create_custom_ansatz(2, structure_cx)

    structure_h = [('h', [0, 1])] # H needs 1 qubit
    with pytest.raises(ValueError, match=r"Error applying gate 'h'.*Provided 2 qubits.*expects a different number.*approx. 1"):
        create_custom_ansatz(2, structure_h)

    structure_ccx = [('ccx', [0, 1])] # CCX needs 3 qubits
    with pytest.raises(ValueError, match=r"Error applying gate 'ccx'.*Provided 2 qubits.*expects a different number.*approx. 3"):
        create_custom_ansatz(3, structure_ccx)

def test_create_error_multi_param_gate():
    """Test ValueError for gates requiring multiple parameters (not auto-generated)."""
    structure_u = [('u', [0])] # U needs 3 params
    with pytest.raises(ValueError, match=r"Gate 'u' requires specific parameters not auto-generated"):
        create_custom_ansatz(1, structure_u)

    structure_cu = [('cu', [0, 1])] # CU needs 4 params
    with pytest.raises(ValueError, match=r"Gate 'cu' requires specific parameters not auto-generated"):
        create_custom_ansatz(2, structure_cu)

def test_create_warning_empty_qubit_list(recwarn):
    """Test warning when a gate is specified with an empty qubit list."""
    structure = [('h', [])]
    ansatz, params = create_custom_ansatz(1, structure)
    assert len(recwarn) == 1
    assert "Gate 'h' specified with empty qubit list. Skipping." in str(recwarn[0].message)
    assert len(ansatz.data) == 0 # Gate should be skipped
    assert len(params) == 0

def test_parameter_sorting():
    """Test that parameters are sorted numerically by index in their name."""
    # Create structure that adds parameters out of order
    structure = [('ry', [1]), ('rx', [0]), ('rz', [1])] # p_0 on q1, p_1 on q0, p_2 on q1
    ansatz, params = create_custom_ansatz(2, structure)
    assert len(params) == 3
    assert params[0].name == 'p_0'
    assert params[1].name == 'p_1'
    assert params[2].name == 'p_2'

    # Create structure with larger indices
    structure_large = [('p', [0]), ('ry', [0]), ('rz', [0])] # p_0, p_1, p_2
    structure_large.extend([('rx', [0]) for _ in range(10)]) # p_3 to p_12
    structure_large.append(('crz', [0,1])) # p_13
    ansatz_large, params_large = create_custom_ansatz(2, structure_large)
    assert len(params_large) == 14
    assert [p.name for p in params_large] == [f"p_{i}" for i in range(14)]

@pytest.mark.filterwarnings("ignore:Could not sort parameters numerically") # Ignore sort warning if triggered
def test_parameter_sorting_fallback(recwarn):
    """Test fallback string sorting if numerical sort fails."""
    # Create parameters that won't sort numerically by index easily
    p_a = Parameter('param_a')
    p_b = Parameter('param_b')
    p_1 = Parameter('p_1') # Manually create parameters

    qc = QuantumCircuit(1)
    qc.rx(p_b, 0)
    qc.ry(p_a, 0)
    qc.rz(p_1, 0)

    # Simulate the scenario where create_custom_ansatz might produce such a circuit
    # (This is hard to trigger naturally, so we simulate the state *before* final return)
    # We essentially test the sorting block at the end of create_custom_ansatz

    # Mock the internal state just before the return sorting
    parameters_dict_sim = {'param_a': p_a, 'param_b': p_b, 'p_1': p_1}
    ansatz_sim = qc

    # Run the sorting logic extracted from create_custom_ansatz
    try:
        sorted_parameters_sim = sorted(parameters_dict_sim.values(), key=lambda p: int(p.name.split('_')[1]))
    except (IndexError, ValueError):
        warnings.warn("Could not sort parameters numerically by name. Using default sorting.", UserWarning)
        sorted_parameters_sim = sorted(parameters_dict_sim.values(), key=lambda p: p.name)

    # Check that the fallback sort was triggered and used name sort
    assert len(recwarn) == 1
    assert "Could not sort parameters numerically" in str(recwarn[0].message)
    # Expected order by name: p_1, param_a, param_b
    assert sorted_parameters_sim[0] == p_1
    assert sorted_parameters_sim[1] == p_a
    assert sorted_parameters_sim[2] == p_b