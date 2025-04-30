import pytest
import numpy as np
from unittest.mock import patch, MagicMock, ANY
import scipy
from scipy.optimize import OptimizeResult
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

# Import modules from easy_vqe
from easy_vqe import vqe_core, hamiltonian, circuit, measurement

# === Fixtures ===

@pytest.fixture
def simple_h2_hamiltonian_str():
    # A common simple Hamiltonian string (coeffs might not be accurate H2)
    return "-0.5 * II + 0.5 * ZZ + 0.2 * XX - 0.3 * YY"

@pytest.fixture
def simple_ansatz_struct():
    # Simple 2-qubit ansatz structure
    return [('h', [0, 1]), ('cry', [0, 1]), ('cx', [1, 0]), ('rz', [0]), ('rx', [1])]
    # Expected params: p_0 (cry), p_1 (rz), p_2 (rx) -> 3 params

@pytest.fixture
def mock_vqe_dependencies(monkeypatch):
    """Mocks core dependencies of find_ground_state."""
    mock_parse = MagicMock(spec=hamiltonian.parse_hamiltonian_expression)
    mock_create_ansatz = MagicMock(spec=circuit.create_custom_ansatz)
    mock_get_expval = MagicMock(spec=measurement.get_hamiltonian_expectation_value)
    mock_minimize = MagicMock(spec=vqe_core.minimize)
    mock_logger_callback = MagicMock(spec=vqe_core.OptimizationLogger.callback)
    mock_plot = MagicMock() # Mock savefig

    monkeypatch.setattr(vqe_core, 'parse_hamiltonian_expression', mock_parse)
    monkeypatch.setattr(vqe_core, 'create_custom_ansatz', mock_create_ansatz)
    monkeypatch.setattr(vqe_core, 'get_hamiltonian_expectation_value', mock_get_expval)
    monkeypatch.setattr(scipy.optimize, 'minimize', mock_minimize)
    monkeypatch.setattr(vqe_core.OptimizationLogger, 'callback', mock_logger_callback)
    monkeypatch.setattr(vqe_core.plt, 'savefig', mock_plot)
    # Prevent plots from showing during tests
    monkeypatch.setattr(vqe_core.plt, 'show', MagicMock())
    monkeypatch.setattr(vqe_core.plt, 'close', MagicMock())

    # Default mock behaviors
    mock_parse.return_value = [(1.0, 'ZZ')] # Default parsed ham
    p0, p1 = Parameter('p_0'), Parameter('p_1')
    mock_create_ansatz.return_value = (QuantumCircuit(2, name="MockAnsatz"), [p0, p1]) # Default ansatz
    mock_get_expval.return_value = -0.5 # Default expectation value
    mock_minimize.return_value = OptimizeResult(
        x=np.array([0.1, 0.2]), fun=-1.0, success=True, message="Success", nfev=10, nit=5
    )

    return {
        "parse": mock_parse,
        "create_ansatz": mock_create_ansatz,
        "get_expval": mock_get_expval,
        "minimize": mock_minimize,
        "logger_callback": mock_logger_callback,
        "savefig": mock_plot,
    }

# === Tests for OptimizationLogger ===

def test_optimization_logger_init():
    logger = vqe_core.OptimizationLogger()
    assert logger.eval_count == 0
    assert logger.params_history == []
    assert logger.value_history == []
    assert logger._last_print_eval == 0

def test_optimization_logger_callback():
    logger = vqe_core.OptimizationLogger()
    params1 = np.array([1.0, 2.0])
    value1 = -0.5
    params2 = np.array([1.1, 2.1])
    value2 = -0.6

    logger.callback(params1, value1)
    assert logger.eval_count == 1
    assert len(logger.params_history) == 1
    assert np.array_equal(logger.params_history[0], params1)
    assert logger.value_history == [value1]
    assert logger.params_history[0] is not params1 # Ensure it's a copy

    logger.callback(params2, value2)
    assert logger.eval_count == 2
    assert len(logger.params_history) == 2
    assert np.array_equal(logger.params_history[1], params2)
    assert logger.value_history == [value1, value2]

@patch('builtins.print')
def test_optimization_logger_callback_display(mock_print):
    logger = vqe_core.OptimizationLogger()
    logger.callback(np.array([0]), 0.1, display_progress=True, print_interval=1)
    mock_print.assert_called_once_with("  Eval:    1 | Energy:  0.10000000")
    assert logger._last_print_eval == 1

    mock_print.reset_mock()
    logger.callback(np.array([1]), 0.2, display_progress=True, print_interval=5) # Should not print yet
    mock_print.assert_not_called()
    assert logger._last_print_eval == 1 # Still 1

    for i in range(4): # Call 4 more times
         logger.callback(np.array([i+2]), 0.3+i*0.1, display_progress=True, print_interval=5)

    # Now it should print (eval 6, last print was 1, interval 5)
    mock_print.assert_called_once_with("  Eval:    6 | Energy:  0.60000000")
    assert logger._last_print_eval == 6


def test_optimization_logger_get_history():
    logger = vqe_core.OptimizationLogger()
    params1 = np.array([1.0])
    value1 = -0.5
    params2 = np.array([1.1])
    value2 = -0.6
    logger.callback(params1, value1)
    logger.callback(params2, value2)

    hist_val, hist_par = logger.get_history()
    assert hist_val == [value1, value2]
    assert len(hist_par) == 2
    assert np.array_equal(hist_par[0], params1)
    assert np.array_equal(hist_par[1], params2)


# === Tests for find_ground_state ===

def test_find_ground_state_successful_run(mock_vqe_dependencies, simple_ansatz_struct, simple_h2_hamiltonian_str):
    """Test a typical successful VQE run workflow."""
    mocks = mock_vqe_dependencies
    ham_str = simple_h2_hamiltonian_str
    ansatz_struct = simple_ansatz_struct
    n_shots = 1000
    optimizer = 'COBYLA'
    initial_params = 'random' # Use random strategy
    plot_file = "test_convergence.png"

    # --- Mock setup specific to this test ---
    parsed_ham = [(-0.5, 'II'), (0.5, 'ZZ'), (0.2, 'XX'), (-0.3, 'YY')] # Example
    num_qubits = 2
    mocks['parse'].return_value = parsed_ham
    # Assume ansatz creator returns 3 parameters for the simple struct
    p0, p1, p2 = Parameter('p_0'), Parameter('p_1'), Parameter('p_2')
    mock_ansatz = QuantumCircuit(num_qubits)
    params_list = [p0, p1, p2]
    mocks['create_ansatz'].return_value = (mock_ansatz, params_list)
    num_params = len(params_list)
    # Mock minimize result
    final_params = np.array([1.1, 2.2, 3.3])
    final_energy = -1.12345
    mocks['minimize'].return_value = OptimizeResult(
        x=final_params, fun=final_energy, success=True, message="Converged", nfev=50
    )
    # Mock get_expval to return decreasing values during optimization trace
    exp_vals_sequence = [-0.1, -0.5, -0.8, -1.0, final_energy]
    mocks['get_expval'].side_effect = exp_vals_sequence

    # --- Run VQE ---
    results = vqe_core.find_ground_state(
        ansatz_structure=ansatz_struct,
        hamiltonian_expression=ham_str,
        n_shots=n_shots,
        optimizer_method=optimizer,
        initial_params_strategy=initial_params,
        plot_filename=plot_file,
        display_progress=False # Keep log clean for asserts
    )

    # --- Assertions ---
    # Check setup calls
    mocks['parse'].assert_called_once_with(ham_str)
    mocks['create_ansatz'].assert_called_once_with(num_qubits, ansatz_struct)

    # Check objective function calls (inside minimize)
    # Initial energy + minimize calls (nfev) = 1 + 50 = 51, but minimize mock returns 50 directly
    # The objective function is called by minimize, AND once before minimize starts.
    # The logger callback is called *within* the objective function.
    assert mocks['get_expval'].call_count >= 1 # Called at least once for initial energy
    # The number of calls depends on the optimizer. Check it was called with the ansatz and parsed ham
    mocks['get_expval'].assert_called_with(ansatz=mock_ansatz, parsed_hamiltonian=parsed_ham, param_values=ANY, n_shots=n_shots)

    # Check minimize call
    mocks['minimize'].assert_called_once()
    minimize_call_args = mocks['minimize'].call_args
    assert minimize_call_args[0][1].shape == (num_params,) # Check initial params shape passed to minimize
    assert minimize_call_args.kwargs['method'] == optimizer
    # Note: Checking the objective function passed requires deeper inspection or different mocking

    # Check logger calls (should match expval calls within minimize, plus initial)
    assert mocks['logger_callback'].call_count == mocks['get_expval'].call_count

    # Check plotting call
    mocks['savefig'].assert_called_once_with(plot_file)

    # Check results dictionary contents
    assert 'error' not in results
    assert results['success'] is True
    assert results['message'] == "Converged"
    assert np.array_equal(results['optimal_params'], final_params)
    assert np.isclose(results['optimal_value'], final_energy)
    assert results['num_qubits'] == num_qubits
    assert results['ansatz'] is mock_ansatz
    assert results['parameters'] == params_list
    assert results['n_shots'] == n_shots
    assert results['optimizer_method'] == optimizer
    assert results['hamiltonian_expression'] == ham_str
    assert results['plot_filename'] == plot_file
    assert results['initial_params_strategy_used'] == 'random'
    assert isinstance(results['initial_params'], np.ndarray)
    assert results['initial_params'].shape == (num_params,)
    # Check history lengths match logger calls (which match expval calls)
    assert len(results['cost_history']) == mocks['logger_callback'].call_count
    assert len(results['parameter_history']) == mocks['logger_callback'].call_count
    assert np.isclose(results['cost_history'][-1], final_energy) # Last cost should be final energy

def test_find_ground_state_hamiltonian_parse_error(mock_vqe_dependencies, simple_ansatz_struct):
    """Test VQE exit if Hamiltonian parsing fails."""
    mocks = mock_vqe_dependencies
    error_msg = "Invalid Pauli string"
    mocks['parse'].side_effect = ValueError(error_msg)

    results = vqe_core.find_ground_state(simple_ansatz_struct, "Invalid H string")

    assert 'error' in results
    assert results['error'] == 'Hamiltonian parsing failed'
    assert results['details'] == error_msg
    mocks['create_ansatz'].assert_not_called()
    mocks['minimize'].assert_not_called()

def test_find_ground_state_ansatz_create_error(mock_vqe_dependencies, simple_h2_hamiltonian_str):
    """Test VQE exit if Ansatz creation fails."""
    mocks = mock_vqe_dependencies
    error_msg = "Qubit index out of bounds"
    mocks['create_ansatz'].side_effect = ValueError(error_msg)

    results = vqe_core.find_ground_state("Valid Struct (mocked)", simple_h2_hamiltonian_str)

    assert 'error' in results
    assert results['error'] == 'Ansatz creation failed'
    assert results['details'] == error_msg
    mocks['parse'].assert_called_once() # Parsing happens first
    mocks['minimize'].assert_not_called()
    assert results.get('num_qubits') is not None # Should be set before ansatz creation

def test_find_ground_state_zero_params(mock_vqe_dependencies, simple_h2_hamiltonian_str):
    """Test VQE with an ansatz having zero parameters."""
    mocks = mock_vqe_dependencies
    ham_str = simple_h2_hamiltonian_str
    ansatz_struct = [('h', [0]), ('cx', [0, 1])] # No params struct

    # Mock setup
    parsed_ham = [(1.0, 'ZZ')]
    num_qubits = 2
    mocks['parse'].return_value = parsed_ham
    mock_ansatz = QuantumCircuit(num_qubits) # No params
    params_list = [] # Empty list
    mocks['create_ansatz'].return_value = (mock_ansatz, params_list)
    fixed_energy = 0.5 # Assume <H> for this fixed circuit is 0.5
    mocks['get_expval'].return_value = fixed_energy

    results = vqe_core.find_ground_state(
        ansatz_struct, ham_str, display_progress=False
    )

    # Assertions
    mocks['parse'].assert_called_once()
    mocks['create_ansatz'].assert_called_once()
    # get_expval called ONCE for the fixed evaluation
    mocks['get_expval'].assert_called_once_with(mock_ansatz, parsed_ham, [], ANY)
    mocks['minimize'].assert_not_called() # Optimizer should be skipped

    assert 'error' not in results
    assert results['success'] is True # Considered successful static evaluation
    assert results['message'] == 'Static evaluation (no parameters)'
    assert np.array_equal(results['optimal_params'], np.array([]))
    assert np.isclose(results['optimal_value'], fixed_energy)
    assert results['cost_history'] == [fixed_energy]
    assert len(results['parameter_history']) == 1
    assert np.array_equal(results['parameter_history'][0], np.array([]))

@pytest.mark.parametrize("strategy, expected_print", [
    ('random', "Strategy: Using 'random'"),
    ('zeros', "Strategy: Using 'zeros'"),
    (np.array([0.1, 0.2, 0.3]), "Strategy: Using provided numpy array"),
    ([0.1, 0.2, 0.3], "Strategy: Using provided list/tuple"),
])
@patch('builtins.print') # Mock print to check output
def test_find_ground_state_initial_params_strategies(mock_print, strategy, expected_print, mock_vqe_dependencies, simple_ansatz_struct, simple_h2_hamiltonian_str):
    """Test different initial parameter strategies."""
    mocks = mock_vqe_dependencies
    num_params = 3 # From simple_ansatz_struct fixture expectation
    p0, p1, p2 = Parameter('p_0'), Parameter('p_1'), Parameter('p_2')
    mocks['create_ansatz'].return_value = (QuantumCircuit(2), [p0, p1, p2])

    results = vqe_core.find_ground_state(
        simple_ansatz_struct, simple_h2_hamiltonian_str,
        initial_params_strategy=strategy,
        display_progress=False
    )

    assert 'error' not in results
    initial_params = results['initial_params']
    assert isinstance(initial_params, np.ndarray)
    assert initial_params.shape == (num_params,)
    if isinstance(strategy, str) and strategy == 'zeros':
        assert np.allclose(initial_params, 0.0)
    elif isinstance(strategy, (np.ndarray, list)):
         assert np.allclose(initial_params, np.array(strategy))

    # Check print output for strategy message
    output = "\n".join([str(c.args[0]) for c in mock_print.call_args_list])
    assert expected_print in output
    # Check the strategy used is stored correctly
    if isinstance(strategy, str):
        assert results['initial_params_strategy_used'] == strategy
    elif isinstance(strategy, np.ndarray):
         assert results['initial_params_strategy_used'] == 'random' # Fallback currently not implemented, direct use
         # This test case needs adjustment based on how the numpy handling evolves. Assuming direct use now.
         # If it *should* record 'numpy_array', the code needs changing. Let's assume it uses it directly.
         assert results['initial_params_strategy_used'] == 'random' # Placeholder, re-evaluate based on desired behavior
    elif isinstance(strategy, list):
         assert results['initial_params_strategy_used'] == 'random' # Placeholder, re-evaluate

@patch('builtins.print') # Mock print to check output
@patch('numpy.random.uniform') # Mock random to check fallback
def test_find_ground_state_initial_params_warnings(mock_uniform, mock_print, mock_vqe_dependencies, simple_ansatz_struct, simple_h2_hamiltonian_str):
    """Test warnings for incorrect initial parameter shapes/lengths."""
    mocks = mock_vqe_dependencies
    num_params = 3 # Expect 3 params
    p0, p1, p2 = Parameter('p_0'), Parameter('p_1'), Parameter('p_2')
    mocks['create_ansatz'].return_value = (QuantumCircuit(2), [p0, p1, p2])
    mock_uniform.return_value = np.array([0.5]*num_params) # Define fallback random value

    # 1. Numpy array with wrong shape
    vqe_core.find_ground_state(simple_ansatz_struct, simple_h2_hamiltonian_str, initial_params_strategy=np.array([1, 2]), display_progress=False)
    output = "\n".join([str(c.args[0]) for c in mock_print.call_args_list])
    assert "[Warning] Provided initial_params numpy array shape (2,) != expected (3,). Defaulting to 'random'." in output
    assert "Strategy: Using 'random'" in output # Check fallback message
    mock_uniform.assert_called_once() # Random should have been called

    # 2. List with wrong length
    mock_uniform.reset_mock()
    mock_print.reset_mock()
    vqe_core.find_ground_state(simple_ansatz_struct, simple_h2_hamiltonian_str, initial_params_strategy=[1, 2, 3, 4], display_progress=False)
    output = "\n".join([str(c.args[0]) for c in mock_print.call_args_list])
    assert "[Warning] Provided initial_params list/tuple length 4 != expected 3. Defaulting to 'random'." in output
    assert "Strategy: Using 'random'" in output
    mock_uniform.assert_called_once()

    # 3. Unknown string strategy
    mock_uniform.reset_mock()
    mock_print.reset_mock()
    vqe_core.find_ground_state(simple_ansatz_struct, simple_h2_hamiltonian_str, initial_params_strategy='bad_strategy', display_progress=False)
    output = "\n".join([str(c.args[0]) for c in mock_print.call_args_list])
    assert "[Warning] Unknown initial_params_strategy 'bad_strategy'. Defaulting to 'random'." in output
    assert "Strategy: Using 'random'" in output
    mock_uniform.assert_called_once()

def test_find_ground_state_optimizer_options(mock_vqe_dependencies, simple_ansatz_struct, simple_h2_hamiltonian_str):
    """Test passing and auto-setting optimizer options."""
    mocks = mock_vqe_dependencies

    # 1. No options, use max_evaluations default -> sets maxiter for COBYLA
    vqe_core.find_ground_state(simple_ansatz_struct, simple_h2_hamiltonian_str, optimizer_method='COBYLA', max_evaluations=100, display_progress=False)
    minimize_kwargs = mocks['minimize'].call_args.kwargs
    assert 'options' in minimize_kwargs
    assert minimize_kwargs['options'] == {'maxiter': 100}
    mocks['minimize'].reset_mock()

    # 2. No options, use max_evaluations default -> sets maxfun for L-BFGS-B
    vqe_core.find_ground_state(simple_ansatz_struct, simple_h2_hamiltonian_str, optimizer_method='L-BFGS-B', max_evaluations=80, display_progress=False)
    minimize_kwargs = mocks['minimize'].call_args.kwargs
    assert 'options' in minimize_kwargs
    assert minimize_kwargs['options'] == {'maxfun': 80}
    mocks['minimize'].reset_mock()

    # 3. Provide explicit options, overriding max_evaluations
    custom_opts = {'maxiter': 50, 'tol': 1e-5}
    vqe_core.find_ground_state(simple_ansatz_struct, simple_h2_hamiltonian_str, optimizer_method='COBYLA', max_evaluations=100, optimizer_options=custom_opts, display_progress=False)
    minimize_kwargs = mocks['minimize'].call_args.kwargs
    assert 'options' in minimize_kwargs
    assert minimize_kwargs['options'] == custom_opts # Explicit options should be used directly
    mocks['minimize'].reset_mock()

    # 4. Provide explicit options including maxiter/maxfun
    custom_opts_maxfun = {'maxfun': 70, 'eps': 1e-8}
    vqe_core.find_ground_state(simple_ansatz_struct, simple_h2_hamiltonian_str, optimizer_method='L-BFGS-B', max_evaluations=100, optimizer_options=custom_opts_maxfun, display_progress=False)
    minimize_kwargs = mocks['minimize'].call_args.kwargs
    assert 'options' in minimize_kwargs
    assert minimize_kwargs['options'] == custom_opts_maxfun # Explicit options used

def test_find_ground_state_objective_error(mock_vqe_dependencies, simple_ansatz_struct, simple_h2_hamiltonian_str):
    """Test handling of error during expectation value calculation."""
    mocks = mock_vqe_dependencies
    error_msg = "Simulation failed"
    # Make get_expval raise error *after* the initial call
    mocks['get_expval'].side_effect = [-0.5, RuntimeError(error_msg)]

    # Mock minimize to show failure because objective returned non-finite value eventually
    mocks['minimize'].return_value = OptimizeResult(
        x=np.array([0,0]), fun=np.inf, success=False, message="Objective error", nfev=2
    )

    results = vqe_core.find_ground_state(simple_ansatz_struct, simple_h2_hamiltonian_str, display_progress=False)

    # Should still "complete" but indicate optimizer failure
    assert 'error' not in results # The VQE function itself didn't raise an unhandled error
    assert results['success'] is False
    assert "Objective error" in results['message'] # Message from minimize
    # Check that get_expval was called twice (initial + first step)
    assert mocks['get_expval'].call_count == 2
    # Check logger history captured the successful call and potentially the error state
    assert len(results['cost_history']) >= 1 # Should capture at least the first successful call
    assert results['cost_history'][0] == -0.5

def test_find_ground_state_initial_energy_inf(mock_vqe_dependencies, simple_ansatz_struct, simple_h2_hamiltonian_str):
    """Test VQE exit if initial parameters yield infinite energy."""
    mocks = mock_vqe_dependencies
    mocks['get_expval'].return_value = np.inf # Initial call returns inf

    results = vqe_core.find_ground_state(simple_ansatz_struct, simple_h2_hamiltonian_str, display_progress=False)

    assert 'error' in results
    assert results['error'] == 'Initial parameters yield invalid energy (inf).'
    mocks['get_expval'].assert_called_once() # Only the initial call
    mocks['minimize'].assert_not_called()

def test_find_ground_state_optimizer_fails(mock_vqe_dependencies, simple_ansatz_struct, simple_h2_hamiltonian_str):
    """Test VQE handling when the optimizer itself fails."""
    mocks = mock_vqe_dependencies
    mocks['minimize'].side_effect = Exception("Scipy minimize internal error")

    results = vqe_core.find_ground_state(simple_ansatz_struct, simple_h2_hamiltonian_str, display_progress=False)

    assert 'error' in results
    assert results['error'] == 'Optimization process failed'
    assert 'Scipy minimize internal error' in results['details']
    assert results['success'] is False # Indicate failure
    assert mocks['get_expval'].call_count >= 1 # Initial energy was calculated
    assert len(results['cost_history']) >= 1 # History up to the point of failure

def test_find_ground_state_no_plot(mock_vqe_dependencies, simple_ansatz_struct, simple_h2_hamiltonian_str):
    """Test VQE run without generating a plot."""
    mocks = mock_vqe_dependencies
    results = vqe_core.find_ground_state(
        simple_ansatz_struct, simple_h2_hamiltonian_str,
        plot_filename=None, # Explicitly None
        display_progress=False
    )
    assert 'error' not in results
    mocks['savefig'].assert_not_called()
    assert results['plot_filename'] is None

@patch('builtins.print')
def test_find_ground_state_plot_error(mock_print, mock_vqe_dependencies, simple_ansatz_struct, simple_h2_hamiltonian_str):
    """Test warning if saving the plot fails."""
    mocks = mock_vqe_dependencies
    plot_file = "bad_path/plot.png"
    error_msg = "Permission denied"
    mocks['savefig'].side_effect = OSError(error_msg)

    results = vqe_core.find_ground_state(
        simple_ansatz_struct, simple_h2_hamiltonian_str,
        plot_filename=plot_file,
        display_progress=False
    )

    assert 'error' not in results # VQE itself succeeded
    mocks['savefig'].assert_called_once_with(plot_file)
    assert results['plot_filename'] is None 

    # Check for warning print message
    output = "\n".join([str(c.args[0]) for c in mock_print.call_args_list])
    assert f"[Warning] Could not save convergence plot to '{plot_file}': {error_msg}" in output