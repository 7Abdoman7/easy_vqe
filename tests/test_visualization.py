import pytest
import numpy as np
from unittest.mock import patch, MagicMock, call
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from scipy.optimize import OptimizeResult

from easy_vqe.visualization import print_results_summary, draw_final_bound_circuit

# === Fixtures ===

@pytest.fixture
def mock_successful_result():
    """Creates a mock result dictionary for a successful VQE run."""
    p0, p1 = Parameter('p_0'), Parameter('p_1')
    ansatz = QuantumCircuit(2, name="MockAnsatz")
    ansatz.rx(p0, 0)
    ansatz.cx(0, 1)
    ansatz.ry(p1, 1)

    opt_result = OptimizeResult(
        x=np.array([1.23456, -0.98765]),
        fun=-1.87654321,
        success=True,
        message="Optimization terminated successfully.",
        nfev=120,
        nit=50
    )
    return {
        'optimal_params': opt_result.x,
        'optimal_value': opt_result.fun,
        'num_qubits': 2,
        'ansatz': ansatz,
        'parameters': [p0, p1],
        'optimization_result': opt_result,
        'cost_history': [-0.5, -1.0, -1.5, -1.8, opt_result.fun],
        'parameter_history': [np.array([0.1, 0.1]), np.array([0.5, -0.2]), np.array([1.0, -0.5]), np.array([1.2, -0.9]), opt_result.x],
        'success': True,
        'message': opt_result.message,
        'n_shots': 1024,
        'optimizer_method': 'COBYLA',
        'hamiltonian_expression': '-1.0*ZZ + 0.5*XI',
        'plot_filename': 'convergence.png',
    }

@pytest.fixture
def mock_failed_result():
    """Creates a mock result dictionary for a failed VQE run."""
    return {
        'error': 'Optimization process failed',
        'details': 'Max iterations reached without convergence.',
        'hamiltonian_expression': 'X',
        'optimizer_method': 'Nelder-Mead',
        'n_shots': 500,
        'plot_filename': None, # Indicate plot wasn't saved
        # Might have some partial history even on failure
        'cost_history': [0.5, 0.3],
        'parameter_history': [np.zeros(1), np.ones(1)*0.1],
        'success': False,
        'message': 'Max iterations reached',
        'num_qubits': 1, # May or may not be present depending on where failure occurred
    }

@pytest.fixture
def mock_result_long_params():
    """Creates a mock result with many parameters."""
    num_params = 20
    params = [Parameter(f'p_{i}') for i in range(num_params)]
    ansatz = QuantumCircuit(5) # Example qubits
    for i, p in enumerate(params):
        ansatz.rx(p, i % 5)

    opt_result = OptimizeResult(
        x=np.linspace(0, np.pi, num_params),
        fun=-5.123,
        success=True,
        message="Optimization successful.",
        nfev=200
    )
    return {
        'optimal_params': opt_result.x,
        'optimal_value': opt_result.fun,
        'num_qubits': 5,
        'ansatz': ansatz,
        'parameters': params,
        'optimization_result': opt_result,
        'cost_history': [-1.0, -3.0, opt_result.fun],
        'parameter_history': [np.zeros(num_params), np.ones(num_params)*0.5, opt_result.x],
        'success': True,
        'message': opt_result.message,
        'n_shots': 4096,
        'optimizer_method': 'L-BFGS-B',
        'hamiltonian_expression': 'Complicated Hamiltonian',
        'plot_filename': None,
    }


# === Tests for print_results_summary ===

@patch('builtins.print') # Mock the print function
def test_print_summary_successful(mock_print, mock_successful_result):
    """Test printing the summary for a successful run."""
    print_results_summary(mock_successful_result)

    # Check that print was called multiple times
    assert mock_print.call_count > 10

    # Convert calls to string for easier searching
    output = "\n".join([str(c.args[0]) for c in mock_print.call_args_list])

    # Check for key information presence
    assert "VQE Final Results Summary" in output
    assert f"Hamiltonian: {mock_successful_result['hamiltonian_expression']}" in output
    assert f"Determined Number of Qubits: {mock_successful_result['num_qubits']}" in output
    assert f"Optimizer Method: {mock_successful_result['optimizer_method']}" in output
    assert f"Shots per evaluation: {mock_successful_result['n_shots']}" in output
    assert f"Optimizer Success: {mock_successful_result['success']}" in output
    assert f"Optimizer Message: {mock_successful_result['message']}" in output
    assert f"Final Function Evaluations: {mock_successful_result['optimization_result'].nfev}" in output
    assert f"Minimum Energy Found: {mock_successful_result['optimal_value']:.10f}" in output
    # Check parameter printing (short case)
    assert "Optimal Parameters Found:\n[ 1.23457 -0.98765]" in output # Check rounded output
    assert f"Convergence plot saved to: {mock_successful_result['plot_filename']}" in output


@patch('builtins.print')
def test_print_summary_successful_long_params(mock_print, mock_result_long_params):
    """Test printing the summary with a long parameter list."""
    print_results_summary(mock_result_long_params)
    output = "\n".join([str(c.args[0]) for c in mock_print.call_args_list])

    assert f"(Array length {len(mock_result_long_params['optimal_params'])})" in output
    assert "First 5:" in output
    assert "Last 5:" in output
    # Check that the full array is NOT printed directly
    assert f"\n{np.round(mock_result_long_params['optimal_params'], 5)}" not in output
    # Check plot filename absence
    assert "Convergence plot saved to:" not in output


@patch('builtins.print')
def test_print_summary_failure(mock_print, mock_failed_result):
    """Test printing the summary for a failed run."""
    print_results_summary(mock_failed_result)
    output = "\n".join([str(c.args[0]) for c in mock_print.call_args_list])

    assert "VQE Final Results Summary" in output
    assert f"VQE Run Failed: {mock_failed_result['error']}" in output
    assert f"Details: {mock_failed_result['details']}" in output
    # Ensure success-specific fields are not printed
    assert "Minimum Energy Found:" not in output
    assert "Optimal Parameters Found:" not in output


# === Tests for draw_final_bound_circuit ===

@patch('builtins.print')
@patch('qiskit.QuantumCircuit.draw', return_value="Mock Circuit Drawing") # Mock draw method
@patch('qiskit.QuantumCircuit.assign_parameters', return_value=QuantumCircuit(1)) # Mock assign
def test_draw_final_circuit_successful(mock_assign, mock_draw, mock_print, mock_successful_result):
    """Test drawing the final bound circuit successfully."""
    # We need to make assign_parameters return a circuit that draw can be called on
    mock_bound_circuit = MagicMock(spec=QuantumCircuit)
    mock_bound_circuit.draw.return_value = "Mock Circuit Drawing Text"
    mock_assign.return_value = mock_bound_circuit

    draw_final_bound_circuit(mock_successful_result)

    # Check assign_parameters was called with the optimal parameters
    mock_assign.assert_called_once_with(mock_successful_result['optimal_params'])

    # Check that the draw method of the *returned* circuit was called
    mock_bound_circuit.draw.assert_called_once_with(output='text', fold=-1)

    # Check that print was called to output the title and the drawing
    assert call("Mock Circuit Drawing Text") in mock_print.call_args_list
    assert call("\nFinal Bound Circuit:") in mock_print.call_args_list


@patch('builtins.print')
@patch('qiskit.QuantumCircuit.draw')
@patch('qiskit.QuantumCircuit.assign_parameters')
def test_draw_final_circuit_missing_data(mock_assign, mock_draw, mock_print):
    """Test drawing when ansatz or parameters are missing."""
    result_no_ansatz = {'optimal_params': np.array([1.0])}
    result_no_params = {'ansatz': QuantumCircuit(1)}
    result_empty = {}

    draw_final_bound_circuit(result_no_ansatz)
    assert call("[Warning] No ansatz or optimal parameters found in result dictionary.") in mock_print.call_args_list
    mock_assign.assert_not_called()
    mock_draw.assert_not_called()
    mock_print.reset_mock() # Reset for next call

    draw_final_bound_circuit(result_no_params)
    assert call("[Warning] No ansatz or optimal parameters found in result dictionary.") in mock_print.call_args_list
    mock_assign.assert_not_called()
    mock_print.reset_mock()

    draw_final_bound_circuit(result_empty)
    assert call("[Warning] No ansatz or optimal parameters found in result dictionary.") in mock_print.call_args_list
    mock_assign.assert_not_called()
    mock_draw.assert_not_called()