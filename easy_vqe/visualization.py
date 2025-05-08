"""
Visualization Module for VQE

This module contains functions for visualizing and printing results from the VQE optimization process.
It includes functions to print a summary of the results and to draw the final bound circuit based on the optimization result.
"""

from typing import Dict, Any
import numpy as np
from qiskit import QuantumCircuit 
import matplotlib.pyplot as plt

def print_results_summary(results: Dict[str, Any]) -> None:
    """
    Prints a summary of the optimization results.

    Args:
        result_dict: Dictionary containing VQE results, including 'optimal_value'.

    Returns:
        None
    """
    print("\n" + "="*40)
    print("          VQE Final Results Summary")
    print("="*40)

    if results.get('error'):
        print(f"VQE Run Failed: {results['error']}")
        if 'details' in results: print(f"Details: {results['details']}")
    else:
        print(f"Hamiltonian: {results.get('hamiltonian_expression', 'N/A')}")
        print(f"Determined Number of Qubits: {results.get('num_qubits', 'N/A')}")
        print(f"Optimizer Method: {results.get('optimizer_method', 'N/A')}")
        print(f"Shots per evaluation: {results.get('n_shots', 'N/A')}")
        print(f"Optimizer Success: {results.get('success', 'N/A')}")
        print(f"Optimizer Message: {results.get('message', 'N/A')}")

        opt_result = results.get('optimization_result')
        eval_count = 'N/A'
        if opt_result and hasattr(opt_result, 'nfev'):
            eval_count = opt_result.nfev
        elif results.get('cost_history'):
            eval_count = f"{len(results['cost_history'])} (from history)"

        print(f"Final Function Evaluations: {eval_count}")

        optimal_value = results.get('optimal_value')
        if optimal_value is not None and np.isfinite(optimal_value):
            print(f"Minimum Energy Found: {optimal_value:.10f}")
        elif optimal_value is not None:
             print(f"Final Value Found: {optimal_value}")
        else:
             print(f"Minimum Energy Found: N/A")


        optimal_params = results.get('optimal_params')
        if optimal_params is not None:
            if len(optimal_params) < 15:
                print(f"Optimal Parameters Found:\n{np.round(optimal_params, 5)}")
            else:
                print(f"Optimal Parameters Found: (Array length {len(optimal_params)})")
                print(f"  First 5: {np.round(optimal_params[:5], 5)}")
                print(f"  Last 5:  {np.round(optimal_params[-5:], 5)}")
        else:
             print("Optimal Parameters Found: N/A")


        if results.get('plot_filename'):
            print(f"Convergence plot saved to: {results['plot_filename']}")

    print("="*40)


def draw_final_bound_circuit(result_dict: Dict[str, Any], draw_type: str, circuit_name: str) -> None:
    """
    Displays the final bound circuit based on the optimization result.

    Args:
        result_dict: Dictionary containing VQE results, including 'ansatz' and 'optimal_params'.
        draw_type: Type of circuit drawing ('text', 'latex', or 'mpl').
        circuit_name: Name to use for the circuit and saved file.

    Returns:
        None
    """
    ansatz = result_dict.get('ansatz')
    optimal_params = result_dict.get('optimal_params')
    parameters = result_dict.get('parameters', []) 

    if not isinstance(ansatz, QuantumCircuit):
        print("[Warning] No valid ansatz QuantumCircuit found in result dictionary.")
        return
    if not isinstance(optimal_params, np.ndarray) or optimal_params.size == 0:
        # Handle case where optimal_params might be None or empty array
        if len(parameters) == 0 and optimal_params is not None and optimal_params.size == 0:
             # Special case: 0 parameters, empty array is valid
             print("\nFinal Bound Circuit (Ansatz has no parameters):")
             if draw_type == 'mpl':
                 fig = ansatz.draw(output='mpl', fold=-1)
                 fig.savefig(f'{circuit_name}.png', bbox_inches='tight', dpi=300)
                 print(f"Circuit visualization saved to '{circuit_name}.png'")
             else:
                 print(ansatz.draw(output='text', fold=-1))
             print("-" * 50)
        else:
            print("[Warning] No valid optimal parameters found in result dictionary.")
        return
    # Check if number of params matches ansatz parameters
    if len(optimal_params) != len(parameters):
        print(f"[Warning] Mismatch between optimal parameters ({len(optimal_params)}) and ansatz parameters ({len(parameters)}). Cannot bind.")
        return

    try:
        final_circuit = ansatz.copy(name=circuit_name)
        param_map = {p: v for p, v in zip(parameters, optimal_params)}
        final_circuit = final_circuit.assign_parameters(param_map)

        print("\nFinal Bound Circuit:")
        if draw_type == 'mpl':
            fig = final_circuit.draw(output='mpl', fold=-1)
            fig.savefig(f'{circuit_name}.png', bbox_inches='tight', dpi=300)
            print(f"Circuit visualization saved to '{circuit_name}.png'")
        else:
            print(final_circuit.draw(output='text', fold=-1))
        print("-" * 50)
    except Exception as e:
        print(f"[Error] Failed to bind parameters or draw the final circuit: {e}")