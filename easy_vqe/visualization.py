"""
Visualization Module for VQE 

This module contains functions for visualizing and printing results from the VQE optimization process.
It includes functions to print a summary of the results and to draw the final bound circuit based on the optimization result.
"""

from typing import Dict, Any
import numpy as np

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

    if 'error' in results:
        print(f"VQE Run Failed: {results['error']}")
        if 'details' in results: print(f"Details: {results['details']}")
    else:
        print(f"Hamiltonian: {results['hamiltonian_expression']}")
        print(f"Determined Number of Qubits: {results['num_qubits']}")
        print(f"Optimizer Method: {results['optimizer_method']}")
        print(f"Shots per evaluation: {results['n_shots']}")
        print(f"Optimizer Success: {results['success']}")
        print(f"Optimizer Message: {results['message']}")
        print(f"Final Function Evaluations: {results['optimization_result'].nfev}")
        print(f"Minimum Energy Found: {results['optimal_value']:.10f}")

        optimal_params = results['optimal_params']
        if len(optimal_params) < 15:
            print(f"Optimal Parameters Found:\n{np.round(optimal_params, 5)}")
        else:
            print(f"Optimal Parameters Found: (Array length {len(optimal_params)})")
            print(f"  First 5: {np.round(optimal_params[:5], 5)}")
            print(f"  Last 5:  {np.round(optimal_params[-5:], 5)}")

        if results.get('plot_filename'):
            print(f"Convergence plot saved to: {results['plot_filename']}")
        print("="*40)


def draw_final_bound_circuit(result_dict: Dict[str, Any]) -> None:
    """
    Displays the final bound circuit based on the optimization result.

    Args:
        result_dict: Dictionary containing VQE results, including 'ansatz' and 'optimal_params'.

    Returns:
        None
    """
    ansatz = result_dict.get('ansatz')
    optimal_params = result_dict.get('optimal_params')

    if ansatz is None or optimal_params is None:
        print("[Warning] No ansatz or optimal parameters found in result dictionary.")
        return

    final_circuit = ansatz.copy(name="Final_Bound_Circuit")
    final_circuit = final_circuit.assign_parameters(optimal_params)

    print("\nFinal Bound Circuit:")
    print(final_circuit.draw(output='text', fold=-1))
    print("-" * 50)