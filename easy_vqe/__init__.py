"""
Easy VQE: A simplified implementation of the Variational Quantum Eigensolver algorithm.

This package provides tools for quantum chemistry simulations using VQE algorithm.
"""

__version__ = "1.1.0"

# Public API exports
from .hamiltonian import parse_hamiltonian_expression, get_theoretical_ground_state_energy
from .circuit import create_custom_ansatz
from .measurement import (
    apply_measurement_basis,
    run_circuit_and_get_counts,
    calculate_term_expectation, 
    get_hamiltonian_expectation_value
)
from .vqe_core import find_ground_state, OptimizationLogger
from .visualization import print_results_summary, draw_final_bound_circuit

__all__ = [
    'parse_hamiltonian_expression',
    'get_theoretical_ground_state_energy',
    'create_custom_ansatz',
    'apply_measurement_basis',
    'run_circuit_and_get_counts',
    'calculate_term_expectation',
    'get_hamiltonian_expectation_value',
    'find_ground_state',
    'OptimizationLogger',
    'print_results_summary',
    'draw_final_bound_circuit',
    '_version_',
]