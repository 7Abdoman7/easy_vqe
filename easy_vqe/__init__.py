"""
Easy VQE Package
----------------

A simple interface for running Variational Quantum Eigensolver (VQE)
simulations using Qiskit, focusing on ease of use for defining Hamiltonians
and ansatz structures.
"""

from .vqe_core import find_ground_state
from .vqe_core import draw_final_bound_circuit
from .vqe_core import print_results_summary
from .vqe_core import get_theoretical_ground_state_energy, parse_hamiltonian_expression
from .vqe_core import create_custom_ansatz


__version__ = "1.0.0"


__all__ = [
    'find_ground_state',
    'create_custom_ansatz',
    'parse_hamiltonian_expression',
    'draw_final_bound_circuit',
    'print_results_summary',
    'get_theoretical_ground_state_energy',
    '__version__'
]

import logging
logging.getLogger(__name__).addHandler(logging.NullHandler()) 