"""
Easy VQE Package
----------------

A simple interface for running Variational Quantum Eigensolver (VQE)
simulations using Qiskit, focusing on ease of use for defining Hamiltonians
and ansatz structures.
"""

from .vqe_core import find_ground_state
from .vqe_core import create_custom_ansatz, parse_hamiltonian_expression

__version__ = "0.1.0" 

__all__ = [
    'find_ground_state',
    'create_custom_ansatz',
    'parse_hamiltonian_expression',
    '__version__'
]

import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())