"""
Circuit creation and manipulation functions for Easy VQE.

This module provides tools for creating custom parameterized quantum circuits
that serve as ansatzes for the VQE algorithm.
"""

import warnings
import numpy as np
from typing import Set, List, Tuple, Union, Dict
from qiskit import QuantumCircuit, ClassicalRegister
from qiskit.circuit import Parameter

# Gate type categorization
PARAMETRIC_SINGLE_QUBIT_TARGET: Set[str] = {'rx', 'ry', 'rz', 'p'}
PARAMETRIC_MULTI_QUBIT: Set[str] = {'crx', 'cry', 'crz', 'cp', 'rxx', 'ryy', 'rzz', 'rzx', 'cu1', 'cu3', 'u2', 'u3'}
NON_PARAM_SINGLE: Set[str] = {'h', 's', 't', 'x', 'y', 'z', 'sdg', 'tdg', 'id'}
NON_PARAM_MULTI: Set[str] = {'cx', 'cy', 'cz', 'swap', 'ccx', 'cswap', 'ch'}
MULTI_PARAM_GATES: Set[str] = {'u', 'cu', 'r'} # Gates requiring specific parameter handling

def create_custom_ansatz(num_qubits: int, ansatz_structure: List[Union[Tuple[str, List[int]], List]]) -> Tuple[QuantumCircuit, List[Parameter]]:
    """
    Creates a parameterized quantum circuit (ansatz) from a simplified structure.

    Automatically generates unique Parameter objects (p_0, p_1, ...) for
    parametric gates based on their order of appearance.

    Args:
        num_qubits: The number of qubits for the circuit.
        ansatz_structure: A list defining the circuit structure. Elements can be:
            - Tuple[str, List[int]]: (gate_name, target_qubit_indices)
            - List: A nested list representing a block of operations, processed sequentially.

    Returns:
        Tuple[QuantumCircuit, List[Parameter]]: A tuple containing:
            - The constructed QuantumCircuit.
            - A sorted list of the Parameter objects used in the circuit.

    Raises:
        ValueError: If input types are wrong, qubit indices are invalid,
                    gate names are unrecognized, or gate application fails.
        TypeError: If the structure format is incorrect.
        RuntimeError: For unexpected errors during gate application.
    """
    if not isinstance(num_qubits, int) or num_qubits <= 0:
        raise ValueError(f"num_qubits must be a positive integer, got {num_qubits}")
    if not isinstance(ansatz_structure, list):
        raise TypeError("ansatz_structure must be a list.")

    ansatz = QuantumCircuit(num_qubits, name="CustomAnsatz")
    parameters_dict: Dict[str, Parameter] = {}
    param_idx_ref: List[int] = [0] 

    def _process_instruction(instruction: Tuple[str, List[int]],
                             current_ansatz: QuantumCircuit,
                             params_dict: Dict[str, Parameter],
                             p_idx_ref: List[int]):
        """Internal helper to apply one gate instruction."""
        if not isinstance(instruction, tuple) or len(instruction) != 2:
            raise TypeError(f"Instruction must be a tuple of (gate_name, qubit_list). Got: {instruction}")

        gate_name, qubit_indices = instruction
        gate_name = gate_name.lower() 

        if not isinstance(gate_name, str) or not isinstance(qubit_indices, list):
             raise TypeError(f"Instruction tuple must contain (str, list). Got: ({type(gate_name)}, {type(qubit_indices)})")

        if gate_name == 'barrier' and not qubit_indices:
             qubit_indices = list(range(current_ansatz.num_qubits))
        elif not qubit_indices and gate_name != 'barrier':
            warnings.warn(f"Gate '{gate_name}' specified with empty qubit list. Skipping.", UserWarning)
            return

        for q in qubit_indices:
             if not isinstance(q, int) or q < 0:
                  raise ValueError(f"Invalid qubit index '{q}' in {qubit_indices} for gate '{gate_name}'. Indices must be non-negative integers.")
             if q >= current_ansatz.num_qubits:
                  raise ValueError(f"Qubit index {q} in {qubit_indices} for gate '{gate_name}' is out of bounds. "
                               f"Circuit has {current_ansatz.num_qubits} qubits (indices 0 to {current_ansatz.num_qubits - 1}).")

        original_gate_name = gate_name 
        if not hasattr(current_ansatz, gate_name):
            if gate_name == 'cnot': gate_name = 'cx'
            elif gate_name == 'toffoli': gate_name = 'ccx'
            elif gate_name == 'meas': gate_name = 'measure' 
            else:
                 raise ValueError(f"Gate '{original_gate_name}' is not a valid method of QuantumCircuit (or a known alias like 'cnot', 'toffoli', 'meas').")
        gate_method = getattr(current_ansatz, gate_name)

        try:
            if gate_name in PARAMETRIC_SINGLE_QUBIT_TARGET:
                for q_idx in qubit_indices:
                    param_name = f"p_{p_idx_ref[0]}" 
                    p_idx_ref[0] += 1
                    if param_name not in params_dict:
                         params_dict[param_name] = Parameter(param_name)
                    gate_method(params_dict[param_name], q_idx)

            # Non-Parametric Single Qubit: Apply individually
            elif gate_name in NON_PARAM_SINGLE:
                for q_idx in qubit_indices:
                    gate_method(q_idx)

            # Parametric Multi Qubit (Single Parameter expected by default structure)
            elif gate_name in PARAMETRIC_MULTI_QUBIT:
                 if gate_name in MULTI_PARAM_GATES:
                     raise ValueError(f"Gate '{original_gate_name}' requires multiple parameters which are not auto-generated "
                                      "by this simple format. Construct this gate explicitly if needed.")
                 param_name = f"p_{p_idx_ref[0]}"
                 p_idx_ref[0] += 1
                 if param_name not in params_dict:
                      params_dict[param_name] = Parameter(param_name)
                 gate_method(params_dict[param_name], *qubit_indices)

            # Non-Parametric Multi Qubit
            elif gate_name in NON_PARAM_MULTI:
                 gate_method(*qubit_indices)

            # Handle Barrier explicitly
            elif gate_name == 'barrier':
                 gate_method(qubit_indices) # Apply barrier to specified qubits

            elif gate_name == 'measure':
                 warnings.warn("Explicit 'measure' instruction found in ansatz structure. "
                               "Measurements are typically added separately based on Hamiltonian terms.", UserWarning)
                 
                 if not current_ansatz.cregs:
                      cr = ClassicalRegister(len(qubit_indices))
                      current_ansatz.add_register(cr)
                      warnings.warn(f"Auto-added ClassicalRegister({len(qubit_indices)}) for measure.", UserWarning)
                 try:
                      current_ansatz.measure(qubit_indices, list(range(len(qubit_indices))))
                 except Exception as me:
                      raise RuntimeError(f"Failed to apply 'measure'. Ensure ClassicalRegister exists or handle measurement outside ansatz structure. Error: {me}")

            else:
                 # Check if it's likely parametric based on name conventions
                 is_likely_parametric = any(gate_name.endswith(p) for p in PARAMETRIC_SINGLE_QUBIT_TARGET) or \
                                        any(gate_name.startswith(p) for p in PARAMETRIC_MULTI_QUBIT)

                 if is_likely_parametric and gate_name not in MULTI_PARAM_GATES:
                      warnings.warn(f"Gate '{original_gate_name}' not in predefined *parametric* categories but looks like one. "
                                    f"Attempting single-parameter multi-qubit application. Specify explicitly if wrong.", UserWarning)
                      param_name = f"p_{p_idx_ref[0]}"
                      p_idx_ref[0] += 1
                      if param_name not in params_dict: params_dict[param_name] = Parameter(param_name)
                      gate_method(params_dict[param_name], *qubit_indices)

                 elif gate_name in MULTI_PARAM_GATES:
                       raise ValueError(f"Gate '{original_gate_name}' requires specific parameters not auto-generated. "
                                        "Use a different construction method or add it to the circuit manually.")
                 else:
                       # Assume non-parametric multi-qubit if not recognized otherwise
                       warnings.warn(f"Gate '{original_gate_name}' not in predefined categories. Assuming non-parametric multi-qubit application.", UserWarning)
                       gate_method(*qubit_indices)

        except TypeError as e:
             num_expected_qubits = 'unknown'
             if gate_name in PARAMETRIC_SINGLE_QUBIT_TARGET or gate_name in NON_PARAM_SINGLE: num_expected_qubits = 1
             elif gate_name in {'cx','cz','cy','cp','crx','cry','crz','swap','rxx','ryy','rzz','rzx', 'cu1', 'u2'}: num_expected_qubits = 2
             elif gate_name in {'ccx', 'cswap', 'ch', 'cu3', 'u3', 'r'}: num_expected_qubits = 3
             elif gate_name in {'u'}: num_expected_qubits = 1
             elif gate_name in {'cu'}: num_expected_qubits = 2
             raise ValueError(
                 f"Error applying gate '{original_gate_name}'. Qiskit TypeError: {e}. "
                 f"Provided {len(qubit_indices)} qubits: {qubit_indices}. "
                 f"Gate likely expects a different number of qubits (approx. {num_expected_qubits}) or parameters. "
                 f"(Check Qiskit docs for '{gate_method.__name__}' signature)."
             )
        except Exception as e:
             raise RuntimeError(f"Unexpected error applying gate '{original_gate_name}' to qubits {qubit_indices}: {e}")

    structure_queue = list(ansatz_structure) 
    while structure_queue:
        element = structure_queue.pop(0) 
        if isinstance(element, tuple):
             _process_instruction(element, ansatz, parameters_dict, param_idx_ref)
        elif isinstance(element, list):
             # Prepend block contents to the front of the queue for sequential processing
             structure_queue[0:0] = element
        else:
             raise TypeError(f"Elements in ansatz_structure must be tuple (gate, qubits) or list (block). "
                             f"Found type '{type(element)}': {element}")

    try:
        sorted_parameters = sorted(parameters_dict.values(), key=lambda p: int(p.name.split('_')[1]))
    except (IndexError, ValueError):
        warnings.warn("Could not sort parameters numerically by name. Using default sorting.", UserWarning)
        sorted_parameters = sorted(parameters_dict.values(), key=lambda p: p.name)

    if set(ansatz.parameters) != set(sorted_parameters):
        warnings.warn(f"Parameter mismatch detected. Circuit params: {len(ansatz.parameters)}, Collected: {len(sorted_parameters)}. "
                      "Using sorted list derived from circuit.parameters.", UserWarning)
        try:
            circuit_params_sorted = sorted(list(ansatz.parameters), key=lambda p: int(p.name.split('_')[1]))
        except (IndexError, ValueError):
            circuit_params_sorted = sorted(list(ansatz.parameters), key=lambda p: p.name)

        if not set(sorted_parameters).issubset(set(ansatz.parameters)):
             warnings.warn("Collected parameters are NOT a subset of circuit parameters. There might be an issue in parameter tracking.", UserWarning)

        return ansatz, circuit_params_sorted

    return ansatz, sorted_parameters