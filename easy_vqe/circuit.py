"""
Circuit creation and manipulation functions for Easy VQE.

This module provides tools for creating custom parameterized quantum circuits
that serve as ansatzes for the VQE algorithm.
"""

import warnings
import re
from typing import Set, List, Tuple, Union, Dict
from qiskit import QuantumCircuit, ClassicalRegister
from qiskit.circuit import Parameter

# Gate type categorization
PARAMETRIC_SINGLE_QUBIT_TARGET: Set[str] = {'rx', 'ry', 'rz', 'p', 'u1'} # Added u1 as alias for p
PARAMETRIC_MULTI_QUBIT: Set[str] = {'crx', 'cry', 'crz', 'cp', 'rxx', 'ryy', 'rzz', 'rzx', 'cu1'} # Removed cu3, u2, u3 (multi-param)
NON_PARAM_SINGLE: Set[str] = {'h', 's', 't', 'x', 'y', 'z', 'sdg', 'tdg', 'id'}
NON_PARAM_MULTI: Set[str] = {'cx', 'cy', 'cz', 'swap', 'ccx', 'cswap', 'ch'}
MULTI_PARAM_GATES: Set[str] = {'u', 'cu', 'r', 'u2', 'u3', 'cu3'} # Gates requiring specific parameter handling


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

        if not isinstance(gate_name, str) or not isinstance(qubit_indices, list):
             raise TypeError(f"Instruction tuple must contain (str, list). Got: ({type(gate_name)}, {type(qubit_indices)})")

        gate_name = gate_name.lower()

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
        gate_method = None # Initialize gate_method

        # Handle aliases first
        if gate_name == 'cnot': gate_name = 'cx'
        elif gate_name == 'toffoli': gate_name = 'ccx'
        elif gate_name == 'meas': gate_name = 'measure'
        elif gate_name == 'u1': gate_name = 'p' # U1 is Phase gate

        if hasattr(current_ansatz, gate_name):
             gate_method = getattr(current_ansatz, gate_name)
        else:
             raise ValueError(f"Gate '{original_gate_name}' is not a valid method of QuantumCircuit (or a known alias like 'cnot', 'toffoli', 'meas', 'u1').")


        try:
            if gate_name in MULTI_PARAM_GATES:
                 raise ValueError(f"Gate '{original_gate_name}' requires multiple parameters which are not auto-generated "
                                  "by this simple format. Construct this gate explicitly if needed.")

            if gate_name in PARAMETRIC_SINGLE_QUBIT_TARGET:
                for q_idx in qubit_indices:
                    param_name = f"p_{p_idx_ref[0]}"
                    p_idx_ref[0] += 1
                    if param_name not in params_dict:
                         params_dict[param_name] = Parameter(param_name)
                    gate_method(params_dict[param_name], q_idx)

            elif gate_name in NON_PARAM_SINGLE:
                for q_idx in qubit_indices:
                    gate_method(q_idx)

            elif gate_name in PARAMETRIC_MULTI_QUBIT: # Excludes MULTI_PARAM_GATES now
                 param_name = f"p_{p_idx_ref[0]}"
                 p_idx_ref[0] += 1
                 if param_name not in params_dict:
                      params_dict[param_name] = Parameter(param_name)
                 gate_method(params_dict[param_name], *qubit_indices)

            elif gate_name in NON_PARAM_MULTI:
                 gate_method(*qubit_indices)

            elif gate_name == 'barrier':
                 gate_method(qubit_indices)

            elif gate_name == 'measure':
                 warnings.warn("Explicit 'measure' instruction found in ansatz structure. "
                               "Measurements are typically added separately based on Hamiltonian terms.", UserWarning)

                 # Try to find or create a suitable classical register
                 creg_name = 'meas_reg' # Default name
                 target_creg = None
                 if current_ansatz.cregs:
                     # Look for existing register of correct size
                     for reg in current_ansatz.cregs:
                         if len(reg) == len(qubit_indices):
                              target_creg = reg
                              break
                     # If no suitable existing register, create one
                     if target_creg is None:
                          reg_suffix = 0
                          while f"{creg_name}{reg_suffix}" in [r.name for r in current_ansatz.cregs]:
                              reg_suffix += 1
                          target_creg = ClassicalRegister(len(qubit_indices), name=f"{creg_name}{reg_suffix}")
                          current_ansatz.add_register(target_creg)
                          warnings.warn(f"Auto-added ClassicalRegister({len(qubit_indices)}) named '{target_creg.name}' for measure.", UserWarning)
                 else: # No classical registers exist yet
                     target_creg = ClassicalRegister(len(qubit_indices), name=creg_name)
                     current_ansatz.add_register(target_creg)
                     warnings.warn(f"Auto-added ClassicalRegister({len(qubit_indices)}) named '{target_creg.name}' for measure.", UserWarning)

                 try:
                      current_ansatz.measure(qubit_indices, target_creg) # Measure into the found/created register
                 except Exception as me:
                      raise RuntimeError(f"Failed to apply 'measure' to qubits {qubit_indices} and register {target_creg.name}. Error: {me}")

            else: # Fallback / Unknown by categories - should not happen if alias/gate check works
                 raise RuntimeError(f"Internal Error: Gate '{original_gate_name}' passed initial checks but was not categorized.")


        except TypeError as e:
             # Determine expected number of qubits based on common gates
             num_expected_qubits = 'unknown'
             # Simple gates
             if gate_name in PARAMETRIC_SINGLE_QUBIT_TARGET or gate_name in NON_PARAM_SINGLE: num_expected_qubits = 1
             # Common 2-qubit gates
             elif gate_name in {'cx','cz','cy','swap','cp','crx','cry','crz','rxx','ryy','rzz','rzx','cu1'}: num_expected_qubits = 2
             # Common 3-qubit gates
             elif gate_name in {'ccx', 'cswap'}: num_expected_qubits = 3
             # Use inspect? Could be complex. Keep simple mapping for now.

             raise ValueError(
                 f"Error applying gate '{original_gate_name}'. Qiskit TypeError: {e}. "
                 f"Provided {len(qubit_indices)} qubits: {qubit_indices}. "
                 f"Gate likely expects a different number of qubits (approx. {num_expected_qubits}) or parameters. "
                 f"(Check Qiskit docs for '{gate_method.__name__}' signature)."
             )
        except ValueError as ve: # Catch the specific ValueError raised for MULTI_PARAM_GATES
            raise ve # Re-raise it directly
        except Exception as e:
             raise RuntimeError(f"Unexpected error applying gate '{original_gate_name}' to qubits {qubit_indices}: {e}")


    structure_queue = list(ansatz_structure)
    while structure_queue:
        element = structure_queue.pop(0)
        if isinstance(element, tuple):
             _process_instruction(element, ansatz, parameters_dict, param_idx_ref)
        elif isinstance(element, list):
             structure_queue[0:0] = element
        else:
             raise TypeError(f"Elements in ansatz_structure must be tuple (gate, qubits) or list (block). "
                             f"Found type '{type(element)}': {element}")

    # Sort parameters collected during processing
    try:
        # Try numerical sort first based on index in name (e.g., p_0, p_1, p_10)
        sorted_collected_parameters = sorted(parameters_dict.values(), key=lambda p: int(re.search(r'\d+', p.name).group()) if re.search(r'\d+', p.name) else float('inf'))
    except (AttributeError, IndexError, ValueError, TypeError):
        # Fallback to string sort if numerical fails
        warnings.warn("Could not sort collected parameters numerically by name. Using default string sorting.", UserWarning)
        sorted_collected_parameters = sorted(parameters_dict.values(), key=lambda p: p.name)

    # Compare with parameters actually present in the circuit
    circuit_parameter_set = set(ansatz.parameters)
    collected_parameter_set = set(sorted_collected_parameters)

    if circuit_parameter_set != collected_parameter_set:
        # Discrepancy found, use the parameters from the circuit as the source of truth
        warnings.warn(f"Parameter mismatch detected. Circuit params ({len(circuit_parameter_set)}): {[p.name for p in sorted(list(circuit_parameter_set), key=lambda p: p.name)]}, "
                      f"Collected ({len(collected_parameter_set)}): {[p.name for p in sorted(list(collected_parameter_set), key=lambda p: p.name)]}. "
                      "Using sorted list derived directly from circuit.parameters.", UserWarning)
        try:
            # Sort the circuit's parameters using the same logic
            circuit_params_sorted = sorted(list(circuit_parameter_set), key=lambda p: int(re.search(r'\d+', p.name).group()) if re.search(r'\d+', p.name) else float('inf'))
        except (AttributeError, IndexError, ValueError, TypeError):
            warnings.warn("Could not sort circuit parameters numerically by name. Using default string sorting for final list.", UserWarning)
            circuit_params_sorted = sorted(list(circuit_parameter_set), key=lambda p: p.name)

        # Log detailed differences if sets don't match exactly
        if not collected_parameter_set.issubset(circuit_parameter_set):
             missing_in_circuit = collected_parameter_set - circuit_parameter_set
             warnings.warn(f"Internal bookkeeping issue: Collected parameters {[p.name for p in missing_in_circuit]} are NOT in the final circuit.", UserWarning)
        if not circuit_parameter_set.issubset(collected_parameter_set):
             missing_in_collected = circuit_parameter_set - collected_parameter_set
             warnings.warn(f"Internal bookkeeping issue: Circuit parameters {[p.name for p in missing_in_collected]} were NOT collected during processing.", UserWarning)

        return ansatz, circuit_params_sorted # Return the list derived FROM the circuit

    # If sets match, return the sorted list derived from the collected dict
    return ansatz, sorted_collected_parameters