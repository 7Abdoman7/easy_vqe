"""
Measurement and quantum circuit execution functions for Easy VQE.

This module provides tools for applying measurement basis transformations,
executing quantum circuits, and calculating expectation values from measurement results.
"""

import warnings
import re
import numpy as np
from typing import List, Tuple, Dict, Union, Optional, Sequence, Any
from qiskit import QuantumCircuit, ClassicalRegister
from qiskit.circuit import Parameter
from qiskit import transpile
from qiskit_aer import AerSimulator
from collections.abc import Sequence as ABCSequence # Use alias to avoid conflict

_simulator_instance: Optional[AerSimulator] = None

def get_simulator() -> AerSimulator:
    """
    Initializes and returns the AerSimulator instance (lazy initialization).

    Returns:
        AerSimulator: The simulator instance.
    """
    global _simulator_instance
    if _simulator_instance is None:
        _simulator_instance = AerSimulator()
    return _simulator_instance

def apply_measurement_basis(quantum_circuit: QuantumCircuit, pauli_string: str) -> Tuple[QuantumCircuit, List[int]]:
    """
    Applies basis change gates IN PLACE to a circuit for measuring a given Pauli string.

    Args:
        quantum_circuit: The QuantumCircuit to modify.
        pauli_string: The Pauli string (e.g., "IXYZ") specifying the measurement basis.

    Returns:
        Tuple[QuantumCircuit, List[int]]: A tuple containing:
            - The modified QuantumCircuit (modified in place).
            - A list of qubit indices that require measurement for this term (non-identity).

    Raises:
        ValueError: If Pauli string length doesn't match circuit qubits or contains invalid characters.
    """
    num_qubits = quantum_circuit.num_qubits
    if len(pauli_string) != num_qubits:
        raise ValueError(f"Pauli string length {len(pauli_string)} mismatches circuit qubits {num_qubits}.")

    measured_qubits_indices = []
    for i, op in enumerate(pauli_string):
        if op == 'X':
            quantum_circuit.h(i)
            measured_qubits_indices.append(i)
        elif op == 'Y':
            # Apply Ry(-pi/2) = Sdg H
            quantum_circuit.sdg(i) # Apply Sdg first
            quantum_circuit.h(i)
            measured_qubits_indices.append(i)
        elif op == 'Z':
             measured_qubits_indices.append(i) # Measure in Z basis (no gate needed for basis change)
        elif op == 'I':
            pass # No basis change, no measurement needed for this qubit for this term
        else:
            raise ValueError(f"Invalid Pauli operator '{op}' in string '{pauli_string}'. Use 'I', 'X', 'Y', 'Z'.")

    return quantum_circuit, sorted(measured_qubits_indices) # Return sorted indices


def run_circuit_and_get_counts(quantum_circuit: QuantumCircuit,
                               param_values: Optional[Union[Sequence[float], Dict[Parameter, float]]] = None,
                               shots: int = 1024) -> Dict[str, int]:
    """
    Assigns parameters (if any), runs the circuit on the simulator, and returns measurement counts.

    Args:
        quantum_circuit: The QuantumCircuit to run (should include measurements).
        param_values: Numerical parameter values. Can be:
            - Sequence (List/np.ndarray): Assigned in the order of sorted `quantum_circuit.parameters`.
            - Dict[Parameter, float]: Mapping Parameter objects to values.
            - None: If the circuit has no parameters.
        shots: Number of simulation shots.

    Returns:
        Dict[str, int]: A dictionary of measurement outcomes (bitstrings) and their counts.
                        Returns an empty dict if shots=0 or no measurements are present.

    Raises:
        ValueError: If the number/type of parameters provided doesn't match the circuit.
        RuntimeError: If simulation or transpilation fails.
    """

    if shots <= 0:
        warnings.warn("run_circuit_and_get_counts called with shots <= 0. Returning empty counts.", UserWarning)
        return {}
    # Delay clbits check until after potential binding

    bound_circuit: QuantumCircuit
    num_circuit_params = quantum_circuit.num_parameters
    param_map: Dict[Parameter, float] = {} # Initialize param_map

    if num_circuit_params > 0:
        if param_values is None:
            raise ValueError(f"Circuit expects {num_circuit_params} parameters, but received None.")

        if isinstance(param_values, dict):
            # Ensure all keys are Parameter objects
            if not all(isinstance(p, Parameter) for p in param_values.keys()):
                 raise TypeError("Parameter dictionary keys must be Qiskit Parameter objects.")
            # Ensure all circuit parameters are present
            circuit_param_set = set(quantum_circuit.parameters)
            provided_param_set = set(param_values.keys())
            if circuit_param_set != provided_param_set:
                 missing = circuit_param_set - provided_param_set
                 extra = provided_param_set - circuit_param_set
                 err_msg = "Parameter dictionary mismatch. "
                 if missing: err_msg += f"Missing: {[p.name for p in sorted(missing, key=lambda p: p.name)]}. " # Sort names
                 if extra: err_msg += f"Extra: {[p.name for p in sorted(extra, key=lambda p: p.name)]}." # Sort names
                 raise ValueError(err_msg)
            # Ensure values are numeric
            try:
                param_map = {p: float(v) for p, v in param_values.items()}
            except (ValueError, TypeError) as e:
                 raise TypeError(f"Parameter dictionary values must be numeric. Error converting: {e}")

        # Check for list or ndarray specifically, EXCLUDING strings implicitly
        elif isinstance(param_values, (list, np.ndarray)):
            if len(param_values) != num_circuit_params:
                 raise ValueError(f"Circuit expects {num_circuit_params} parameters, but received {len(param_values)}.")
            # Sort parameters for consistent binding order
            try:
                # Try sorting numerically by index first (e.g., p_0, p_1, p_10)
                sorted_params = sorted(quantum_circuit.parameters, key=lambda p: int(re.search(r'\d+', p.name).group()) if re.search(r'\d+', p.name) else float('inf'))
            except (AttributeError, IndexError, ValueError, TypeError):
                # Fallback to string sorting if numerical fails or name has no digits
                warnings.warn("Could not sort parameters numerically for binding. Using default name sort.", UserWarning)
                sorted_params = sorted(quantum_circuit.parameters, key=lambda p: p.name)

            # Ensure values are numeric and create map
            try:
                param_map = {p: float(v) for p, v in zip(sorted_params, param_values)}
            except (ValueError, TypeError) as e:
                 raise TypeError(f"Parameter sequence values must be numeric. Error converting: {e}")

        # Catch other unsupported types (including strings, tuples unless explicitly supported)
        else:
             raise TypeError(f"Unsupported type for 'param_values': {type(param_values)}. Use list, np.ndarray, dict, or None.")

        # Bind parameters
        try:
            bound_circuit = quantum_circuit.assign_parameters(param_map)
        except Exception as e:
             # Catch potential errors during binding itself (e.g., type mismatch if Parameter expects specific type)
             raise RuntimeError(f"Unexpected error during parameter assignment: {e}")

    else: # No parameters in circuit
        if param_values is not None and param_values != {} and param_values != []: # Check if non-empty params were provided
             warnings.warn(f"Circuit has no parameters, but received parameters ({type(param_values)}). Ignoring them.", UserWarning)
        bound_circuit = quantum_circuit

    # --- Execution Logic ---
    try:
        # Check for measure instructions *after* binding, as binding might fail first
        has_measurements = any(instruction.operation.name == 'measure' for instruction in bound_circuit.data)
        if not has_measurements:
             # Check if classical bits exist even without measure ops
             if not bound_circuit.clbits:
                  warnings.warn("Circuit contains no classical bits for measurement. Returning empty counts.", UserWarning)
             else:
                  warnings.warn("Circuit submitted for execution contains no measure instructions (but has classical bits). Returning empty counts.", RuntimeWarning)
             return {}

        sim = get_simulator() # Get simulator instance here
        compiled_circuit = transpile(bound_circuit, sim)
        result = sim.run(compiled_circuit, shots=shots).result()
        counts = result.get_counts(compiled_circuit) # Use compiled circuit for get_counts
    except Exception as e:
        raise RuntimeError(f"Error during circuit transpilation or execution: {e}")

    return counts


def calculate_term_expectation(counts: Dict[str, int]) -> float:
    """
    Calculates the expectation value for a Pauli term measurement (Z-basis after transformation)
    based on counts, using parity. Assumes counts correspond to the relevant qubits.

    Args:
        counts: Dictionary of measurement outcomes (bitstrings) and counts.
                Bitstring length corresponds to the number of measured qubits for the term.

    Returns:
        float: The calculated expectation value for the term.
               Returns 0.0 if counts dict is empty or all counts are zero.
    """
    if not counts:
        return 0.0

    expectation_value_sum = 0.0
    total_counts = sum(counts.values())

    if total_counts == 0:
         warnings.warn("Calculating expectation from counts with zero total shots.", RuntimeWarning)
         return 0.0

    for bitstring, count in counts.items():
        parity = bitstring.count('1') % 2
        expectation_value_sum += ((-1)**parity) * count

    return expectation_value_sum / total_counts


def get_hamiltonian_expectation_value(
    ansatz: QuantumCircuit,
    parsed_hamiltonian: List[Tuple[float, str]],
    param_values: Union[Sequence[float], Dict[Parameter, float], None], # Allow None explicitly
    n_shots: int = 1024
) -> float:
    """
    Calculates the total expectation value of a Hamiltonian for a given ansatz and parameters.

    For each Pauli term in the Hamiltonian:
    1. Copies the ansatz.
    2. Binds the parameters.
    3. Applies the appropriate basis change gates.
    4. Adds measurement instructions for relevant qubits.
    5. Runs the circuit and calculates the term's expectation value from counts.
    6. Multiplies by the term's coefficient and sums the results.

    Args:
        ansatz: The (parameterized) ansatz circuit. *Should not contain measurements.*
        parsed_hamiltonian: List of (coefficient, pauli_string) tuples from `parse_hamiltonian_expression`.
        param_values: Numerical parameter values for the ansatz (Sequence, dict or None).
        n_shots: Number of shots for *each* Pauli term measurement circuit.

    Returns:
        float: The total expectation value <H>.

    Raises:
        ValueError: If Pauli string length mismatches ansatz qubits, or parameter issues during binding.
        RuntimeError: If circuit execution fails for any term.
    """
    num_qubits = ansatz.num_qubits
    total_expected_value = 0.0
    bound_ansatz: QuantumCircuit
    num_ansatz_params = ansatz.num_parameters

    param_map: Dict[Parameter, float] = {}
    if num_ansatz_params > 0:
        if param_values is None:
            raise ValueError(f"Ansatz expects {num_ansatz_params} parameters, but received None.")

        if isinstance(param_values, dict):
            if not all(isinstance(p, Parameter) for p in param_values.keys()):
                raise TypeError("Parameter dictionary keys must be Qiskit Parameter objects for ansatz binding.")
            circuit_param_set = set(ansatz.parameters)
            provided_param_set = set(param_values.keys())
            if circuit_param_set != provided_param_set:
                missing = circuit_param_set - provided_param_set
                extra = provided_param_set - circuit_param_set
                err_msg = "Parameter dictionary mismatch for ansatz binding. "
                if missing: err_msg += f"Missing: {[p.name for p in sorted(missing, key=lambda p: p.name)]}. "
                if extra: err_msg += f"Extra: {[p.name for p in sorted(extra, key=lambda p: p.name)]}."
                raise ValueError(err_msg)
            try:
                param_map = {p: float(v) for p, v in param_values.items()}
            except (ValueError, TypeError) as e:
                 raise TypeError(f"Ansatz parameter dictionary values must be numeric. Error converting: {e}")

        elif isinstance(param_values, (list, np.ndarray)):
            if len(param_values) != num_ansatz_params:
                raise ValueError(f"Ansatz expects {num_ansatz_params} parameters, but received sequence of length {len(param_values)}.")
            try:
                sorted_params = sorted(ansatz.parameters, key=lambda p: int(re.search(r'\d+', p.name).group()) if re.search(r'\d+', p.name) else float('inf'))
            except (AttributeError, IndexError, ValueError, TypeError):
                warnings.warn("Could not sort ansatz parameters numerically for binding. Using default name sort.", UserWarning)
                sorted_params = sorted(ansatz.parameters, key=lambda p: p.name)
            try:
                param_map = {p: float(v) for p, v in zip(sorted_params, param_values)}
            except (ValueError, TypeError) as e:
                 raise TypeError(f"Ansatz parameter sequence values must be numeric. Error converting: {e}")
        else:
            raise TypeError(f"Unsupported type for 'param_values' for ansatz binding: {type(param_values)}. Use list, np.ndarray, dict, or None.")

        try:
            bound_ansatz = ansatz.assign_parameters(param_map)
        except Exception as e:
            raise ValueError(f"Failed to bind parameters to ansatz. Error: {e}")
    else: # No parameters in ansatz
        if param_values is not None and param_values != {} and param_values != []:
             warnings.warn(f"Ansatz has no parameters, but received parameters ({type(param_values)}). Ignoring them.", UserWarning)
        bound_ansatz = ansatz


    for coefficient, pauli_string in parsed_hamiltonian:
        if np.isclose(coefficient, 0.0):
            continue # Skip terms with zero coefficient

        if len(pauli_string) != num_qubits:
             raise ValueError(f"Hamiltonian term '{pauli_string}' length {len(pauli_string)} "
                              f"mismatches ansatz qubits {num_qubits}.")

        # --- Build & Run Measurement Circuit for this Term ---
        qc_term = bound_ansatz.copy(name=f"Measure_{pauli_string}")

        # Apply basis transformation gates IN PLACE and get indices to measure
        qc_term, measured_qubit_indices = apply_measurement_basis(qc_term, pauli_string)

        term_exp_val: float
        # If no qubits are measured (Pauli string is all 'I'), expectation is 1.0
        if not measured_qubit_indices:
             term_exp_val = 1.0
        else:
             num_measured = len(measured_qubit_indices)
             cr_name = f"c_{pauli_string.replace('I','_')}" # Create a somewhat unique name
             cr_name = re.sub(r'[^a-zA-Z0-9_]', '', cr_name)
             cr_name = cr_name[:20] # Limit length

             existing_regs = {reg.name for reg in qc_term.cregs}
             reg_suffix = 0
             final_cr_name = cr_name
             while final_cr_name in existing_regs:
                 reg_suffix += 1
                 final_cr_name = f"{cr_name}_{reg_suffix}"

             cr = ClassicalRegister(num_measured, name=final_cr_name)
             qc_term.add_register(cr)
             qc_term.measure(measured_qubit_indices, cr) # Measure to the newly added register

             # Run this specific measurement circuit
             # Parameters are already bound, so pass param_values=None
             counts = run_circuit_and_get_counts(qc_term, param_values=None, shots=n_shots)
             term_exp_val = calculate_term_expectation(counts)

        # Add the weighted term expectation value to the total
        total_expected_value += coefficient * term_exp_val

    return total_expected_value