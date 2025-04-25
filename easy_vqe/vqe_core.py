# =============================================================================
#           easy_vqe: Core VQE Implementation Logic
# =============================================================================
# This module contains the core functions for parsing Hamiltonians,
# building ansatz circuits, calculating expectation values, and running
# the VQE optimization loop using Qiskit and SciPy.
# =============================================================================

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter
from qiskit_aer import AerSimulator
from qiskit import transpile
import matplotlib.pyplot as plt
import re
import warnings
from scipy.optimize import minimize
from typing import Set, List, Tuple, Union, Dict, Optional, Any, Sequence

PARAMETRIC_SINGLE_QUBIT_TARGET: Set[str] = {'rx', 'ry', 'rz', 'p'}
PARAMETRIC_MULTI_QUBIT: Set[str] = {'crx', 'cry', 'crz', 'cp', 'rxx', 'ryy', 'rzz', 'rzx', 'cu1', 'cu3', 'u2', 'u3'}
NON_PARAM_SINGLE: Set[str] = {'h', 's', 't', 'x', 'y', 'z', 'sdg', 'tdg', 'id'}
NON_PARAM_MULTI: Set[str] = {'cx', 'cy', 'cz', 'swap', 'ccx', 'cswap', 'ch'}
MULTI_PARAM_GATES: Set[str] = {'u', 'cu', 'r'} # Gates requiring specific parameter handling


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


def parse_hamiltonian_expression(hamiltonian_string: str) -> List[Tuple[float, str]]:
    """
    Parses a Hamiltonian string expression into a list of (coefficient, pauli_string) tuples.

    Handles explicit coefficients (e.g., "-1.5 * XY"), implicit coefficients (e.g., "+ ZZ", "- YI"),
    various spacing, combinations like "+ -0.5 * ZIZ", and validates the format.

    Args:
        hamiltonian_string: The Hamiltonian expression (e.g., "1.0*XX - 0.5*ZI + YZ").

    Returns:
        List[Tuple[float, str]]: List of (coefficient, pauli_string) tuples.

    Raises:
        ValueError: If the string format is invalid, contains inconsistent Pauli lengths,
                    invalid characters, or invalid numeric coefficients.
        TypeError: If input is not a string.
    """
    if not isinstance(hamiltonian_string, str):
        raise TypeError("Hamiltonian expression must be a string.")

    hamiltonian_string = hamiltonian_string.strip()
    if not hamiltonian_string:
         raise ValueError("Hamiltonian expression cannot be empty.")
    
    # Handles explicit coeffs like "+ -1.5 * XY" or "1.5*XY"
    # Handles implicit coeffs like "- YZ" or "+ XX" or "II" 
    combined_pattern = re.compile(
        # Option 1: Explicit Coefficient Term (handles +/- coeff * Pauli)
        r"([+\-]?\s*(?:(?:\d+\.?\d*|\.?\d+)(?:[eE][+\-]?\d+)?)\s*\*\s*([IXYZ]+))" # Added scientific notation support
        # OR Option 1b: Explicit Coefficient Term (handles coeff * Pauli, assumes positive) - needed if first term has no sign
        r"|(\s*(?:(?:\d+\.?\d*|\.?\d+)(?:[eE][+\-]?\d+)?)\s*\*\s*([IXYZ]+))"
        # OR Option 2: Implicit Coefficient Term (handles +/- Pauli)
        r"|(([+\-])\s*([IXYZ]+))"
        # OR Option 2b: Implicit Coefficient Term (handles Pauli at start, assumes positive)
        r"|^([IXYZ]+)" # Use ^ to anchor only at the beginning
    )

    parsed_terms = []
    current_pos = 0
    ham_len = len(hamiltonian_string)

    while current_pos < ham_len:
        match_start_search = re.search(r'\S', hamiltonian_string[current_pos:])
        if not match_start_search:
            break 
        search_pos = current_pos + match_start_search.start()

        match = combined_pattern.match(hamiltonian_string, search_pos)

        if not match:
            remaining_str = hamiltonian_string[search_pos:]
            if remaining_str.startswith('*'):
                 raise ValueError(f"Syntax error near position {search_pos}: Unexpected '*' without preceding coefficient.")
            raise ValueError(f"Could not parse term starting near position {search_pos}: '{hamiltonian_string[search_pos:min(search_pos+20, ham_len)]}...'. Check syntax (e.g., signs, '*', Pauli chars [IXYZ]).")

        coefficient: float = 1.0
        pauli_str: Optional[str] = None
        term_str = match.group(0).strip() 

        if match.group(1): # Option 1 (+/- coeff * Pauli)
            # Extract coeff carefully, handling potential combined sign like "+-1.5"
            num_part = re.match(r"([+\-]?)\s*(.*)", match.group(1).split('*')[0].strip())
            sign_str = num_part.group(1)
            coeff_val_str = num_part.group(2)
            try:
                coefficient = float(coeff_val_str)
                if sign_str == '-': coefficient *= -1.0
            except ValueError: raise ValueError(f"Invalid numeric coefficient '{match.group(1).split('*')[0].strip()}' in term '{term_str}'.")
            pauli_str = match.group(2)
        elif match.group(3): # Option 1b (coeff * Pauli, positive implicit sign)
            try: coefficient = float(match.group(3).split('*')[0].strip())
            except ValueError: raise ValueError(f"Invalid numeric coefficient '{match.group(3).split('*')[0].strip()}' in term '{term_str}'.")
            pauli_str = match.group(4)
        elif match.group(5): # Option 2 (+/- Pauli)
            sign = match.group(6)
            coefficient = -1.0 if sign == '-' else 1.0
            pauli_str = match.group(7)
        elif match.group(8): # Option 2b (Pauli at start, positive implicit sign)
             if search_pos != 0: 
                   raise ValueError(f"Ambiguous term '{match.group(8)}' at pos {search_pos}. Terms after the first need explicit '+', '-', or 'coeff *'.")
             coefficient = 1.0
             pauli_str = match.group(8)
        else:
             raise RuntimeError(f"Internal parsing error: Regex match failed unexpectedly near '{hamiltonian_string[search_pos:search_pos+10]}...'")


        if pauli_str is None: # This should not happen due to regex structure
             raise ValueError(f"Failed to extract Pauli string from parsed term '{term_str}'.")
        if not pauli_str: # Pauli string cannot be empty
             raise ValueError(f"Empty Pauli string found in term '{term_str}'.")
        if not all(c in 'IXYZ' for c in pauli_str):
              raise ValueError(f"Invalid character found in Pauli string '{pauli_str}' within term '{term_str}'. Only 'I', 'X', 'Y', 'Z' allowed.")
        if np.isnan(coefficient) or np.isinf(coefficient): # Check for NaN/Infinity coefficients
              raise ValueError(f"Invalid coefficient value ({coefficient}) found for term '{pauli_str}'.")

        parsed_terms.append((coefficient, pauli_str))
        current_pos = match.end()

    if not parsed_terms:
        # This should only happen if the input string was non-empty but contained no parsable terms
        raise ValueError(f"Could not parse any valid Hamiltonian terms from the input string: '{hamiltonian_string}'.")

    # Check for consistent Pauli string lengths across all terms
    num_qubits = len(parsed_terms[0][1])
    if num_qubits == 0: raise ValueError("Parsed Pauli string has zero length (Internal Error).") # Should be caught by [IXYZ]+
    for i, (coeff, p_str) in enumerate(parsed_terms):
        if len(p_str) != num_qubits:
            raise ValueError(f"Inconsistent Pauli string lengths found: Term 0 '{parsed_terms[0][1]}' (len {num_qubits}) vs Term {i} '{p_str}' (len {len(p_str)}). All terms must act on the same number of qubits.")

    return parsed_terms


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
    sim = get_simulator()

    if shots <= 0:
        warnings.warn("run_circuit_and_get_counts called with shots <= 0. Returning empty counts.", UserWarning)
        return {}
    if not quantum_circuit.clbits:
         warnings.warn("Circuit contains no classical bits for measurement. Returning empty counts.", UserWarning)
         return {}

    bound_circuit: QuantumCircuit
    num_circuit_params = quantum_circuit.num_parameters

    if num_circuit_params > 0:
        if param_values is None:
            raise ValueError(f"Circuit expects {num_circuit_params} parameters, but received None.")

        param_map: Dict[Parameter, float]
        if isinstance(param_values, dict):
            # Ensure all circuit parameters are present in the dict keys
            circuit_param_set = set(quantum_circuit.parameters)
            provided_param_set = set(param_values.keys())
            if circuit_param_set != provided_param_set:
                 missing = circuit_param_set - provided_param_set
                 extra = provided_param_set - circuit_param_set
                 err_msg = "Parameter dictionary mismatch. "
                 if missing: err_msg += f"Missing: {[p.name for p in missing]}. "
                 if extra: err_msg += f"Extra: {[p.name for p in extra]}."
                 raise ValueError(err_msg)
            param_map = param_values
        elif isinstance(param_values, (list, np.ndarray, Sequence)): 
            if len(param_values) != num_circuit_params:
                 raise ValueError(f"Circuit expects {num_circuit_params} parameters, but received {len(param_values)}.")
            try:
                sorted_params = sorted(quantum_circuit.parameters, key=lambda p: int(p.name.split('_')[1]))
            except (IndexError, ValueError):
                warnings.warn("Could not sort parameters numerically for binding. Using default name sort.", UserWarning)
                sorted_params = sorted(quantum_circuit.parameters, key=lambda p: p.name)
            param_map = {p: float(v) for p, v in zip(sorted_params, param_values)} 
        else:
             raise TypeError(f"Unsupported type for 'param_values': {type(param_values)}. Use Sequence (list/array), dict, or None.")

        try:
            bound_circuit = quantum_circuit.assign_parameters(param_map)
        except TypeError as e:
             raise ValueError(f"Failed to assign parameters. Check parameter types. Error: {e}")
        except Exception as e:
             raise RuntimeError(f"Unexpected error during parameter assignment: {e}")

    else: # No parameters in circuit
        if param_values is not None and len(param_values) > 0:
             # Check if param_values is just an empty list/dict, which is okay
             if isinstance(param_values, (dict, Sequence)) and not param_values:
                  pass # Empty container is fine
             else:
                  warnings.warn(f"Circuit has no parameters, but received parameters ({type(param_values)}). Ignoring them.", UserWarning)
        bound_circuit = quantum_circuit

    try:
        has_measurements = any(instruction.operation.name == 'measure' for instruction in bound_circuit.data)
        if not has_measurements:
             warnings.warn("Circuit submitted for execution contains no measure instructions. Returning empty counts.", RuntimeWarning)
             return {}

        compiled_circuit = transpile(bound_circuit, sim)
        result = sim.run(compiled_circuit, shots=shots).result()
        counts = result.get_counts(compiled_circuit)
    except Exception as e:
        # Catch potential Aer errors, transpilation issues, or other Qiskit errors
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
         # This might happen if shots=0 or simulation failed, handled earlier usually
         warnings.warn("Calculating expectation from counts with zero total shots.", RuntimeWarning)
         return 0.0

    for bitstring, count in counts.items():
        parity = bitstring.count('1') % 2
        expectation_value_sum += ((-1)**parity) * count

    # Normalize by the total number of shots that yielded results
    return expectation_value_sum / total_counts


def get_hamiltonian_expectation_value(
    ansatz: QuantumCircuit,
    parsed_hamiltonian: List[Tuple[float, str]],
    param_values: Union[Sequence[float], Dict[Parameter, float]],
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
        param_values: Numerical parameter values for the ansatz (Sequence or dict).
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
    if ansatz.num_parameters > 0:
        # Use the same binding logic as run_circuit_and_get_counts for consistency
        if isinstance(param_values, dict):
            if set(ansatz.parameters) != set(param_values.keys()): raise ValueError("Param dict keys mismatch ansatz.")
            param_map = param_values
        elif isinstance(param_values, (list, np.ndarray, Sequence)):
            if len(param_values) != ansatz.num_parameters: raise ValueError("Param sequence length mismatch ansatz.")
            try: sorted_params = sorted(ansatz.parameters, key=lambda p: int(p.name.split('_')[1]))
            except (IndexError, ValueError): sorted_params = sorted(ansatz.parameters, key=lambda p: p.name)
            param_map = {p: float(v) for p, v in zip(sorted_params, param_values)}
        else:
             raise TypeError(f"Unsupported type for 'param_values': {type(param_values)}")
        try:
            bound_ansatz = ansatz.assign_parameters(param_map)
        except Exception as e:
            raise ValueError(f"Failed to bind parameters to ansatz. Error: {e}")
    else: # No parameters
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
             cr = ClassicalRegister(num_measured, name="c")
             qc_term.add_register(cr)

            # Add measurement instructions for the qubits that are not 'I'
             qc_term.measure(measured_qubit_indices, cr)

             # Run this specific measurement circuit
             # param_values=None because parameters are already bound in qc_term
             counts = run_circuit_and_get_counts(qc_term, param_values=None, shots=n_shots)

             # Calculate expectation value for this term using parity from counts
             term_exp_val = calculate_term_expectation(counts)

        # Add the weighted term expectation value to the total
        total_expected_value += coefficient * term_exp_val

    return total_expected_value


class OptimizationLogger:
    """Helper class to store optimization history during scipy.minimize."""
    def __init__(self):
        self.eval_count = 0
        self.params_history: List[np.ndarray] = []
        self.value_history: List[float] = []
        self._last_print_eval = 0

    def callback(self, current_params: np.ndarray, current_value: float, display_progress: bool = False, print_interval: int = 10):
        """Stores current parameters and value, optionally prints progress."""
        self.eval_count += 1
        self.params_history.append(np.copy(current_params)) 
        self.value_history.append(current_value)

        if display_progress and (self.eval_count - self._last_print_eval >= print_interval):
             print(f"  Eval: {self.eval_count:4d} | Energy: {current_value: .8f}")
             self._last_print_eval = self.eval_count

    def get_history(self) -> Tuple[List[float], List[np.ndarray]]:
        """Returns the recorded history."""
        return self.value_history, self.params_history


def find_ground_state(
    ansatz_structure: List[Union[Tuple[str, List[int]], List]],
    hamiltonian_expression: str,
    n_shots: int = 2048,
    optimizer_method: str = 'COBYLA',
    optimizer_options: Optional[Dict[str, Any]] = None,
    initial_params_strategy: Union[str, np.ndarray, Sequence[float]] = 'random',
    max_evaluations: Optional[int] = 150, 
    display_progress: bool = True,
    plot_filename: Optional[str] = None 
) -> Dict[str, Any]:
    """
    Performs the Variational Quantum Eigensolver (VQE) algorithm to find the
    approximate ground state energy of a given Hamiltonian using simulation.

    Args:
        ansatz_structure: Definition for `create_custom_ansatz`.
        hamiltonian_expression: Hamiltonian string (e.g., "-1.0*ZZ + 0.5*X").
        n_shots: Number of shots per expectation value estimation. Higher values
                 reduce noise but increase simulation time.
        optimizer_method: Name of the SciPy optimizer to use (e.g., 'COBYLA',
                          'Nelder-Mead', 'L-BFGS-B', 'Powell', 'SLSQP').
        optimizer_options: Dictionary of options passed directly to the SciPy
                           optimizer (e.g., {'maxiter': 200, 'tol': 1e-6}).
                           Overrides `max_evaluations` if relevant keys exist.
        initial_params_strategy: Method for generating initial parameters:
            - 'random': Uniformly random values in [0, 2*pi).
            - 'zeros': All parameters initialized to 0.0.
            - np.ndarray or Sequence: A specific array/list of initial values.
        max_evaluations: Approximate maximum number of objective function calls
                         (used to set 'maxiter' or 'maxfun'/'maxfev' in options
                         if not already specified).
        display_progress: If True, prints energy updates during optimization.
        plot_filename: If a filename string is provided (e.g., "convergence.png"),
                       saves the energy convergence plot to that file. If None,
                       no plot is saved.

    Returns:
        Dict[str, Any]: A dictionary containing VQE results:
            - 'optimal_params' (np.ndarray): Best parameters found.
            - 'optimal_value' (float): Minimum expectation value (energy) found.
            - 'num_qubits' (int): Number of qubits determined from Hamiltonian.
            - 'ansatz' (QuantumCircuit): The constructed ansatz circuit.
            - 'parameters' (List[Parameter]): Parameter objects in the ansatz.
            - 'optimization_result' (OptimizeResult): Full result from `scipy.optimize.minimize`.
            - 'cost_history' (List[float]): Energy values at each evaluation.
            - 'parameter_history' (List[np.ndarray]): Parameter vectors at each evaluation.
            - 'success' (bool): Optimizer success flag.
            - 'message' (str): Optimizer termination message.
            - 'n_shots' (int): Shots used per evaluation.
            - 'optimizer_method' (str): Optimizer used.
            - 'hamiltonian_expression' (str): Original Hamiltonian string.
            - 'plot_filename' (Optional[str]): Filename if plot was saved.
        Returns {'error': ..., 'details': ...} dictionary on critical failure during setup.
    """
    print("-" * 50)
    print("           Easy VQE - Ground State Search")
    print("-" * 50)
    print(f"Hamiltonian: {hamiltonian_expression}")
    print(f"Optimizer: {optimizer_method} | Shots per Eval: {n_shots}")

    result_dict: Dict[str, Any] = { 
        'hamiltonian_expression': hamiltonian_expression,
        'optimizer_method': optimizer_method,
        'n_shots': n_shots,
        'plot_filename': plot_filename,
    }

    try:
        parsed_hamiltonian = parse_hamiltonian_expression(hamiltonian_expression)
        if not parsed_hamiltonian: 
             print("[Error] Hamiltonian expression parsed successfully but resulted in zero terms.")
             return {'error': 'Hamiltonian parsing resulted in zero terms'}
        num_qubits = len(parsed_hamiltonian[0][1])
        result_dict['num_qubits'] = num_qubits
        print(f"Parsed Hamiltonian: {len(parsed_hamiltonian)} terms | Qubits: {num_qubits}")
    except (ValueError, TypeError) as e:
        print(f"\n[Error] Failed to parse Hamiltonian: {e}")
        return {'error': 'Hamiltonian parsing failed', 'details': str(e)}

    try:
        ansatz, parameters = create_custom_ansatz(num_qubits, ansatz_structure)
        num_params = len(parameters)
        result_dict.update({'ansatz': ansatz, 'parameters': parameters})
        print(f"Created Ansatz: {num_params} parameters")

        if num_params == 0:
            warnings.warn("Ansatz has no parameters. Calculating fixed expectation value.", UserWarning)
            try:
                fixed_value = get_hamiltonian_expectation_value(ansatz, parsed_hamiltonian, [], n_shots)
                print(f"Fixed Expectation Value: {fixed_value:.8f}")
                result_dict.update({
                    'optimal_params': np.array([]), 'optimal_value': fixed_value,
                    'optimization_result': None, 'cost_history': [fixed_value],
                    'parameter_history': [np.array([])], 'success': True,
                    'message': 'Static evaluation (no parameters)'
                 })
                return result_dict
            except Exception as e:
                print(f"\n[Error] Failed to calculate fixed expectation value: {e}")
                return {'error': 'Failed static evaluation', 'details': str(e), **result_dict}

    except (ValueError, TypeError, RuntimeError, IndexError) as e:
        print(f"\n[Error] Failed to create Ansatz: {e}")
        # Include num_qubits if determined before failure
        return {'error': 'Ansatz creation failed', 'details': str(e), **result_dict}

    logger = OptimizationLogger()

    def objective_function(current_params: np.ndarray) -> float:
        """Closure for the optimizer, calculates Hamiltonian expectation value."""
        try:
            exp_val = get_hamiltonian_expectation_value(
                ansatz=ansatz,
                parsed_hamiltonian=parsed_hamiltonian,
                param_values=current_params,
                n_shots=n_shots
            )
            logger.callback(current_params, exp_val, display_progress=display_progress)
            return exp_val
        except (ValueError, RuntimeError, TypeError) as e:
             print(f"\n[Warning] Error during expectation value calculation (params={np.round(current_params[:4], 3)}...): {e}")
             return np.inf
        except Exception as e:
             print(f"\n[Critical Warning] Unexpected error in objective function: {e}")
             return np.inf

    initial_params: np.ndarray
    strategy_check_value = initial_params_strategy

    print("\nProcessing Initial Parameters Strategy...") 

    if isinstance(strategy_check_value, np.ndarray):
        if strategy_check_value.shape == (num_params,):
            initial_params = strategy_check_value.astype(float)
            print(f"Strategy: Using provided numpy array (shape {initial_params.shape}) for initial parameters.")
        else:
            print(f"[Warning] Provided initial_params numpy array shape {strategy_check_value.shape} != expected ({num_params},). Defaulting to 'random'.")
            initial_params = np.random.uniform(0, 2 * np.pi, num_params)
            print(f"Strategy: Using 'random' (generated {num_params} parameters).")
            strategy_check_value = 'random' 

    elif isinstance(strategy_check_value, (list, tuple)):
         if len(strategy_check_value) == num_params:
             try:
                 initial_params = np.array(strategy_check_value, dtype=float)
                 print(f"Strategy: Using provided list/tuple (length {len(strategy_check_value)}) for initial parameters.")
             except ValueError as ve:
                 print(f"[Warning] Could not convert provided list/tuple to numeric array: {ve}. Defaulting to 'random'.")
                 initial_params = np.random.uniform(0, 2 * np.pi, num_params)
                 print(f"Strategy: Using 'random' (generated {num_params} parameters).")
                 strategy_check_value = 'random'
         else:
            print(f"[Warning] Provided initial_params list/tuple length {len(strategy_check_value)} != expected {num_params}. Defaulting to 'random'.")
            initial_params = np.random.uniform(0, 2 * np.pi, num_params)
            print(f"Strategy: Using 'random' (generated {num_params} parameters).")
            strategy_check_value = 'random'

    elif strategy_check_value == 'zeros':
         initial_params = np.zeros(num_params)
         print(f"Strategy: Using 'zeros' (generated {num_params} parameters).")
    elif strategy_check_value == 'random':
         initial_params = np.random.uniform(0, 2 * np.pi, num_params)
         print(f"Strategy: Using 'random' (generated {num_params} parameters).")
    else:
         print(f"[Warning] Unknown initial_params_strategy '{strategy_check_value}'. Defaulting to 'random'.")
         initial_params = np.random.uniform(0, 2 * np.pi, num_params)
         print(f"Strategy: Using 'random' (generated {num_params} parameters).")
         strategy_check_value = 'random'

    result_dict['initial_params'] = np.copy(initial_params)
    result_dict['initial_params_strategy_used'] = strategy_check_value 

    opt_options = optimizer_options if optimizer_options is not None else {}
    if 'maxiter' not in opt_options and 'maxfev' not in opt_options and max_evaluations is not None:
         if optimizer_method.upper() in ['COBYLA', 'NELDER-MEAD', 'POWELL']:
             opt_options['maxiter'] = int(max_evaluations) # These often use maxiter
             print(f"Setting optimizer 'maxiter' to {opt_options['maxiter']}")
         elif optimizer_method.upper() in ['L-BFGS-B', 'SLSQP', 'TNC']:
              opt_options['maxfun'] = int(max_evaluations) # Others might use maxfun/maxfev
              print(f"Setting optimizer 'maxfun' to {opt_options['maxfun']}")

    print(f"\nStarting Optimization with {optimizer_method}...")
    print(f"Initial Parameters (first 5): {np.round(initial_params[:5], 5)}")

    initial_energy = objective_function(initial_params)
    if np.isinf(initial_energy):
        print("[Error] Objective function returned infinity for initial parameters. Cannot start optimization.")
        return {'error': 'Initial parameters yield invalid energy (inf).', 'details': 'Check ansatz or Hamiltonian.', **result_dict}
    print(f"Initial Energy: {initial_energy:.8f}")

    try:
        result = minimize(objective_function,
                          initial_params,
                          method=optimizer_method,
                          options=opt_options)

    except Exception as e:
        print(f"\n[Error] Optimization process failed: {e}")

        cost_history, param_history = logger.get_history()
        result_dict.update({
            'error': 'Optimization process failed', 'details': str(e),
            'cost_history': cost_history, 'parameter_history': param_history,
            'optimization_result': None, 'success': False,
            'message': f'Optimization terminated due to error: {e}'
        })
        return result_dict

    print("\n" + "-"*20 + " Optimization Finished " + "-"*20)
    cost_history, param_history = logger.get_history() 
    result_dict.update({
        'optimal_params': result.x,
        'optimal_value': result.fun,
        'optimization_result': result,
        'cost_history': cost_history,
        'parameter_history': param_history,
        'success': result.success,
        'message': result.message,
    })

    if result.success:
        print("Optimizer terminated successfully.")
    else:
        print(f"[Warning] Optimizer terminated unsuccessfully: {result.message}")

    if hasattr(result, 'nfev'): print(f"Function Evaluations: {result.nfev}")
    if hasattr(result, 'nit'): print(f"Iterations: {result.nit}") 

    print(f"Optimal Energy Found: {result_dict['optimal_value']:.10f}")
    opt_params = result_dict['optimal_params']
    if len(opt_params) < 15:
         print(f"Optimal Parameters:\n{np.round(opt_params, 5)}")
    else:
         print(f"Optimal Parameters: (Array length {len(opt_params)})")
         print(f"  First 5: {np.round(opt_params[:5], 5)}")
         print(f"  Last 5:  {np.round(opt_params[-5:], 5)}")
    print("-" * 50)

    if plot_filename and cost_history:
        try:
            fig, ax = plt.subplots(figsize=(10, 6)) 
            ax.plot(range(len(cost_history)), cost_history, marker='.', linestyle='-', markersize=4)
            ax.set_xlabel("Optimization Evaluation Step")
            ax.set_ylabel("Hamiltonian Expectation Value (Energy)")
            ax.set_title(f"VQE Convergence ({optimizer_method}, {n_shots} shots)")
            ax.grid(True, linestyle='--', alpha=0.6)
            fig.tight_layout()
            fig.savefig(plot_filename) # Save the plot to the specified file
            plt.close(fig) 
            print(f"Convergence plot saved to '{plot_filename}'")
            result_dict['plot_filename'] = plot_filename # Confirm saved filename
        except Exception as e:
            print(f"[Warning] Could not save convergence plot to '{plot_filename}': {e}")
            result_dict['plot_filename'] = None

    return result_dict

def get_theoretical_ground_state_energy(hamiltonian_expression: str) -> float:
    """
    Calculates the theoretical ground state energy of a Hamiltonian.

    Args:
        hamiltonian_expression: Hamiltonian string (e.g., "-1.0*ZZ + 0.5*X").

    Returns:
        float: The theoretical ground state energy.

    Raises:
        ValueError: If the Hamiltonian expression is invalid
    """
    pauli_i = np.array([[1, 0], [0, 1]], dtype=complex)
    pauli_x = np.array([[0, 1], [1, 0]], dtype=complex)
    pauli_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    pauli_z = np.array([[1, 0], [0, -1]], dtype=complex)
    pauli_map = {'I': pauli_i, 'X': pauli_x, 'Y': pauli_y, 'Z': pauli_z}

    parsed_ham = parse_hamiltonian_expression(hamiltonian_expression) 

    num_qubits = len(parsed_ham[0][1])
    dim = 2**num_qubits
    ham_matrix = np.zeros((dim, dim), dtype=complex)

    for coeff, pauli_str in parsed_ham:
        term_matrix = np.array([1], dtype=complex) 
        for pauli_char in pauli_str:
            term_matrix = np.kron(term_matrix, pauli_map[pauli_char])
        ham_matrix += coeff * term_matrix

    eigenvalues = np.linalg.eigvalsh(ham_matrix)
    ground_state_energy_exact = np.min(eigenvalues)

    return ground_state_energy_exact.real


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