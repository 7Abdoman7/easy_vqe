import warnings
import re
from typing import Set, List, Tuple, Union, Dict
from qiskit import QuantumCircuit, ClassicalRegister
from qiskit.circuit import Parameter

# Gate type categorization
PARAMETRIC_SINGLE_QUBIT_TARGET: Set[str] = {'rx', 'ry', 'rz', 'p', 'u1'}
PARAMETRIC_MULTI_QUBIT: Set[str] = {} # User emptied
NON_PARAM_SINGLE: Set[str] = {'h', 's', 't', 'x', 'y', 'z', 'sdg', 'tdg', 'id'}
NON_PARAM_MULTI: Set[str] = {'cx', 'cy', 'cz', 'swap', 'ccx', 'cswap', 'ch'}
MULTI_PARAM_GATES: Set[str] = {} # User emptied

# Special string instructions
LAYER_INSTRUCTIONS: Set[str] = {"rx_layer", "ry_layer", "rz_layer"}
BARRIER_INSTRUCTION: str = "barrier"
ENTANGLEMENT_INSTRUCTIONS: Set[str] = {
    "linear_entanglement",
    "circular_entanglement",
    "full_entanglement"
}

# --- Entanglement Helper Functions ---
def _create_linear_entanglement_ops(num_qubits: int) -> List[Tuple[str, List[int]]]:
    ops = []
    if num_qubits >= 2:
        for i in range(num_qubits - 1):
            ops.append(('cx', [i, i + 1]))
    return ops

def _create_circular_entanglement_ops(num_qubits: int) -> List[Tuple[str, List[int]]]:
    ops = []
    if num_qubits >= 2:
        for i in range(num_qubits - 1):
            ops.append(('cx', [i, i + 1]))
        ops.append(('cx', [num_qubits - 1, 0])) # Connect last to first
    elif num_qubits == 1: # For 1 qubit, circular is trivial (no ops) but doesn't hurt
        pass
    return ops

def _create_full_entanglement_ops(num_qubits: int) -> List[Tuple[str, List[int]]]:
    ops = []
    if num_qubits >= 2:
        for i in range(num_qubits):
            for j in range(i + 1, num_qubits):
                ops.append(('cx', [i, j]))
    return ops

# --- Layer Creation Functions (unchanged) ---
def create_rotation_layer(num_qubits: int, rotation_type: str) -> tuple:
    if rotation_type not in {'rx', 'ry', 'rz'}:
        raise ValueError(f"Invalid rotation type '{rotation_type}'. Must be one of 'rx', 'ry', or 'rz'.")
    return (rotation_type, list(range(num_qubits)))

def create_rx_layer(num_qubits: int) -> tuple:
    return create_rotation_layer(num_qubits, 'rx')

def create_ry_layer(num_qubits: int) -> tuple:
    return create_rotation_layer(num_qubits, 'ry')

def create_rz_layer(num_qubits: int) -> tuple:
    return create_rotation_layer(num_qubits, 'rz')


def create_custom_ansatz(num_qubits: int, ansatz_structure: List[Union[str, Tuple[str, List[int]], List]]) -> Tuple[QuantumCircuit, List[Parameter]]:
    """
    Creates a parameterized quantum circuit (ansatz) from a simplified structure.

    Automatically generates unique Parameter objects (p_0, p_1, ...) for
    parametric gates based on their order of appearance.

    Args:
        num_qubits: The number of qubits for the circuit.
        ansatz_structure: A list defining the circuit structure. Elements can be:
            - str: A special instruction:
                - Layer instructions: "rx_layer", "ry_layer", "rz_layer" (apply to all qubits).
                - "barrier" (applies to all qubits).
                - Entanglement instructions: "linear_entanglement", "circular_entanglement",
                  "full_entanglement" (apply CNOTs based on the pattern).
            - Tuple[str, List[int]]: (gate_name, target_qubit_indices).
              If `gate_name` is a layer instruction, `target_qubit_indices` is ignored.
              If `gate_name` is "barrier" and `target_qubit_indices` is empty,
              the barrier applies to all qubits.
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

    # _process_instruction remains the same as in your previous version
    def _process_instruction(instruction_from_structure: Tuple[str, List[int]],
                             current_ansatz: QuantumCircuit,
                             params_dict: Dict[str, Parameter],
                             p_idx_ref: List[int]):
        if not isinstance(instruction_from_structure, tuple) or len(instruction_from_structure) != 2:
            raise TypeError(f"Internal Error: _process_instruction expects a tuple. Got: {instruction_from_structure}")

        gate_name_in_struct, qubit_indices_in_struct = instruction_from_structure

        if not isinstance(gate_name_in_struct, str) or not isinstance(qubit_indices_in_struct, list):
             raise TypeError(f"Internal Error: Instruction tuple must contain (str, list). Got: ({type(gate_name_in_struct)}, {type(qubit_indices_in_struct)})")

        processed_gate_name = gate_name_in_struct.lower()
        processed_qubit_indices = list(qubit_indices_in_struct)

        original_instruction_was_layer_type = False
        is_barrier_instruction = False

        if processed_gate_name in LAYER_INSTRUCTIONS:
            original_instruction_was_layer_type = True
            if qubit_indices_in_struct:
                warnings.warn(
                    f"For layer instruction tuple ('{gate_name_in_struct}', {qubit_indices_in_struct}), "
                    f"the provided qubit list is ignored. "
                    f"The layer will be applied to all {current_ansatz.num_qubits} qubits.",
                    UserWarning
                )
            if processed_gate_name == "rx_layer":
                processed_gate_name, processed_qubit_indices = create_rx_layer(current_ansatz.num_qubits)
            elif processed_gate_name == "ry_layer":
                processed_gate_name, processed_qubit_indices = create_ry_layer(current_ansatz.num_qubits)
            elif processed_gate_name == "rz_layer":
                processed_gate_name, processed_qubit_indices = create_rz_layer(current_ansatz.num_qubits)
        elif processed_gate_name == BARRIER_INSTRUCTION:
            is_barrier_instruction = True

        if processed_gate_name == BARRIER_INSTRUCTION and not processed_qubit_indices:
             processed_qubit_indices = list(range(current_ansatz.num_qubits))
        elif not processed_qubit_indices and processed_gate_name != BARRIER_INSTRUCTION:
            warnings.warn(f"Gate '{processed_gate_name}' specified with empty qubit list. Skipping.", UserWarning)
            return

        for q_idx_val, q_val in enumerate(processed_qubit_indices):
             if not isinstance(q_val, int) or q_val < 0:
                  raise ValueError(f"Invalid qubit index '{q_val}' (at position {q_idx_val}) in {processed_qubit_indices} for gate '{processed_gate_name}'. Indices must be non-negative integers.")
             if q_val >= current_ansatz.num_qubits:
                  raise ValueError(f"Qubit index {q_val} (at position {q_idx_val}) in {processed_qubit_indices} for gate '{processed_gate_name}' is out of bounds. "
                               f"Circuit has {current_ansatz.num_qubits} qubits (indices 0 to {current_ansatz.num_qubits - 1}).")

        current_gate_to_apply = processed_gate_name
        gate_method = None

        if current_gate_to_apply == 'cnot': current_gate_to_apply = 'cx'
        elif current_gate_to_apply == 'toffoli': current_gate_to_apply = 'ccx'
        elif current_gate_to_apply == 'meas': current_gate_to_apply = 'measure'
        elif current_gate_to_apply == 'u1': current_gate_to_apply = 'p'

        if hasattr(current_ansatz, current_gate_to_apply):
             gate_method = getattr(current_ansatz, current_gate_to_apply)
        else:
             error_msg_gate_part = f"Gate '{current_gate_to_apply}'"
             if original_instruction_was_layer_type or (is_barrier_instruction and gate_name_in_struct.lower() == BARRIER_INSTRUCTION):
                 error_msg_gate_part += f" (derived from '{gate_name_in_struct}')"
             raise ValueError(f"{error_msg_gate_part} is not a valid method of QuantumCircuit (or a known alias).")

        try:
            error_context_gate_part = f"Gate '{current_gate_to_apply}'"
            if original_instruction_was_layer_type or (is_barrier_instruction and gate_name_in_struct.lower() == BARRIER_INSTRUCTION):
                error_context_gate_part += f" (derived from '{gate_name_in_struct}')"

            if current_gate_to_apply in MULTI_PARAM_GATES:
                 raise ValueError(f"{error_context_gate_part} requires multiple parameters which are not auto-generated "
                                  "by this simple format. Construct this gate explicitly if needed.")

            if current_gate_to_apply in PARAMETRIC_SINGLE_QUBIT_TARGET:
                for q_idx_single in processed_qubit_indices:
                    param_name = f"p_{p_idx_ref[0]}"
                    p_idx_ref[0] += 1
                    if param_name not in params_dict:
                         params_dict[param_name] = Parameter(param_name)
                    gate_method(params_dict[param_name], q_idx_single)
            elif current_gate_to_apply in NON_PARAM_SINGLE:
                for q_idx_single in processed_qubit_indices:
                    gate_method(q_idx_single)
            elif current_gate_to_apply in PARAMETRIC_MULTI_QUBIT:
                 param_name = f"p_{p_idx_ref[0]}"
                 p_idx_ref[0] += 1
                 if param_name not in params_dict:
                      params_dict[param_name] = Parameter(param_name)
                 gate_method(params_dict[param_name], *processed_qubit_indices)
            elif current_gate_to_apply in NON_PARAM_MULTI:
                 gate_method(*processed_qubit_indices)
            elif current_gate_to_apply == BARRIER_INSTRUCTION:
                 gate_method(processed_qubit_indices)
            elif current_gate_to_apply == 'measure':
                 warnings.warn("Explicit 'measure' instruction found in ansatz structure. "
                               "Measurements are typically added separately based on Hamiltonian terms.", UserWarning)
                 creg_name = 'meas_reg'
                 target_creg = None
                 if current_ansatz.cregs:
                     for reg in current_ansatz.cregs:
                         if len(reg) == len(processed_qubit_indices):
                              target_creg = reg
                              break
                     if target_creg is None:
                          reg_suffix = 0
                          while f"{creg_name}{reg_suffix}" in [r.name for r in current_ansatz.cregs]:
                              reg_suffix += 1
                          target_creg = ClassicalRegister(len(processed_qubit_indices), name=f"{creg_name}{reg_suffix}")
                          current_ansatz.add_register(target_creg)
                          warnings.warn(f"Auto-added ClassicalRegister({len(processed_qubit_indices)}) named '{target_creg.name}' for measure.", UserWarning)
                 else:
                     target_creg = ClassicalRegister(len(processed_qubit_indices), name=creg_name)
                     current_ansatz.add_register(target_creg)
                     warnings.warn(f"Auto-added ClassicalRegister({len(processed_qubit_indices)}) named '{target_creg.name}' for measure.", UserWarning)
                 try:
                      current_ansatz.measure(processed_qubit_indices, target_creg)
                 except Exception as me:
                      raise RuntimeError(f"Failed to apply 'measure' to qubits {processed_qubit_indices} and register {target_creg.name}. Error: {me}")
            else:
                 if len(processed_qubit_indices) == 1:
                     warnings.warn(f"Gate '{current_gate_to_apply}' is uncategorized but applying to single qubit {processed_qubit_indices[0]}. Parameterization may be incorrect.", UserWarning)
                     gate_method(processed_qubit_indices[0])
                 elif len(processed_qubit_indices) > 1:
                     warnings.warn(f"Gate '{current_gate_to_apply}' is uncategorized but applying to multiple qubits {processed_qubit_indices}. Parameterization may be incorrect.", UserWarning)
                     gate_method(*processed_qubit_indices)
                 else:
                     raise RuntimeError(f"Internal Error: {error_context_gate_part} passed initial checks but was not categorized and has no qubits.")
        except TypeError as e:
             num_expected_qubits = 'unknown'
             if current_gate_to_apply in PARAMETRIC_SINGLE_QUBIT_TARGET or current_gate_to_apply in NON_PARAM_SINGLE: num_expected_qubits = 1
             elif current_gate_to_apply in {'cx','cz','cy','swap','cp','crx','cry','crz','rxx','ryy','rzz','rzx','cu1'}: num_expected_qubits = 2
             elif current_gate_to_apply in {'ccx', 'cswap'}: num_expected_qubits = 3
             raise ValueError(
                 f"Error applying {error_context_gate_part}. Qiskit TypeError: {e}. "
                 f"Provided {len(processed_qubit_indices)} qubits: {processed_qubit_indices}. "
                 f"Gate likely expects a different number of qubits (approx. {num_expected_qubits}) or parameters. "
                 f"Your current gate categories (PARAMETRIC_MULTI_QUBIT, MULTI_PARAM_GATES) are empty, "
                 f"so parametric multi-qubit gates will not be auto-parameterized. "
                 f"(Check Qiskit docs for '{gate_method.__name__}' signature)."
             )
        except ValueError as ve:
            if f"{error_context_gate_part} requires multiple parameters" in str(ve):
                raise ve
            raise ValueError(f"Error applying {error_context_gate_part} to qubits {processed_qubit_indices}. Qiskit ValueError: {ve}")
        except Exception as e:
             raise RuntimeError(f"Unexpected error applying {error_context_gate_part} to qubits {processed_qubit_indices}: {e}")


    structure_queue = list(ansatz_structure)
    while structure_queue:
        element = structure_queue.pop(0)
        if isinstance(element, str):
            element_lower = element.lower()
            if element_lower in LAYER_INSTRUCTIONS:
                instruction_tuple = (element_lower, [])
                _process_instruction(instruction_tuple, ansatz, parameters_dict, param_idx_ref)
            elif element_lower == BARRIER_INSTRUCTION:
                instruction_tuple = (BARRIER_INSTRUCTION, [])
                _process_instruction(instruction_tuple, ansatz, parameters_dict, param_idx_ref)
            elif element_lower in ENTANGLEMENT_INSTRUCTIONS: # New block for entanglement
                entanglement_ops: List[Tuple[str, List[int]]] = []
                current_num_qubits = ansatz.num_qubits # Use num_qubits from the ansatz being built
                if element_lower == "linear_entanglement":
                    entanglement_ops = _create_linear_entanglement_ops(current_num_qubits)
                elif element_lower == "circular_entanglement":
                    entanglement_ops = _create_circular_entanglement_ops(current_num_qubits)
                elif element_lower == "full_entanglement":
                    entanglement_ops = _create_full_entanglement_ops(current_num_qubits)

                if entanglement_ops:
                    structure_queue[0:0] = entanglement_ops # Prepend generated CNOTs
                elif current_num_qubits < 2: # No warning if < 2 qubits as expected for entanglement
                    pass
                else: # Warn if ops are unexpectedly empty for >= 2 qubits
                     warnings.warn(f"Entanglement instruction '{element_lower}' produced no operations for {current_num_qubits} qubits. This might be unexpected.", UserWarning)

            else:
                supported_strings = sorted(list(LAYER_INSTRUCTIONS) + [BARRIER_INSTRUCTION] + list(ENTANGLEMENT_INSTRUCTIONS))
                raise ValueError(f"Unrecognized string instruction '{element}'. "
                                 f"Supported string instructions are: {', '.join(supported_strings)}. "
                                 f"Other gates must be in (gate_name, qubit_list) tuple format.")
        elif isinstance(element, tuple):
             _process_instruction(element, ansatz, parameters_dict, param_idx_ref)
        elif isinstance(element, list):
             structure_queue[0:0] = element
        else:
             supported_strings = sorted(list(LAYER_INSTRUCTIONS) + [BARRIER_INSTRUCTION] + list(ENTANGLEMENT_INSTRUCTIONS))
             raise TypeError(f"Elements in ansatz_structure must be a supported string "
                             f"(e.g., {', '.join(supported_strings)}), "
                             f"a tuple (gate_name, qubits_list), or a list (block). "
                             f"Found type '{type(element)}': {element}")

    # Parameter collection and sorting (unchanged)
    try:
        sorted_collected_parameters = sorted(parameters_dict.values(), key=lambda p: int(re.search(r'\d+', p.name).group()) if re.search(r'\d+', p.name) else float('inf'))
    except (AttributeError, IndexError, ValueError, TypeError):
        warnings.warn("Could not sort collected parameters numerically by name. Using default string sorting.", UserWarning)
        sorted_collected_parameters = sorted(parameters_dict.values(), key=lambda p: p.name)

    circuit_parameter_set = set(ansatz.parameters)
    collected_parameter_set = set(sorted_collected_parameters)

    if circuit_parameter_set != collected_parameter_set:
        warnings.warn(f"Parameter mismatch detected. Circuit params ({len(circuit_parameter_set)}): {[p.name for p in sorted(list(circuit_parameter_set), key=lambda p: p.name)]}, "
                      f"Collected ({len(collected_parameter_set)}): {[p.name for p in sorted(list(collected_parameter_set), key=lambda p: p.name)]}. "
                      "Using sorted list derived directly from circuit.parameters.", UserWarning)
        try:
            circuit_params_sorted = sorted(list(circuit_parameter_set), key=lambda p: int(re.search(r'\d+', p.name).group()) if re.search(r'\d+', p.name) else float('inf'))
        except (AttributeError, IndexError, ValueError, TypeError):
            warnings.warn("Could not sort circuit parameters numerically by name. Using default string sorting for final list.", UserWarning)
            circuit_params_sorted = sorted(list(circuit_parameter_set), key=lambda p: p.name)

        if not collected_parameter_set.issubset(circuit_parameter_set):
             missing_in_circuit = collected_parameter_set - circuit_parameter_set
             warnings.warn(f"Internal bookkeeping issue: Collected parameters {[p.name for p in missing_in_circuit]} are NOT in the final circuit.", UserWarning)
        if not circuit_parameter_set.issubset(collected_parameter_set):
             missing_in_collected = circuit_parameter_set - collected_parameter_set
             warnings.warn(f"Internal bookkeeping issue: Circuit parameters {[p.name for p in missing_in_collected]} were NOT collected during processing.", UserWarning)
        return ansatz, circuit_params_sorted
    return ansatz, sorted_collected_parameters