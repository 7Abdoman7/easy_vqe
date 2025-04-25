import numpy as np
from easy_vqe import find_ground_state, draw_final_bound_circuit, print_results_summary, get_theoretical_ground_state_energy

# --- Define Hamiltonian ---
# Example: Simpler 4-Qubit Hamiltonian (adjust as needed)
hamiltonian_4q = """\
- 0.81054778 * IIII \
+ 0.17141281 * IIIZ \
+ 0.17141281 * IIZI \
+ 0.12062863 * IIZZ \
- 0.22343154 * IZII \
+ 0.16862219 * IZIZ \
+ 0.12062863 * IZZI \
- 0.22343154 * ZIII \
+ 0.16862219 * ZIIZ \
+ 0.16591703 * ZIZI \
+ 0.16591703 * ZZII \
+ 0.04532223 * IXXI \
+ 0.04532223 * XYYX \
+ 0.04532223 * YXXY \
+ 0.04532223 * YYII\
"""

# --- Define Ansatz Structure ---
ansatz_block = [
    ('ry', list(range(4))),  # Apply ry to all qubits
    ('rz', list(range(4))),  # Apply rz to all qubits
    ('ry', list(range(4))),  # Apply ry to all qubits
    ('cx', [0, 1]),
    ('cx', [1, 2]),
    ('cx', [2, 3]),
    ('ry', list(range(4))),  # Apply ry to all qubits
    ('rz', list(range(4))),  # Apply rz to all qubits
    ('ry', list(range(4))),  # Apply ry to all qubits
    ('cx', [0, 1]),
    ('cx', [1, 2]),
    ('cx', [2, 3]),
]
ansatz_structure = [
    ansatz_block,
    ansatz_block
]

# --- Run VQE ---
print("Starting VQE calculation...")

results = find_ground_state(
    ansatz_structure=ansatz_structure,
    hamiltonian_expression=hamiltonian_4q,
    n_shots=8192,
    optimizer_method='COBYLA',
    optimizer_options={'maxiter': 500, 'rhobeg': 0.5, 'tol': 1e-5},
    initial_params_strategy='random',
    display_progress=True,
    plot_filename="h4_convergence.png",
)

# --- Print Summary from Results Dictionary ---
print_results_summary(results)

# --- Draw Final Bound Circuit ---
draw_final_bound_circuit(results)

# --- Theoretical Ground State Energy ---
theoretical_energy = get_theoretical_ground_state_energy(hamiltonian_4q)
print(f"Theoretical Ground State Energy: {theoretical_energy}")

# --- Compare with VQE Result ---
vqe_energy = results['final_energy']
print(f"VQE Ground State Energy: {vqe_energy}")

if np.isclose(vqe_energy, theoretical_energy, atol=1e-2):
    print("VQE result is close to the theoretical ground state energy.")
else:
    print("VQE result is NOT close to the theoretical ground state energy.")