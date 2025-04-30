import numpy as np
from easy_vqe import find_ground_state, draw_final_bound_circuit, print_results_summary, get_theoretical_ground_state_energy

# --- Define Hamiltonian ---
# Example: Simpler 2-Qubit Hamiltonian (adjust as needed)
hamiltonian_2q = "2*II-2*XX+3*YY-3*ZZ"

# --- Define Ansatz Structure ---
ansatz_block_linear_ent = [
    ('ry', [0, 1]),
    ('cx', [0, 1]),
    ('rz', [0, 1]),
]

ansatz_structure = [
    ansatz_block_linear_ent,
    ("barrier", []),  
    ansatz_block_linear_ent, 
]


# --- Run VQE ---
print("Starting VQE calculation...")

results = find_ground_state(
    ansatz_structure=ansatz_structure,
    hamiltonian_expression=hamiltonian_2q,
    n_shots=8192,
    optimizer_method='COBYLA',
    optimizer_options={'maxiter': 500, 'rhobeg': 0.5, 'tol': 1e-5}, 
    initial_params_strategy='random',
    display_progress=True,
    plot_filename="h2_convergence.png" 
)

# --- Print Summary from Results Dictionary ---
print_results_summary(results)

# --- Draw Final Bound Circuit ---
draw_final_bound_circuit(results)

# --- Theoretical Ground State Energy ---
theoretical_energy = get_theoretical_ground_state_energy(hamiltonian_2q)
print(f"Theoretical Ground State Energy: {theoretical_energy}")

# --- Compare with VQE Result ---
vqe_energy = results['optimal_value']
print(f"VQE Ground State Energy: {vqe_energy}")

if np.isclose(vqe_energy, theoretical_energy, atol=1e-1):
    print("VQE result is close to the theoretical ground state energy.")
else:
    print("VQE result is NOT close to the theoretical ground state energy.")

# --- Percent Error ---
print("Percent error:", abs((vqe_energy - theoretical_energy) / theoretical_energy) * 100, "%")