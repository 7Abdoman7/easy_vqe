import numpy as np
from easy_vqe import find_ground_state, draw_final_bound_circuit, print_results_summary, get_theoretical_ground_state_energy

# --- Define Hamiltonian ---
# Example: Simpler 2-Qubit Hamiltonian (adjust as needed)
hamiltonian_2q = "2*II-2*XX+3*YY-3*ZZ"

ansatz_block = [
    "ry_layer",
    "linear_entanglement",
    "rz_layer",
    "barrier" 
]

ansatz_structure = [
    ansatz_block,
    ansatz_block, 
    ansatz_block,
]

# --- Define Initial Parameters ---
n_shots = 8192
optimizer_method = 'COBYLA'
optimizer_options = {'maxiter': 500, 'rhobeg': 0.5, 'tol': 1e-5}
display_progress = True
plot_filename = "h2_convergence.png"
initial_params_strategy = 'random'  

# --- Run VQE ---
print("Starting VQE calculation...")

results = find_ground_state(
    ansatz_structure=ansatz_structure,
    hamiltonian_expression=hamiltonian_2q,
    n_shots=n_shots,
    optimizer_method=optimizer_method,
    optimizer_options=optimizer_options, 
    initial_params_strategy=initial_params_strategy,
    display_progress=display_progress,
    plot_filename=plot_filename
)

# --- Print Summary from Results Dictionary ---
print_results_summary(results)

# --- Draw Final Bound Circuit ---
draw_final_bound_circuit(results, draw_type='mpl', circuit_name="ansatz_2q")

# --- Theoretical Ground State Energy ---
theoretical_energy = get_theoretical_ground_state_energy(hamiltonian_2q)
print(f"Theoretical Ground State Energy: {theoretical_energy}")

# --- Compare with VQE Result ---
vqe_energy = results['optimal_value']
print(f"VQE Ground State Energy: {vqe_energy}")