import numpy as np
from easy_vqe import find_ground_state, draw_final_bound_circuit, print_results_summary, get_theoretical_ground_state_energy

# --- Define Hamiltonian ---
# Example: Simpler 3-Qubit Hamiltonian (adjust as needed)
hamiltonian_3q = "-ZZI+0.9*ZIZ-0.5*IZZ+0.2*XXX-0.3*YYY"

# --- Define Ansatz Structure ---
ansatz_block = [
    "ry_layer",
    "linear_entanglement",
    "rz_layer",
    "barrier",
]

ansatz_structure = [
    ansatz_block,
    ansatz_block,
]

# --- Define Initial Parameters ---
n_shots = 8192
optimizer_method = 'COBYLA'
optimizer_options = {'maxiter': 250, 'rhobeg': 0.5, 'tol': 1e-5}
display_progress = True
plot_filename = "h3_convergence.png"
initial_params_strategy = 'zeros'  

# --- Run VQE ---
print("Starting VQE calculation...")

results = find_ground_state(
    ansatz_structure=ansatz_structure,
    hamiltonian_expression=hamiltonian_3q,
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
draw_final_bound_circuit(results)

# --- Theoretical Ground State Energy ---
theoretical_energy = get_theoretical_ground_state_energy(hamiltonian_3q)
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