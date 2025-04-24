import numpy as np
from easy_vqe import find_ground_state # Import the main function

# --- Define Hamiltonian ---
# Example: Simpler 3-Qubit Hamiltonian (adjust as needed)
hamiltonian_3q = "-1.0 * ZZI + 0.9 * ZIZ - 0.5 * IZZ + 0.2 * XXX - 0.3 * YYY"

# --- Define Ansatz Structure ---
ansatz_block = [
    ('ry', [0, 1, 2]),
    ('cx', [0, 1]),
    ('cx', [1, 2]),
    ('rz', [0, 1, 2]),
]
ansatz_structure = [
    ('h', [0, 1, 2]),
    ansatz_block,
    ('cx', [0, 2]),
    ('barrier', []),
    ansatz_block,
]

# --- Run VQE ---
print("Starting VQE calculation...")

results = find_ground_state(
    ansatz_structure=ansatz_structure,
    hamiltonian_expression=hamiltonian_3q,
    n_shots=2048,
    optimizer_method='COBYLA',
    # Allow more iterations than before
    optimizer_options={'maxiter': 200, 'rhobeg': 0.5, 'tol': 1e-5}, # Added tolerance
    initial_params_strategy='random',
    # initial_params_strategy=np.array([...]), # Can provide specific start values
    display_progress=True,
    plot_filename="h3_convergence.png" # Specify filename to save plot
)

# --- Print Summary from Results Dictionary ---
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

    print("\nFinal Ansatz Circuit:")
    try:
        # Draw with parameters bound to optimal values
        final_bound_circuit = results['ansatz'].assign_parameters(results['optimal_params'])
        print(final_bound_circuit.draw(output='text', fold=-1))
    except Exception as e:
        print(f"(Could not draw circuit: {e})")

print("="*40)

if not 'error' in results:
    print(f"\nLowest energy: {results['optimal_value']}")
    print(f"Parameter history length: {len(results['parameter_history'])}")