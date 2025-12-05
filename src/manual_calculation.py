import numpy as np
import sys

# Try to import PySCF (Standard Python library for DFT)
try:
    from pyscf import gto, dft, grad
except ImportError:
    print("Error: PySCF is not installed.")
    print("Please install it to run DFT calculations: pip install pyscf")
    sys.exit(1)

def calculate_dft_energy_forces(positions, atom_labels, basis='6-31G', functional='B3LYP'):
    """
    Calculates the Total Energy and Forces on atoms using Density Functional Theory (DFT).
    
    This uses the PySCF library to solve the Kohn-Sham equations.
    
    Args:
        positions (np.ndarray): N x 3 array of atomic positions (in Angstroms).
        atom_labels (list): List of atomic symbols corresponding to positions 
                            (e.g., ['H', 'O', 'H']).
        basis (str): The basis set to use (e.g., 'sto-3g', '6-31g', 'cc-pvdz').
                     'sto-3g' is fast/inaccurate. '6-31g' is standard.
        functional (str): The DFT functional (e.g., 'lda', 'b3lyp', 'pbe').

    Returns:
        tuple: 
            - total_energy (float): Electronic energy in Hartree (atomic units).
            - forces (np.ndarray): N x 3 array of forces in Hartree/Bohr.
    """
    
    # 1. Format the input for PySCF
    # PySCF expects an atom string like: "O 0 0 0; H 0 0 0.7; H 0 0.6 0"
    atom_string = []
    for i, label in enumerate(atom_labels):
        x, y, z = positions[i]
        atom_string.append(f"{label} {x} {y} {z}")
    
    atom_string = "; ".join(atom_string)
    
    # 2. Build the Molecule object
    mol = gto.M(
        atom=atom_string,
        basis=basis,
        unit='Angstrom', # We specify inputs are in Angstroms
        verbose=0        # Suppress massive console output
    )
    
    # 3. Setup the DFT Method (Restricted Kohn-Sham)
    mf = dft.RKS(mol)
    mf.xc = functional  # Set the exchange-correlation functional (e.g., B3LYP)
    
    # 4. Run the Kernel (Calculate Energy)
    # This solves the Self-Consistent Field (SCF) equations.
    # This is the "expensive" part that takes time.
    total_energy = mf.kernel()
    
    # Check if the calculation converged
    if not mf.converged:
        print("Warning: DFT calculation did not converge!")

    # 5. Calculate Gradients (Forces)
    # In physics, Force = -Gradient. PySCF calculates gradients (dE/dx).
    # So we must negate the result to get Force.
    gradients = mf.nuc_grad_method().kernel()
    forces = -gradients

    return total_energy, forces

# --- Usage Example ---
if __name__ == "__main__":
    # Example: Water Molecule (H2O)
    # Approximate geometry in Angstroms
    atom_types = ['O', 'H', 'H']
    
    # O at origin, H slightly offset
    pos = np.array([
        [0.0000,  0.0000,  0.0000], # Oxygen
        [0.7570,  0.5860,  0.0000], # Hydrogen 1
        [-0.7570, 0.5860,  0.0000]  # Hydrogen 2
    ])
    
    print(f"--- Running DFT ({len(atom_types)} atoms) ---")
    print("Method: B3LYP / 6-31G")
    print("This may take a few seconds...\n")
    
    try:
        # Calculate
        energy, forces = calculate_dft_energy_forces(pos, atom_types, basis='6-31G')
        
        # Output Results
        print(f"Total DFT Energy: {energy:.6f} Hartree")
        print("\nDFT Forces (Hartree/Bohr):")
        for i, (label, force) in enumerate(zip(atom_types, forces)):
            print(f"{label} (Atom {i}): {force}")
            
        print("\nNote: These forces push the atoms toward the equilibrium geometry.")
        
    except Exception as e:
        print(f"An error occurred: {e}")    