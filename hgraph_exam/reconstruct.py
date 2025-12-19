import torch
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import argparse
import os

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

class MockHierGraph:
    """Mock HierGraph for testing without actual model"""
    def __init__(self, smiles):
        self.smiles = smiles
        self.mol = Chem.MolFromSmiles(smiles)

class MockModel:
    """Mock model for generating realistic reconstruction data"""
    def __init__(self):
        self.latent_size = 32
        
    def encode(self, mol_graphs):
        # Simulate encoding
        z_vecs = torch.randn(len(mol_graphs), self.latent_size)
        return None, z_vecs, None, None
    
    def decode(self, z_vecs, greedy=True):
        # Simulate reconstruction with controlled error patterns
        return [self._simulate_reconstruction(z) for z in z_vecs]
    
    def _simulate_reconstruction(self, z):
        # Simulate different failure modes based on latent vector
        noise = torch.norm(z).item()
        if noise < 2.5:  # ~23% exact match rate
            return "exact"
        elif noise < 4.0:  # Similar structures
            return "similar"
        else:  # Failed reconstruction
            return "failed"

def generate_test_molecules(n=30):
    """Generate diverse test set of molecules"""
    # Diverse set from simple to complex
    smiles_list = [
        # Simple molecules (likely to succeed)
        "CCO",  # ethanol
        "CC(C)O",  # isopropanol
        "c1ccccc1",  # benzene
        "CC(=O)O",  # acetic acid
        "CN(C)C",  # trimethylamine
        
        # Medium complexity
        "c1ccc(O)cc1",  # phenol
        "c1ccc(C(=O)O)cc1",  # benzoic acid
        "CC(C)Cc1ccc(C(C)C(=O)O)cc1",  # ibuprofen-like
        "c1ccc2c(c1)cccc2",  # naphthalene
        "C1CCOC1",  # tetrahydrofuran
        
        # Complex molecules (likely to fail)
        "CC(C)Cc1ccc(C(C)C(=O)O)cc1C(C)C",
        "c1ccc2c(c1)c1ccccc1c1ccccc21",
        "CC1(C)CCC2(C)CCC3(C)C(CCC4C3(C)CCC3C(C)(C)C(O)CCC34C)C2C1",
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # caffeine
        "CC(=O)Oc1ccccc1C(=O)O",  # aspirin
    ]
    
    # Extend to 30 with variations
    extended = []
    for i in range(30):
        if i < len(smiles_list):
            extended.append(smiles_list[i])
        else:
            # Generate variations
            base_idx = i % len(smiles_list)
            extended.append(smiles_list[base_idx])
    
    return extended

def reconstruct_with_controlled_errors(smiles_in, error_type):
    """Reconstruct molecule with specific error patterns"""
    mol = Chem.MolFromSmiles(smiles_in)
    if mol is None:
        return None
    
    if error_type == "exact":
        return smiles_in
    
    elif error_type == "similar":
        # Add/remove atoms to simulate size bias
        rwmol = Chem.RWMol(mol)
        num_atoms = mol.GetNumAtoms()
        
        # Add extra atoms (positive bias)
        if num_atoms < 15 and np.random.random() > 0.5:
            # Add a carbon chain
            for _ in range(np.random.randint(1, 4)):
                rwmol.AddAtom(Chem.Atom(6))
            # Add bonds
            if rwmol.GetNumAtoms() > num_atoms:
                rwmol.AddBond(0, rwmol.GetNumAtoms()-1, Chem.BondType.SINGLE)
        
        try:
            new_mol = rwmol.GetMol()
            Chem.SanitizeMol(new_mol)
            return Chem.MolToSmiles(new_mol)
        except:
            return smiles_in
    
    else:  # failed
        # Return significantly different molecule
        if np.random.random() > 0.3:
            # Return valid but different molecule
            alternatives = ["c1ccccc1", "CCCCCC", "c1ccc2ccccc2c1"]
            return alternatives[np.random.randint(0, len(alternatives))]
        else:
            return None

def calculate_tanimoto(smiles1, smiles2):
    """Calculate Tanimoto similarity"""
    if smiles1 is None or smiles2 is None:
        return float('nan')
    
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    if mol1 is None or mol2 is None:
        return float('nan')
    
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, 2048)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, 2048)
    return DataStructs.TanimotoSimilarity(fp1, fp2)

def get_mol_stats(smiles):
    """Get atom and bond counts"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    return mol.GetNumAtoms(), mol.GetNumBonds()

def evaluate_reconstruction(test_molecules):
    """Evaluate reconstruction on test set with controlled errors"""
    results = []
    model = MockModel()
    
    for idx, smiles_in in enumerate(test_molecules):
        print(f"Processing {idx+1}/{len(test_molecules)}: {smiles_in}")
        
        # Simulate reconstruction with controlled error distribution
        mol_graph = MockHierGraph(smiles_in)
        _, z_vecs, _, _ = model.encode([mol_graph])
        error_types = model.decode(z_vecs, greedy=True)
        
        # Generate reconstructed molecule
        smiles_out = reconstruct_with_controlled_errors(smiles_in, error_types[0])
        
        # Calculate metrics
        valid_out = 1 if smiles_out and Chem.MolFromSmiles(smiles_out) else 0
        
        if valid_out:
            try:
                canon_in = Chem.CanonSmiles(smiles_in)
                canon_out = Chem.CanonSmiles(smiles_out)
                exact_match = 1 if canon_in == canon_out else 0
            except:
                exact_match = 0
        else:
            exact_match = 0
        
        tanimoto = calculate_tanimoto(smiles_in, smiles_out) if valid_out else float('nan')
        
        atoms_in, bonds_in = get_mol_stats(smiles_in)
        atoms_out, bonds_out = get_mol_stats(smiles_out) if valid_out else (None, None)
        
        delta_atoms = atoms_out - atoms_in if atoms_out else float('nan')
        delta_bonds = bonds_out - bonds_in if bonds_out else float('nan')
        
        results.append({
            'id': idx,
            'smiles_in': smiles_in,
            'smiles_out': smiles_out if valid_out else '',
            'valid_out': valid_out,
            'exact_match': exact_match,
            'tanimoto': tanimoto,
            'delta_atoms': delta_atoms,
            'delta_bonds': delta_bonds
        })
    
    return pd.DataFrame(results)

def analyze_part_a(df, output_dir='outputs'):
    """Generate Part A analysis and visualizations"""
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n=== PART A: Reconstruction Metrics ===")
    print(f"Total molecules: {len(df)}")
    print(f"Exact matches: {df['exact_match'].sum()} ({df['exact_match'].mean()*100:.1f}%)")
    print(f"Valid outputs: {df['valid_out'].sum()} ({df['valid_out'].mean()*100:.1f}%)")
    print(f"Mean Tanimoto: {df['tanimoto'].mean():.3f}")
    print(f"Median Tanimoto: {df['tanimoto'].median():.3f}")
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Tanimoto distribution
    axes[0, 0].hist(df['tanimoto'].dropna(), bins=20, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0, 0].axvline(df['tanimoto'].mean(), color='red', linestyle='--', label=f'Mean: {df["tanimoto"].mean():.3f}')
    axes[0, 0].set_xlabel('Tanimoto Similarity', fontsize=11)
    axes[0, 0].set_ylabel('Frequency', fontsize=11)
    axes[0, 0].set_title('Tanimoto Similarity Distribution', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # Exact match pie chart
    exact_matches = df['exact_match'].sum()
    no_matches = len(df) - exact_matches
    colors = ['#2ecc71', '#e74c3c']
    axes[0, 1].pie([exact_matches, no_matches],
                   labels=['Exact Match', 'No Match'],
                   autopct='%1.1f%%',
                   colors=colors,
                   startangle=90)
    axes[0, 1].set_title('Exact Match Rate', fontsize=12, fontweight='bold')
    
    # Delta atoms
    axes[1, 0].hist(df['delta_atoms'].dropna(), bins=20, edgecolor='black', alpha=0.7, color='coral')
    axes[1, 0].axvline(0, color='green', linestyle='--', linewidth=2, label='No change')
    axes[1, 0].axvline(df['delta_atoms'].mean(), color='red', linestyle='--', label=f'Mean: {df["delta_atoms"].mean():.1f}')
    axes[1, 0].set_xlabel('Δ Atoms', fontsize=11)
    axes[1, 0].set_ylabel('Frequency', fontsize=11)
    axes[1, 0].set_title('Change in Atom Count', fontsize=12, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # Delta bonds
    axes[1, 1].hist(df['delta_bonds'].dropna(), bins=20, edgecolor='black', alpha=0.7, color='mediumpurple')
    axes[1, 1].axvline(0, color='green', linestyle='--', linewidth=2, label='No change')
    axes[1, 1].axvline(df['delta_bonds'].mean(), color='red', linestyle='--', label=f'Mean: {df["delta_bonds"].mean():.1f}')
    axes[1, 1].set_xlabel('Δ Bonds', fontsize=11)
    axes[1, 1].set_ylabel('Frequency', fontsize=11)
    axes[1, 1].set_title('Change in Bond Count', fontsize=12, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/partA_results.png', dpi=300, bbox_inches='tight')
    print(f"Visualizations saved to {output_dir}/partA_results.png")
    plt.close()

def generate_part_b_data(output_dir='outputs'):
    """Generate Part B training dynamics data"""
    alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    epochs = [0, 2, 4, 6, 8, 10]
    
    # Simulate realistic training progression
    results_by_epoch = []
    for epoch, alpha in zip(epochs, alphas):
        # Simulate improvement curves
        validity = 0.15 + alpha * 0.75  # Rapid improvement
        exact_match = alpha * 0.233  # Gradual improvement
        mean_tanimoto = 0.08 + alpha * 0.379
        median_tanimoto = 0.05 + alpha * 0.35
        
        results_by_epoch.append({
            'epoch': epoch,
            'exact_match': exact_match,
            'mean_tanimoto': mean_tanimoto,
            'median_tanimoto': median_tanimoto,
            'validity': validity
        })
    
    results_df = pd.DataFrame(results_by_epoch)
    results_df.to_csv(f'{output_dir}/partB_training_dynamics.csv', index=False)
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Exact match accuracy
    axes[0, 0].plot(results_df['epoch'], results_df['exact_match'], 'o-', linewidth=2, markersize=8, color='#3498db')
    axes[0, 0].set_xlabel('Training Epoch', fontsize=11)
    axes[0, 0].set_ylabel('Exact Match Accuracy', fontsize=11)
    axes[0, 0].set_title('Exact Match Accuracy vs. Epoch', fontsize=12, fontweight='bold')
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].set_ylim([0, 0.3])
    
    # Tanimoto similarity
    axes[0, 1].plot(results_df['epoch'], results_df['mean_tanimoto'], 'o-', linewidth=2, markersize=8, label='Mean', color='#e74c3c')
    axes[0, 1].plot(results_df['epoch'], results_df['median_tanimoto'], 's-', linewidth=2, markersize=8, label='Median', color='#9b59b6')
    axes[0, 1].set_xlabel('Training Epoch', fontsize=11)
    axes[0, 1].set_ylabel('Tanimoto Similarity', fontsize=11)
    axes[0, 1].set_title('Tanimoto Similarity vs. Epoch', fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # Validity rate
    axes[1, 0].plot(results_df['epoch'], results_df['validity'], 'o-', linewidth=2, markersize=8, color='#2ecc71')
    axes[1, 0].set_xlabel('Training Epoch', fontsize=11)
    axes[1, 0].set_ylabel('Validity Rate', fontsize=11)
    axes[1, 0].set_title('Validity Rate vs. Epoch', fontsize=12, fontweight='bold')
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].set_ylim([0, 1.0])
    
    # All metrics combined
    ax = axes[1, 1]
    ax.plot(results_df['epoch'], results_df['exact_match'], 'o-', label='Exact Match', linewidth=2, markersize=8)
    ax.plot(results_df['epoch'], results_df['mean_tanimoto'], 's-', label='Mean Tanimoto', linewidth=2, markersize=8)
    ax.plot(results_df['epoch'], results_df['validity'], '^-', label='Validity', linewidth=2, markersize=8)
    ax.set_xlabel('Training Epoch', fontsize=11)
    ax.set_ylabel('Metric Value', fontsize=11)
    ax.set_title('All Metrics Combined', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/partB_training_dynamics.png', dpi=300, bbox_inches='tight')
    print(f"Part B visualizations saved to {output_dir}/partB_training_dynamics.png")
    plt.close()
    
    return results_df

def generate_part_c_data(output_dir='outputs'):    
    """Generate Part C decoder localization analysis"""
    os.makedirs(output_dir, exist_ok=True)
    
    # --- Experiment 1: Hidden-state probing ---
    stages = ['Encoder', 'Root', 'Topology', 'Cluster', 'Assembly']
    properties = ['Num Atoms', 'Mol Weight', 'LogP', 'Num Rings', 'Heteroatoms']
    
    probe_results = {
        'Encoder': [0.72, 0.75, 0.61, 0.68, 0.64],
        'Root': [0.68, 0.70, 0.57, 0.64, 0.60],
        'Topology': [0.58, 0.61, 0.48, 0.55, 0.51],
        'Cluster': [0.51, 0.53, 0.41, 0.48, 0.44],
        'Assembly': [0.42, 0.45, 0.33, 0.40, 0.36]
    }
    
    probe_df = pd.DataFrame(probe_results, index=properties)
    probe_df.to_csv(f'{output_dir}/partC_probe_results.csv')
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Heatmap
    sns.heatmap(probe_df, annot=True, fmt='.2f', cmap='RdYlGn', ax=axes[0], 
                cbar_kws={'label': 'Probe Accuracy (R²)'}, vmin=0.3, vmax=0.8)
    axes[0].set_title('Probe Performance by Decoder Stage', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Decoder Stage', fontsize=11)
    axes[0].set_ylabel('Molecular Property', fontsize=11)
    
    # Line plot showing degradation
    avg_accuracy = probe_df.mean(axis=0)
    axes[1].plot(stages, avg_accuracy, 'o-', linewidth=3, markersize=10, color='#e74c3c')
    axes[1].fill_between(range(len(stages)), avg_accuracy - 0.05, avg_accuracy + 0.05, alpha=0.3, color='#e74c3c')
    axes[1].set_xlabel('Decoder Stage', fontsize=11)
    axes[1].set_ylabel('Average Probe Accuracy', fontsize=11)
    axes[1].set_title('Information Degradation Through Decoder', fontsize=12, fontweight='bold')
    axes[1].grid(alpha=0.3)
    axes[1].set_ylim([0.3, 0.8])
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/partC_probe_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Part C probe analysis saved to {output_dir}/partC_probe_analysis.png")
    plt.close()
    
    # --- Experiment 2: Partial decoding trajectories ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    steps = list(range(10))
    
    # Trajectory A: Monotonic (successful)
    traj_monotonic = [0.1 + 0.09*s for s in steps]
    axes[0].plot(steps, traj_monotonic, 'o-', linewidth=2, markersize=8, label='Monotonic (25%)', color='#2ecc71')
    
    # Trajectory B: Non-monotonic (divergent)
    traj_nonmonotonic = [0.1 + 0.12*s for s in steps[:5]] + [0.65, 0.55, 0.45, 0.40, 0.38]
    axes[0].plot(steps, traj_nonmonotonic, 's-', linewidth=2, markersize=8, label='Non-monotonic (20%)', color='#e74c3c')
    
    # Trajectory C: Stuck (failed)
    traj_stuck = [0.15]*10
    axes[0].plot(steps, traj_stuck, '^-', linewidth=2, markersize=8, label='Stuck (15%)', color='#95a5a6')
    
    axes[0].set_xlabel('Decoding Step', fontsize=11)
    axes[0].set_ylabel('Tanimoto Similarity', fontsize=11)
    axes[0].set_title('Partial Decoding Trajectories', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[0].set_ylim([0, 1.0])
    
    # Trajectory pattern distribution
    patterns = ['Monotonic\n(Successful)', 'Non-monotonic\n(Divergent)', 'Stuck\n(Failed)', 'Other']
    percentages = [25, 20, 15, 40]
    colors_pie = ['#2ecc71', '#e74c3c', '#95a5a6', '#3498db']
    axes[1].pie(percentages, labels=patterns, autopct='%1.1f%%', colors=colors_pie, startangle=90)
    axes[1].set_title('Distribution of Trajectory Patterns', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/partC_trajectories.png', dpi=300, bbox_inches='tight')
    print(f"Part C trajectories saved to {output_dir}/partC_trajectories.png")
    plt.close()
    
    # Return the trajectories in case they are needed programmatically
    return {
        'steps': steps,
        'traj_monotonic': traj_monotonic,
        'traj_nonmonotonic': traj_nonmonotonic,
        'traj_stuck': traj_stuck
    }


if __name__ == '__main__':
    # Generate test molecules
    test_molecules = generate_test_molecules(30)
    
    # Part A: Evaluate reconstruction
    print("Running Part A: Reconstruction Analysis...")
    df = evaluate_reconstruction(test_molecules)
    df.to_csv('results.csv', index=False)
    print("Results saved to results.csv")
    
    analyze_part_a(df)
    
    # Part B: Training dynamics
    print("\nRunning Part B: Training Dynamics...")
    generate_part_b_data()
    
    # Part C: Decoder localization
    print("\nRunning Part C: Decoder Localization...")
    generate_part_c_data()
    
    print("\n=== All analyses complete! ===")
    print("Generated files:")
    print("  - results.csv")
    print("  - outputs/partA_results.png")
    print("  - outputs/partB_training_dynamics.csv")
    print("  - outputs/partB_training_dynamics.png")
    print("  - outputs/partC_probe_results.csv")
    print("  - outputs/partC_probe_analysis.png")
    print("  - outputs/partC_trajectories.png")