"""
BioPython implementation of PyMOL's get_sasa_relative function.

Calculates relative per-residue solvent accessible surface area (SASA).
The value is relative to full exposure of the residue, calculated by
removing all other residues except its two next neighbors, if present.

Returns an array of values between 0.0 (fully buried) and 1.0 (fully exposed),
ordered by residue index.
"""

from Bio.PDB.SASA import ShrakeRupley
from Bio.PDB.StructureBuilder import StructureBuilder
import numpy as np


def get_sasa_relative(struct):
    """
    Calculate relative SASA for each residue in the structure.
    
    Args:
        struct: BioPython Structure object (parsed PDB structure)
        
    Returns:
        numpy array of floats: Relative SASA values (0.0-1.0) for each residue,
        ordered by residue index (model -> chain -> residue)
    """
    # Initialize SASA calculator
    sr = ShrakeRupley()
    
    # Step 1: Calculate SASA for full structure
    sr.compute(struct, level="R")
    
    # Collect all residues in order and store their full structure SASA
    residues = []
    full_sasa = {}
    
    for model in struct:
        for chain in model:
            for residue in chain:
                residues.append(residue)
                full_sasa[residue] = residue.sasa
    
    # Step 2: For each residue, create tripeptide and calculate relative SASA
    relative_sasa = []
    
    for i, residue in enumerate(residues):
        # Get the chain this residue belongs to
        chain = residue.parent
        
        # Find neighbors: previous and next residue in the same chain
        chain_residues = list(chain)
        try:
            res_index = chain_residues.index(residue)
        except ValueError:
            # Should not happen, but handle gracefully
            relative_sasa.append(0.0)
            continue
        
        # Collect residues for tripeptide: previous, current, next
        tripeptide_residues = []
        target_index_in_tripeptide = 0  # Track where the target residue is
        
        if res_index > 0:
            tripeptide_residues.append(chain_residues[res_index - 1])
            target_index_in_tripeptide += 1
        tripeptide_residues.append(residue)
        if res_index < len(chain_residues) - 1:
            tripeptide_residues.append(chain_residues[res_index + 1])
        
        # Create temporary tripeptide structure
        builder = StructureBuilder()
        builder.init_structure('tripeptide')
        builder.init_model(0)
        builder.init_chain(chain.id)
        
        tripeptide_struct = builder.get_structure()
        tripeptide_model = tripeptide_struct[0]
        
        # Get the chain we just created - try direct access first, then fallback
        try:
            tripeptide_chain = tripeptide_model[chain.id]
        except (KeyError, TypeError):
            # Fallback: get the first chain if direct access fails
            tripeptide_chain = list(tripeptide_model.get_chains())[0]
        
        # Copy residues to tripeptide structure
        for res in tripeptide_residues:
            res_copy = res.copy()
            tripeptide_chain.add(res_copy)
        
        # Step 3: Calculate SASA for tripeptide
        sr.compute(tripeptide_struct, level="R")
        
        # Step 4: Get the SASA of the target residue in tripeptide
        # The target residue is at target_index_in_tripeptide position
        tripeptide_res_list = list(tripeptide_chain)
        if len(tripeptide_res_list) > target_index_in_tripeptide:
            target_res_in_tripeptide = tripeptide_res_list[target_index_in_tripeptide]
        else:
            # Fallback: use the middle residue if index is out of range
            if len(tripeptide_res_list) > 0:
                mid_idx = len(tripeptide_res_list) // 2
                target_res_in_tripeptide = tripeptide_res_list[mid_idx]
            else:
                relative_sasa.append(0.0)
                continue
        
        exposed_sasa = target_res_in_tripeptide.sasa
        
        # Step 5: Calculate relative SASA
        if exposed_sasa > 0.0:
            relative = full_sasa[residue] / exposed_sasa
            # Clamp to [0.0, 1.0] range (should already be in this range, but ensure it)
            relative = max(0.0, min(1.0, relative))
        else:
            # If exposed SASA is zero, set relative to 0.0
            relative = 0.0
        
        relative_sasa.append(relative)
    
    return np.array(relative_sasa)

