from __future__ import annotations
from pymatgen.core import Structure, Lattice, PeriodicSite
from pymatgen.io.cif import CifWriter
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer, SymmetrizedStructure
import numpy as np
from pymatgen.symmetry.groups import SpaceGroup
from typing import List, Tuple

class MyCrystal:
    def __init__(self, sg: str | SpaceGroup, lattice: np.ndarray | List | Lattice, species: List[str], coords: List[np.ndarray], multiplicities: List[int]):
        if(isinstance(sg, str)):
            sg = SpaceGroup(sg)
        self.sg = sg
        
        if(isinstance(lattice, np.ndarray) or isinstance(lattice, List)):
            lattice = Lattice.from_parameters(lattice[0], lattice[1], lattice[2], lattice[3], lattice[4], lattice[5])
        self.lattice = lattice

        self.multiplicities = multiplicities
        self.species = species
        self.coords = coords
        assert(len(multiplicities) == len(species) and len(species) == len(coords))
        self.tol = 0.01
        pass

    def __str__(self):
        s = f"sg = {self.sg}\nlattice = {self.lattice}\n"
        for idx, (sp, c, m) in enumerate(zip(self.species, self.coords, self.multiplicities)):
            s += f"{sp} {c} {m}\n"
        return s

    @classmethod
    def from_simple(cls, simple_str: str):
        lines = [line.strip() for line in simple_str.strip().split('\n') if line.strip()]
        sg_symbol = lines[0]
        lengths = [float(x) for x in lines[1].split()]
        angles = [float(x) for x in lines[2].split()]
        lattice_params = lengths + angles
        species = []
        coords = []
        input_multiplicities = [] 
        for line in lines[3:]:
            parts = line.split()
            species.append(parts[0])
            input_multiplicities.append(int(parts[1]))
            coords.append(np.array([float(parts[2]), float(parts[3]), float(parts[4])]))
        
        return MyCrystal(sg_symbol, lattice_params, species, coords, input_multiplicities)
    
    def get_primitive_structure(self):
        species = []
        coords = []
        multiplicities = []
        for idx, (sp, c, m) in enumerate(zip(self.species, self.coords, self.multiplicities)):
            cc = self.sg.get_orbit(c, tol=self.tol)
            if(m != len(cc)):
                raise Exception(f"Error in get_primitive_structure\n{self.sg} {sp} {c} {m}\n{cc} ")
            species.extend([sp] * len(cc))
            coords.extend(cc)
            multiplicities.extend([1] * len(cc))
        
        return MyCrystal("P1", self.lattice, species, coords, multiplicities)
    
    def _get_comparable_site_list(self) -> List[Tuple[str, int, Tuple]]:
        site_list = []
        for s, m, c in zip(self.species, self.multiplicities, self.coords):
            coords_tuple = tuple(c.flatten())
            site_list.append((s, m, coords_tuple))
            
        return site_list

    def __eq__(self, crys: 'MyCrystal') -> bool:
        if not isinstance(crys, MyCrystal):
            return NotImplemented
        
        if self.sg != crys.sg:
            return False

        if self.lattice != crys.lattice:
            return False

        if len(self.species) != len(crys.species):
            return False
        
        self_sites_sorted = sorted(self._get_comparable_site_list())
        crys_sites_sorted = sorted(crys._get_comparable_site_list())
        for idx, (self_site, crys_site) in enumerate(zip(self_sites_sorted, crys_sites_sorted)):
            if self_site[0] != crys_site[0] or self_site[1] != crys_site[1]:
                print(idx,self_site, crys_site)
                return False
            self_coords_array = np.array(self_site[2])
            crys_coords_array = np.array(crys_site[2])
            if not np.allclose(self_coords_array, crys_coords_array, rtol=0, atol=self.tol):
                print(idx,self_site, crys_site)
                return False
            
        return True
    
    def compare_with_primitive(self, crys: MyCrystal):
        my_primitive = self.get_primitive_structure()
        return my_primitive == crys


class SimpleCrystal(object):

    def __init__(self, structure: Structure):
        self.structure = structure
        self.sga = None
        self.sym_structure = None

    @classmethod
    def from_sym_structure(cls, structure: Structure):
        crys = SimpleCrystal(structure)
        crys.sga = SpacegroupAnalyzer(structure, symprec=0.1)
        crys.sym_structure = crys.sga.get_refined_structure()
        return crys

    @classmethod
    def from_simple_no_sym(cls, simple_str: str):
        lines = [line.strip() for line in simple_str.strip().split('\n') if line.strip()]
        sg_symbol = lines[0]
        lengths = [float(x) for x in lines[1].split()]
        angles = [float(x) for x in lines[2].split()]
        lattice_params = lengths + angles
        lattice = Lattice.from_parameters(lengths[0], lengths[1], lengths[2], angles[0], angles[1], angles[2])
        species = []
        coords = []
        input_multiplicities = [] 
        for line in lines[3:]:
            parts = line.split()
            species.append(parts[0])
            input_multiplicities.append(int(parts[1]))
            coords.append([float(parts[2]), float(parts[3]), float(parts[4])])

        structure = Structure(
            lattice=lattice, species=species, coords=coords, coords_are_cartesian=False
        )
        return SimpleCrystal(structure)

    def to_simple_no_sym(self):
        sg_symbol = "P1" 
        
        lat = self.structure.lattice
        lengths_str = f"{lat.a:.8f} {lat.b:.8f} {lat.c:.8f}"
        angles_str = f"{lat.alpha:g} {lat.beta:g} {lat.gamma:g}"
        
        lines = [sg_symbol, lengths_str, angles_str]

        for site in self.structure:
            specie = site.specie.symbol
            multiplicity = 1 
            frac = site.frac_coords
            site_str = f"{specie} {multiplicity} {frac[0]:.8f} {frac[1]:.8f} {frac[2]:.8f}"
            lines.append(site_str)

        return "\n".join(lines)

    def to_cif(self):
        cif_writer = CifWriter(self.structure, symprec=0.01)
        return str(cif_writer)
    
if __name__ == "__main__":
    simple = """
P3_121
4.50482070 4.50482070 4.97879632
90 90 120
Se 3 0.21789298 0.00000000 0.33333333
"""
    crystal = SimpleCrystal.from_simple_no_sym(simple)
    print(crystal.to_cif())
    print("")
    print(crystal.to_simple_no_sym())