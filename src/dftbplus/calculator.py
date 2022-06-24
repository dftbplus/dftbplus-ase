#------------------------------------------------------------------------------#
#  DFTB+: general package for performing fast atomistic simulations            #
#  Copyright (C) 2006 - 2022  DFTB+ developers group                           #
#                                                                              #
#  See the LICENSE file for terms of usage and distribution.                   #
#------------------------------------------------------------------------------#

"""This module defines an ASE interface to DFTB+, based on FileIO."""


import os
import copy
import warnings

import dftbplus_ptools.resultstag
from dftbplus_ptools.hsdinput import Hsdinput
from dftbplus_ase.io import atom_to_geometry
from ase.io import write
from ase.units import Hartree, Bohr
from ase.calculators.calculator import FileIOCalculator


class DftbPlus(FileIOCalculator):
    """ASE interface to DFTB+.

       A DFTB+ calculator with ase-FileIOCalculator nomenclature.
    """

    implemented_properties = ('energy', 'forces', 'charges', 'stress',
                              'occupations', 'fermi_levels', 'fermi_level',
                              'eigenvalues', 'dipole')


    def __init__(self, inp, binary='dftb+',
                 restart=None, label='dftbplus', atoms=None, **kwargs):
        """Initializes a ctypes DFTB+ calculator object.

        Args:
            inp (dict): dictionary containing (additional)
                initialization options for DFTB+
            binary (str): location of dftb+ binary
            restart (str): Prefix for restart file. May contain a directory.
            label (str): Name used for all files.
        """

        self._hsdinput = Hsdinput(dictionary=copy.deepcopy(inp))
        self._hamiltonian = self._hsdinput.get_hamiltonian()
        self._do_forces = False
        self._atoms = None
        self._atoms_input = None

        FileIOCalculator.__init__(self, restart=restart, label=label,
                                  atoms=atoms, command=binary, **kwargs)

        # Determine number of spin channels
        try:
            spinpolkeys = list(self._hsdinput['hamiltonian']\
                               [self._hamiltonian]['spinpolarisation'].keys())
            self.nspin = 2 if 'colinear' in spinpolkeys else 1
        except KeyError:
            self.nspin = 1


    def _write_dftb_in(self, filename):
        """Dumps Python input dictionary to HSD format and writes input file.
           If necessary, geometry-relevant entries in the input are updated.
           If not all max. angular momenta have been specified by user, an
           attempt is made to extract remaining ones out of Slater-Koster
           files.

        Args:
            filename (str): name of input file to be written

        Returns:
            (file): containing dictionary in HSD format
        """
        geometry = atom_to_geometry(self._atoms_input)

        specieslist = []
        for species in geometry.specienames:
            specieslist.append(species.lower())

        speciesdiff = set(specieslist) - \
            set(self._hsdinput['hamiltonian'][self._hamiltonian]
                ['maxangularmomentum'].keys())
        maxangs = self._hsdinput['hamiltonian'][self._hamiltonian]\
            ['maxangularmomentum']

        path = os.path.join(self.directory, filename)
        self._hsdinput.set_filename(path)
        self._hsdinput.set_geometry(geometry)

        if speciesdiff and self._hamiltonian == "dftb":
            self._hsdinput.set_maxang(maxangs=maxangs, try_reading=speciesdiff)
        elif speciesdiff and self._hamiltonian != "dftb":
            raise ValueError(f"max. angular momanta for '{speciesdiff}' " +
                             "missing!")
        else:
            self._hsdinput.set_maxang(maxangs=maxangs)

        self._hsdinput.write_resultstag()
        warnings.warn("Change in input: Because the calculator requires " +
                      "'WriteResultsTag' to be set to 'Yes', it was set to " +
                      "'Yes' regardless of its initial value.")

        self._hsdinput.write_hsd()


    def set(self, **kwargs):
        """sets and changes parameters

        Returns:
            changed_parameters (dict): containing changed parameters
        """
        changed_parameters = FileIOCalculator.set(self, **kwargs)
        if changed_parameters:
            self.reset()
        return changed_parameters


    def check_state(self, atoms, tol=1e-15):
        """Check for any system changes since last calculation.

        Args:
            atoms (atoms object): atom to be compared to
            tol (float): tolerance for comparison

        Returns:
            system_changes (list): containing changed properties
        """
        system_changes = FileIOCalculator.check_state(self, atoms, tol=1e-15)
        # Ignore unit cell for molecules:
        if not atoms.pbc.any() and 'cell' in system_changes:
            system_changes.remove('cell')
        return system_changes


    def write_input(self, atoms, properties=None, system_changes=None):
        """Writes input files

        Args:
            atoms (atoms object): atom to read from
            properties (list): list of what needs to be calculated
            system_changes (list): List of what has changed since last
                calculation
        """
        if properties is not None:
            if 'forces' in properties or 'stress' in properties:
                self._do_forces = True
                self._hsdinput.calc_forces()
            if 'charges' in properties:
                self._hsdinput.calc_charges()

        FileIOCalculator.write_input(self, atoms, properties, system_changes)

        # self._atoms is none until results are read out,
        # then it is set to the ones at writing input
        self._atoms_input = atoms
        self._atoms = None

        self._write_dftb_in('dftb_in.hsd')

        write(os.path.join(self.directory, 'geo_end.gen'), atoms)


    def read_results(self):
        """All results are read from results.tag file.
        """

        self._atoms = self._atoms_input

        path = os.path.join(self.directory, "results.tag")
        resultstag = dftbplus_ptools.resultstag.Output(path)

        if resultstag.get_gross_atomic_charges() is not None:
            self.results['charges'] = resultstag.get_gross_atomic_charges()
        self.results['energy'] = resultstag.get_total_energy() * Hartree

        if resultstag.get_fermi_level() is not None:
            self.results['fermi_levels'] = resultstag.get_fermi_level() \
                * Hartree
            self.results['fermi_level'] = max(resultstag.get_fermi_level()
                                              * Hartree)

        if resultstag.get_eigenvalues() is not None:
            self.results['eigenvalues'] = resultstag.get_eigenvalues() \
                * Hartree

        if self._do_forces:
            self.results['forces'] = resultstag.get_forces() * Hartree / Bohr

        if resultstag.get_stress() is not None:
            self.results['stress'] = - resultstag.get_stress() * Hartree \
                / Bohr**3

        if resultstag.get_filling() is not None:
            self.results['occupations'] = resultstag.get_filling()

        if resultstag.get_dipole_moments() is not None:
            self.results['dipole'] = resultstag.get_dipole_moments() * Bohr
        os.remove(os.path.join(self.directory, 'results.tag'))


    def get_charges(self, atoms):
        """Get the calculated charges. This is inhereted to atoms object. (In
        get_charges of atoms object: "return self._calc.get_charges(self)")

        Args:
            atoms (atoms object): atom to read charges from
        """

        return self.get_property('charges', allow_calculation=True).copy()


    def get_number_of_spins(self):
        """Auxiliary function to extract result: number of spins.

        Returns:
            self.nspin (int): number of spins
        """

        return self.nspin


    def get_eigenvalues(self, kpt=0, spin=0):
        """Auxiliary function to extract result: eigenvalues.

        Args:
            kpt (int): kpt where eigenvalues are desired
            spin (int): spin where eigenvalues are desired

        Returns:
            (array): desired eigenvalues
        """
        eigenval = self.get_property('eigenvalues', allow_calculation=False)
        return eigenval[spin][kpt].copy()


    def get_fermi_levels(self):
        """Auxiliary function to extract result: Fermi level(s).

        Returns:
            (array): containing fermi levels
        """

        return self.get_property('fermi_levels',
                                 allow_calculation=False).copy()


    def get_fermi_level(self):
        """Auxiliary function to extract result: result: max. Fermi level.

        Returns:
            (np.float64): max. Fermi level
        """

        return self.get_property('fermi_level', allow_calculation=False)


    def get_occupation_numbers(self, kpt=0, spin=0):
        """Auxiliary function to extract result: occupations.

        Args:
            kpt (int): kpt where eigenvalues are desired
            spin (int): spin where eigenvalues are desired

        Returns:
            occs (array): desired occupations
        """

        occs = self.get_property('occupations', allow_calculation=False).copy()
        return occs[kpt, spin]

    def get_dipole_moment(self, atoms=None):
        return self.get_property('dipole', atoms,
                                 allow_calculation=False).copy()
