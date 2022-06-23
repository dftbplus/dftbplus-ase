#------------------------------------------------------------------------------#
#  dftbplus-ase: Interfacing DFTB+ with the Atomic Simulation Environment      #
#  Copyright (C) 2006 - 2022  DFTB+ developers group                           #
#                                                                              #
#  See the LICENSE file for terms of usage and distribution.                   #
#------------------------------------------------------------------------------#


'''This module defines an ASE interface to DFTB+, based on FileIO.'''


import os
import copy
import numpy as np

import hsd
from ase.io import write
from ase.units import Hartree, Bohr
from ase.calculators.calculator import FileIOCalculator


def read_input(filename):
    '''Deserializes HSD input into nested Python dictionaries.

    Args:
        filename (str): name of HSD input to read

    Returns:
        dftbinp (dict): DFTB+ input as nested dictionaries
    '''

    dictbuilder = hsd.HsdDictBuilder()
    parser = hsd.HsdParser(eventhandler=dictbuilder)
    with open(os.path.join(os.getcwd(), filename), 'r') as fobj:
        parser.feed(fobj)
    dftbinp = dictbuilder.hsddict

    return dftbinp


def get_dftb_parameters(skdir='./', driver=None, drivermaxforce=None,
                        drivermaxsteps=None, scc=False, scctol=None,
                        maxang=None, kpts=None):
    '''Generates a suitable Python dictionary for a
       calculation with the hamiltonian set to DFTB.

    Args:
        skdir (str):             path to Slater-Koster files
        driver (str):            DFTB+ driver for geometry optimization
        drivermaxforce (float):  max. force component as convergence
                                 criterion of geometry optimization
        drivermaxsteps (int):    max. number of geometry steps
        scc (bool):              True for self-consistent calculations
        scctol (float):          convergence criterion of SCC cycles
        maxang (dict):           max. angular momentum of atom types
        kpts (list/tuple):       K-points for Brillouin zone sampling
                                 (periodic structures)

    Returns:
        dftbinp (dict):          DFTB+ input as nested dictionaries
    '''

    dftbinp = {}

    if driver is not None:
        dftbinp['Driver'] = {}
        dftbinp['Driver'][str(driver)] = {}
        if drivermaxforce is not None:
            dftbinp['Driver'][str(driver)]['MaxForceComponent'] = drivermaxforce
        if drivermaxsteps is not None:
            dftbinp['Driver'][str(driver)]['MaxSteps'] = drivermaxsteps

    dftbinp['Hamiltonian'] = {}
    dftbinp['Hamiltonian']['DFTB'] = {}
    dftbinp['Hamiltonian']['DFTB']['Scc'] = 'Yes' if scc else 'No'

    if scctol is not None:
        dftbinp['Hamiltonian']['DFTB']['SccTolerance'] = scctol

    if maxang is not None:
        dftbinp['Hamiltonian']['DFTB']['MaxAngularMomentum'] = maxang
    else:
        dftbinp['Hamiltonian']['DFTB']['MaxAngularMomentum'] = {}

    if not skdir.endswith('/'):
        skdir += '/'

    skfiledict = {
        'Prefix': skdir,
        'Separator': '"-"',
        'Suffix': '".skf"'
        }

    dftbinp['Hamiltonian']['DFTB']['SlaterKosterFiles'] = {
        'Type2FileNames': skfiledict
        }

    # Handle different K-point formats
    # (note: the ability to handle bandpaths has not yet been implemented)
    if kpts is not None:
        if np.array(kpts).ndim == 1:
            # Case: K-points as (gamma-centered) Monkhorst-Pack grid
            mp_mesh = kpts
            offsets = [0.] * 3
            props = [elem for elem in mp_mesh if isinstance(elem, str)]
            props = [elem.lower() for elem in props]
            tgamma = 'gamma' in props and len(props) == 1
            if tgamma:
                mp_mesh = mp_mesh[:-1]
                eps = 1e-10
                for ii in range(3):
                    offsets[ii] *= mp_mesh[ii]
                    assert abs(offsets[ii]) < eps \
                        or abs(offsets[ii] - 0.5) < eps
                    if mp_mesh[ii] % 2 == 0:
                        offsets[ii] += 0.5
            elif not tgamma and len(mp_mesh) != 3:
                raise ValueError('Illegal K-Points definition: ' + str(kpts))
            kpts_mp = np.vstack((np.eye(3) * mp_mesh, offsets))
            dftbinp['Hamiltonian']['DFTB']['KPointsAndWeights'] = {
                'SupercellFolding': kpts_mp
                }
        elif np.array(kpts).ndim == 2:
            # Case: single K-points explicitly as (N x 3) or (N x 4) matrix
            kpts_coord = np.array(kpts)
            if np.shape(kpts_coord)[1] == 4:
                kptsweights = kpts_coord
                kpts_coord = kpts_coord[:, :-1]
            elif np.shape(kpts_coord)[1] == 3:
                kptsweights = np.hstack([kpts_coord, [[1.0],] * \
                                         np.shape(kpts_coord)[0]])
            else:
                raise ValueError('Illegal K-Points definition: ' + str(kpts))
            dftbinp['Hamiltonian']['DFTB']['KPointsAndWeights'] = kptsweights
        else:
            raise ValueError('Illegal K-Points definition: ' + str(kpts))

    dftbinp['Options'] = {
        'WriteResultsTag': 'Yes'
        }
    dftbinp['ParserOptions'] = {
        'IgnoreUnprocessedNodes': 'Yes'
        }

    dftbinp['Analysis'] = {}
    dftbinp['Analysis']['CalculateForces'] = 'No'

    return dftbinp


class DftbPlus(FileIOCalculator):
    '''ASE interface to DFTB+.

       A DFTB+ calculator with ase-FileIOCalculator nomenclature.
    '''

    implemented_properties = ('energy', 'forces', 'charges', 'stress')


    def __init__(self, binary='dftb+', parameters=None,
                 restart=None, ignore_bad_restart_file=False,
                 label='dftbplus', **kwargs):
        '''Initializes a ctypes DFTB+ calculator object.

        Args:
            parameters (dict): dictionary containing (additional)
                               initialization options for DFTB+
        '''

        self.input = copy.deepcopy(parameters)

        self._do_forces = False

        self._atoms = None
        self._pcpot = None
        self._lines = None
        self._symids = None
        self._atoms_input = None
        self._mmpositions = None

        self._skdir = self.input['Hamiltonian']['DFTB'] \
                             ['SlaterKosterFiles']['Type2FileNames']['Prefix']

        FileIOCalculator.__init__(self, restart, ignore_bad_restart_file,
                                  label, self._atoms, binary, **kwargs)

        # Determine number of spin channels
        try:
            spinpolkeys = list(self.input['Hamiltonian']\
                               ['DFTB']['SpinPolarisation'].keys())
            self.nspin = 2 if 'colinear' in spinpolkeys else 1
        except KeyError:
            self.nspin = 1


    def _write_dftb_in(self, filename):
        '''Dumps Python input dictionary to HSD format and writes input file.
           If necessary, geometry-relevant entries in the input are updated.
           If not all max. angular momenta have been specified by user, an
           attempt is made to extract remaining ones out of Slater-Koster files.

        Args:
            filename (str): name of input file to be written
        '''

        # Setting up the desired, geometry-relevant entries
        specieslist, self._symids = self._chemsym2indices()

        geodict = {}
        geodict['TypeNames'] = specieslist
        typesandcoords = np.empty([len(self._symids), 4], dtype=object)
        typesandcoords[:, 0] = self._symids
        typesandcoords[:, 1:] = np.array(self._atoms_input.get_positions(),
                                         dtype=float)
        geodict['TypesAndCoordinates'] = typesandcoords
        geodict['TypesAndCoordinates.attribute'] = 'Angstrom'

        periodic = any(self._atoms_input.get_pbc())

        if periodic:
            geodict['Periodic'] = 'Yes'
            geodict['LatticeVectors'] = self._atoms_input.get_cell()[:]
            geodict['LatticeVectors.attribute'] = 'Angstrom'
        else:
            geodict['Periodic'] = 'No'

        self.input['Geometry'] = geodict

        # Setting up max. angular momenta if not specified by user
        # extract maxang's out of Slater-Koster files
        speciesdiff = set(specieslist) - \
            set(self.input['Hamiltonian']['DFTB']['MaxAngularMomentum'].keys())
        maxangs = self.input['Hamiltonian']['DFTB']['MaxAngularMomentum']
        for species in speciesdiff:
            path = os.path.join(self._skdir, '{0}-{0}.skf'.format(species))
            maxang = read_max_angular_momentum(path)
            if maxang is None:
                msg = 'Error: Could not read max. angular momentum from ' + \
                      'Slater-Koster file "{0}-{0}.skf".'.format(species) + \
                      ' Please specify manually!'
                raise TypeError(msg)
            if maxang not in range(0, 4):
                msg = 'The obtained max. angular momentum from ' + \
                      'Slater-Koster file "{0}-{0}.skf"'.format(species) + \
                      ' is out of range. Please check!'
                raise ValueError(msg)
            maxang = '"{}"'.format('spdf'[maxang])
            maxangs[species] = maxang

        with open(filename, 'w') as outfile:
            outfile.write(hsd.dumps(self.input))


    def _chemsym2indices(self):
        '''Assigns indices to the chemical symbols of the Atoms object.

        Returns:
            specieslist (list): occurring atom types in Atoms object
            symids (list):      indices assigned to atom types in specieslist
        '''

        chemsyms = self._atoms_input.get_chemical_symbols()

        symdict = {}
        for sym in chemsyms:
            if not sym in symdict:
                symdict[sym] = len(symdict) + 1

        specieslist = list(['null'] * len(symdict.keys()))
        for sym in symdict:
            specieslist[symdict[sym] - 1] = sym

        symids = []
        for sym in chemsyms:
            symids.append(symdict[sym])

        return specieslist, symids


    def set(self, **kwargs):
        changed_parameters = FileIOCalculator.set(self, **kwargs)
        if changed_parameters:
            self.reset()
        return changed_parameters


    def check_state(self, atoms):
        system_changes = FileIOCalculator.check_state(self, atoms)
        # Ignore unit cell for molecules:
        if not atoms.pbc.any() and 'cell' in system_changes:
            system_changes.remove('cell')
        if self._pcpot and self._pcpot.mmpositions is not None:
            system_changes.append('positions')
        return system_changes


    def write_input(self, atoms, properties=None, system_changes=None):
        if properties is not None:
            if 'forces' in properties or 'stress' in properties:
                self._do_forces = True
                self.input['Analysis']['CalculateForces'] = 'Yes'
        FileIOCalculator.write_input(
            self, atoms, properties, system_changes)

        # self._atoms is none until results are read out,
        # then it is set to the ones at writing input
        self._atoms_input = atoms
        self._atoms = None

        self._write_dftb_in(os.path.join(self.directory, 'dftb_in.hsd'))

        write(os.path.join(self.directory, 'geo_end.gen'), atoms)

        if self._pcpot:
            self._pcpot.write_mmcharges('dftb_external_charges.dat')


    def read_results(self):
        ''' All results are read from results.tag file.
            It will be destroyed after it is read to avoid
            reading it once again after some runtime error.
        '''

        myfile = open(os.path.join(self.directory, 'results.tag'), 'r')
        self._lines = myfile.readlines()
        myfile.close()

        self._atoms = self._atoms_input
        charges, energy = self.read_charges_and_energy()
        if charges is not None:
            self.results['charges'] = charges
        self.results['energy'] = energy
        if self._do_forces:
            forces = self.read_forces()
            self.results['forces'] = forces
        self._mmpositions = None

        # stress stuff begins
        sstring = 'stress'
        have_stress = False
        stress = list()
        for iline, line in enumerate(self._lines):
            if sstring in line:
                have_stress = True
                start = iline + 1
                end = start + 3
                for i in range(start, end):
                    cell = [float(x) for x in self._lines[i].split()]
                    stress.append(cell)
        if have_stress:
            stress = -np.array(stress) * Hartree / Bohr**3
            self.results['stress'] = stress.flat[[0, 4, 8, 5, 2, 1]]
        # stress stuff ends

        # eigenvalues and fermi levels
        fermi_levels = self.read_fermi_levels()
        if fermi_levels is not None:
            self.results['fermi_levels'] = fermi_levels

        eigenvalues = self.read_eigenvalues()
        if eigenvalues is not None:
            self.results['eigenvalues'] = eigenvalues

        # calculation was carried out with atoms written in write_input
        os.remove(os.path.join(self.directory, 'results.tag'))


    def read_forces(self):
        '''Read forces from DFTB+ output file (results.tag).'''

        # Force line indexes
        for iline, line in enumerate(self._lines):
            fstring = 'forces   '
            if line.find(fstring) >= 0:
                index_force_begin = iline + 1
                line1 = line.replace(':', ',')
                index_force_end = iline + 1 + \
                    int(line1.split(',')[-1])
                break

        gradients = []
        for j in range(index_force_begin, index_force_end):
            word = self._lines[j].split()
            gradients.append([float(word[k]) for k in range(0, 3)])

        return np.array(gradients) * Hartree / Bohr


    def read_charges_and_energy(self):
        '''Get partial charges on atoms
           in case we cannot find charges they are set to None
        '''

        infile = open(os.path.join(self.directory, 'detailed.out'), 'r')
        lines = infile.readlines()
        infile.close()

        for line in lines:
            if line.strip().startswith('Total energy:'):
                energy = float(line.split()[2]) * Hartree
                break

        qm_charges = []
        for nn, line in enumerate(lines):
            if ('Atom' and 'Charge' in line):
                chargestart = nn + 1
                break
        else:
            # print('Warning: did not find DFTB-charges')
            # print('This is ok if flag SCC=No')
            return None, energy

        lines1 = lines[chargestart:(chargestart + len(self._atoms))]
        for line in lines1:
            qm_charges.append(float(line.split()[-1]))

        return np.array(qm_charges), energy


    def get_charges(self, atoms):
        '''Get the calculated charges.
           This is inhereted to atoms object.
        '''

        if 'charges' in self.results:
            return self.results['charges']

        return None


    def read_eigenvalues(self):
        '''Read Eigenvalues from DFTB+ output file (results.tag).
           Unfortunately, the order seems to be scrambled.
        '''

        # Eigenvalue line indexes
        index_eig_begin = None
        for iline, line in enumerate(self._lines):
            fstring = 'eigenvalues   '
            if line.find(fstring) >= 0:
                index_eig_begin = iline + 1
                line1 = line.replace(':', ',')
                ncol, nband, nkpt, nspin = map(int, line1.split(',')[-4:])
                break
        else:
            return None

        # Take into account that the last row may lack
        # columns if nkpt * nspin * nband % ncol != 0
        nrow = int(np.ceil(nkpt * nspin * nband * 1. / ncol))
        index_eig_end = index_eig_begin + nrow
        ncol_last = len(self._lines[index_eig_end - 1].split())
        self._lines[index_eig_end - 1] += ' 0.0 ' * (ncol - ncol_last)

        eig = np.loadtxt(self._lines[index_eig_begin:index_eig_end]).flatten()
        eig *= Hartree
        nn = nkpt * nband
        eigenvalues = [eig[i * nn:(i + 1) * nn].reshape((nkpt, nband))
                       for i in range(nspin)]

        return eigenvalues


    def read_fermi_levels(self):
        '''Read Fermi level(s) from DFTB+ output file (results.tag).'''

        # Fermi level line indexes
        for iline, line in enumerate(self._lines):
            fstring = 'fermi_level   '
            if line.find(fstring) >= 0:
                index_fermi = iline + 1
                break
        else:
            return None

        fermi_levels = []
        words = self._lines[index_fermi].split()
        assert len(words) == 2

        for word in words:
            ee = float(word)
            if abs(ee) > 1e-08:
                # Without spin polarization, one of the Fermi
                # levels is equal to 0.000000000000000E+000
                fermi_levels.append(ee)

        return np.array(fermi_levels) * Hartree


    def get_number_of_spins(self):
        '''Auxiliary function to extract result: number of spins.'''

        return self.nspin


    # !!!BROKEN!!!
    def get_ibz_k_points(self):
        '''Auxiliary function to extract: K-points.'''

        return self.kpts_coord.copy()


    # !!!BROKEN!!!
    def get_eigenvalues(self, kpt=0, spin=0):
        '''Auxiliary function to extract result: eigenvalues.'''

        return self.results['eigenvalues'][spin][kpt].copy()


    # !!!BROKEN!!!
    def get_fermi_levels(self):
        '''Auxiliary function to extract result: Fermi level(s).'''

        return self.results['fermi_levels'].copy()


    # !!!BROKEN!!!
    def get_fermi_level(self):
        '''Auxiliary function to extract result: result: max. Fermi level.'''

        return max(self.get_fermi_levels())


def read_max_angular_momentum(path):
    '''Read maximum angular momentum from .skf file.
       See dftb.org/fileadmin/DFTB/public/misc/slakoformat.pdf
       for a detailed description of the Slater-Koster file format.

    Args:
        path (str):   path to Slater-Koster file

    Returns:
        maxang (int): max. angular momentum (None if extraction fails)
    '''

    with open(path, 'r') as skf:
        line = skf.readline()
        if line[0] == '@':
            # Skip additional line for extended format
            line = skf.readline()

    # Replace any commas that may appear
    # (inconsistency in .skf files)
    line = line.replace(',', ' ').split()

    if len(line) == 3:
        # max. angular momentum specified:
        # extraction possible
        maxang = int(line[2]) - 1
    else:
        # max. angular momentum not specified
        # or wrong format: extraction not possible
        maxang = None

    return maxang


#    def embed(self, mmcharges=None, directory='./'):
#        '''Embed atoms in point-charges (mmcharges).'''
#
#        self._pcpot = PointChargePotential(mmcharges, self.directory)
#        return self._pcpot


#class PointChargePotential:
#    def __init__(self, mmcharges, directory='./'):
#        '''Point-charge potential for DFTB+.'''
#
#        self.mmcharges = mmcharges
#        self.directory = directory
#        self._mmpositions = None
#        self.mmforces = None
#
#
#    def set_positions(self, mmpositions):
#        self._mmpositions = mmpositions
#
#
#    def set_charges(self, mmcharges):
#        self.mmcharges = mmcharges
#
#
#    def write_mmcharges(self, filename='dftb_external_charges.dat'):
#        '''mok all
#           Write external charges as monopoles for DFTB+.
#        '''
#
#        if self.mmcharges is None:
#            print('DFTB: Warning: not writing external charges ')
#            return
#        charge_file = open(os.path.join(self.directory, filename), 'w')
#        for [pos, charge] in zip(self._mmpositions, self.mmcharges):
#            [x, y, z] = pos
#            charge_file.write('%12.6f %12.6f %12.6f %12.6f \n'
#                              % (x, y, z, charge))
#        charge_file.close()
#
#
#    def get_forces(self, calc, get_forces=True):
#        '''Returns forces on point charges if the flag get_forces=True.'''
#
#        if get_forces:
#            return self.read_forces_on_pointcharges()
#        else:
#            return np.zeros_like(self._mmpositions)
#
#
#    def read_forces_on_pointcharges(self):
#        '''Read forces from DFTB+ output file (detailed.out).'''
#
#        from ase.units import Hartree, Bohr
#        infile = open(os.path.join(self.directory, 'detailed.out'), 'r')
#        lines = infile.readlines()
#        infile.close()
#
#        external_forces = []
#        for n, line in enumerate(lines):
#            if 'Forces on external charges' in line:
#                chargestart = n + 1
#                break
#        else:
#            raise RuntimeError(
#                'Problem in reading forces on MM external-charges')
#        lines1 = lines[chargestart:(chargestart + len(self.mmcharges))]
#        for line in lines1:
#            external_forces.append(
#                [float(i) for i in line.split()])
#        return np.array(external_forces) * Hartree / Bohr

