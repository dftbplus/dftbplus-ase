#------------------------------------------------------------------------------#
#  DFTB+: general package for performing fast atomistic simulations            #
#  Copyright (C) 2006 - 2022  DFTB+ developers group                           #
#                                                                              #
#  See the LICENSE file for terms of usage and distribution.                   #
#------------------------------------------------------------------------------#

import numpy as np
from ase.atoms import Atoms
from dftbplus_ptools.xyz import Xyz
from dftbplus_ptools.gen import Gen
from dftbplus_ptools.geometry import Geometry
from dftbplus_ptools.hsdinput import Hsdinput


def read_dftb(filename):
    """Method to read coordinates from the Geometry section
    of a DFTB+ input file (typically called "dftb_in.hsd").

    As described in the DFTB+ manual, this section can be
    in a number of different formats. This reader supports
    the GEN format, the so-called "explicit" format and the XYZ format.

    The "explicit" format is unique to DFTB+ input files.
    The GEN format can also be used in a stand-alone fashion,
    as coordinate files with a `.gen` extension. Reading and
    writing such files is implemented in `ase.io.gen`.
    """

    try:
        xyz = Xyz.fromhsd(filename)
        geo = xyz.geometry
    except KeyError:
        try:
            gen = Gen.fromhsd(filename)
            geo = gen.geometry
        except KeyError:
            raise NotImplementedError(f"The Geometry scetion in '{filename}'" +
                                      " is not supported!")

    return geometry_to_atom(geo)


def atom_to_geometry(atom):
    """converts atom object to dftbplus_ptools.geometry.Geometry object

    Args:
        atom (ase atoms object): atom to be converted

    Returns:
        (dftbplus_ptools.geometry.Geometry object): converted geometry
    """
    chemsyms = atom.get_chemical_symbols()

    symdict = {}
    for sym in chemsyms:
        if sym not in symdict:
            symdict[sym] = len(symdict)

    specienames = list(['null'] * len(symdict.keys()))
    for sym, num in symdict.items():
        specienames[num] = sym

    indexes = []
    for sym in chemsyms:
        indexes.append(symdict[sym])
    coords = atom.get_positions()

    if any(atom.get_pbc()) and all(atom.get_pbc()):
        latvecs = atom.get_cell().array
    elif not any(atom.get_pbc()) and not all(atom.get_pbc()):
        latvecs = None
    else:
        raise NotImplementedError("The converted Atoms object contains " +
                                  "1D or 2D PBC's. The geometry object " +
                                  "only supports 3D PBC's!")

    return Geometry(specienames, indexes, coords, latvecs=latvecs)


def geometry_to_atom(geometry):
    """converts dftbplus_ptools.geometry.Geometry object to atom object

    Args:
        geometry (dftbplus_ptools.geometry.Geometry object): geometry to be
            converted

    Returns:
        (ase atoms object): atom with geometry
    """
    specienames = geometry.specienames
    indexes = geometry.indexes
    atoms_pos = geometry.coords
    my_pbc = geometry.periodic

    atom_symbols = []
    for index in indexes:
        atom_symbols.append(specienames[index])

    if my_pbc:
        mycell = geometry.latvecs
    elif not my_pbc:
        mycell = np.zeros((3, 3))

    return Atoms(positions=atoms_pos, symbols=atom_symbols, cell=mycell,
                 pbc=my_pbc)


def atom_to_hsd(atom, filename="dftb_in.hsd", directory=".", dictionary=None,
                get_class=False):
    """help function for reading geometry from atom and adding to hsd

    Args:
        atom (ase atoms object): atom with geometry
        directory (str): directory of file
        filename (str): name of file
        dictionary (dict): option to change existing dict
        get_class (bool): True if dftbplus_ptools Hsdinput class is returned,
            False if dict is returned

    Returns:
        (dftbplus_ptools Hsdinput class/dict): Hsdinput class/dict with
            geometry added
    """
    geo = atom_to_geometry(atom)
    hsd = Hsdinput(filename, directory, dictionary)
    hsd.set_geometry(geo)
    if get_class:
        return hsd
    return hsd.get_hsd()
