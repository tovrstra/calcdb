#!/usr/bin/env python
"""Tools for making and using HDF5 databases of calculations

   This python file can be used in two ways: (i) as a main program to create the
   interaction database or (ii) as a python module to interact with the
   database.

   Pattern documentation
   ---------------------

   Several functions below support a ``pattern`` argument. This is either a
   string with a pattern supported by the fnmatch module, or a numpy array with
   integer indexes refering to individual xmers.
"""

print 'Importing stuff'

from bisect import bisect_left
from collections import namedtuple
from fnmatch import fnmatch
from glob import glob
import json
import os
import re

import h5py as h5
import numpy as np

from horton import IOData, log, FCHKFile


log.set_level(log.silent)


__all__ = ['CalcDB', 'TXTFieldInfo', 'JSONFieldInfo']


Molecule = namedtuple('Molecule', 'name nfrag begin end')


class CalcDB(object):
    def __init__(self, fnh5, root=None, frag_path=None, report_missing=True):
        """Initialize a calculation database

        Only the first argument is needed to use the database. The rest is also needed
        when storing data in the database.


        Parameters
        ----------
        fnh5 : str
               The name of the HDF5 file.
        root : str
               The root directory where the calculations and the database are stored.
        frag_path : str
                   Relative path to the fragment calculations (contains on time '%i').
        report_missing : boolean
                         When True, every name is printed for which no data is provided.
        """
        self.fnh5 = fnh5
        self.root = root
        self.frag_path = frag_path
        self.report_missing = report_missing

        with h5.File(self.fnh5, 'r') as f:
            self.mols = []
            self._names = f['geometries/names'][:]  # only to be used by lookup
            begin = 0
            for name, nfrag in zip(self._names, f['geometries/nfrags'][:]):
                end = begin + nfrag
                self.mols.append(Molecule(name, nfrag, begin, end))
                begin = end
        print 'Number of molecules', len(self.mols)
        print 'Number of fragments', sum(mol.nfrag for mol in self.mols)

    @classmethod
    def from_scratch(cls, fnh5, root, patterns, frag_path=None, report_missing=True):
        """Initialize a calculation database from scratch

        Parameters
        ----------
        fnh5 : str
               The (base)name of the HDF5 file.
        root : str
               The root directory where the calculations and the database are stored.
        patterns : list of str
                   List of fnmatch strings with directories containint calculations
        frag_path : str
                   Relative path to the fragment calculations (contains on time '%i').
        report_missing : boolean
                         When True, every name is printed for which no data is provided.
        """
        if not os.path.isfile(fnh5):
            print 'Looking up all directories (slow)'
            mols = []
            for pattern in patterns:
                names = [match[len(root)+1:] for match in glob(os.path.join(root, pattern))]
                for name in names:
                    if frag_path is None:
                        nfrag = 0
                    else:
                        nfrag = 0
                        frag_dirnames = []
                        while True:
                            frag_dirname = os.path.join(root, name, frag_path % nfrag)
                            if os.path.isdir(frag_dirname):
                                frag_dirnames.append(frag_dirname)
                            else:
                                break
                            nfrag += 1
                        nfrag = len(frag_dirnames)
                    mols.append(Molecule(name, nfrag, None, None))
            mols.sort()
            with h5.File(fnh5) as f:
                f['geometries/names'] = np.array([mol.name for mol in mols])
                f['geometries/nfrags'] = np.array([mol.nfrag for mol in mols])
                frag_ranges = []
                begin = 0
                for mol in mols:
                    end = begin + mol.nfrag
                    frag_ranges.append([begin, end])
                    begin = end
                f['geometries/frag_ranges'] = np.array(frag_ranges)
        return cls(fnh5, root, frag_path, report_missing)

    def select(self, pattern):
        """Find all the molecules that match the given pattern

        Parameters
        ----------
        pattern : str
                  See general pattern documentation above.
        """
        if isinstance(pattern, int) or isinstance(pattern, np.ndarray):
            return pattern
        else:
            indexes = []
            for i, mol in enumerate(self.mols):
                if fnmatch(mol.name, pattern):
                    indexes.append(i)
            return np.array(indexes)

    def lookup(self, name, ifrag):
        """Look up the index of a specific name.

        Parameters
        ----------
        name : str
                   The complete name to be looked up
        ifrag : int or None
                When provided, the index of a fragment is returned
        """
        index = bisect_left(self._names, name)
        if index != len(self._names) and self._names[index] == name:
            if ifrag is None:
                return index
            else:
                return self.mols[index].begin + ifrag
        raise ValueError('Name not found: %s' % name)

    def load_atom_data(self, source, indexes):
        """Load per-atom data for all molecules that match pattern

        Parameters
        ----------
        source : str
                 The path to the HDF5 dataset with data for all molecules.
        indexes : int or list of ints
                  The molecule index(es) to be loaded
        """
        with h5.File(self.fnh5, 'r') as f:
            assert f[source].shape[0] == f['geometries/atom_ranges'][-1,1]
            if isinstance(indexes, int):
                begin, end = f['geometries/atom_ranges'][indexes]
                return f[source][begin:end]
            else:
                result = []
                for index in indexes:
                    begin, end = f['geometries/atom_ranges'][index]
                    result.append(f[source][begin:end])
                return result

    def load_frag_data(self, source, indexes):
        with h5.File(self.fnh5, 'r') as f:
            assert f[source].shape[0] == f['geometries/frag_ranges'][-1,1]
            if isinstance(indexes, int):
                begin, end = f['geometries/frag_ranges'][indexes]
                return f[source][begin:end]
            else:
                result = []
                for index in indexes:
                    begin, end = f['geometries/frag_ranges'][index]
                    result.append(f[source][begin:end])
                return result

    def load_mol_data(self, source, indexes):
        """Load per-molecule data for all molecules that match pattern

        Parameters
        ----------
        source : str
                 The path to the HDF5 dataset with data for all molecules.
        indexes : int or list of ints
                  The molecule index(es) to be loaded
        """
        with h5.File(self.fnh5, 'r') as f:
            assert f[source].shape[0] == f['geometries/names'].shape[0]
            if isinstance(indexes, int):
                return f[source][indexes]
            else:
                result = []
                for index in indexes:
                    result.append(f[source][index])
                return np.array(result)

    def store_data(self, destination, data, shape, kind, dtype, do_frag):
        """Generic function to store per-atom data in the HDF5 file

        If some data was already present in the destination, it will be lost!

        Parameters
        ----------
        destination : str
                      The name of the dataset where the results will be stored.
        data : list of tuples (name, np.ndarray) or (name, float)
               A list of (name, data array) pairs, one array for each atom.
        shape : tuple of ints
                The shape of the data for one atom.
        kind : str
               'atom' or 'mol'
        dtype : np.dtype
                Array data type
        do_frag : bool
                  When True, data for fragments must be stored
        """
        # Prepare the array to be stored.
        ranges = None
        if kind == 'atom':
            with h5.File(self.fnh5, 'r') as f:
                if do_frag:
                    ranges = f['frag/geometries/atom_ranges'][:]
                else:
                    ranges = f['geometries/atom_ranges'][:]
            ntotal = ranges[-1,1]
        elif kind == 'mol':
            if do_frag:
                ntotal = sum(mol.nfrag for mol in self.mols)
            else:
                ntotal = len(self.mols)
        all_data_array = np.empty((ntotal,) + shape, dtype=dtype)
        if issubclass(dtype, float):
            all_data_array.fill(np.nan)
        else:
            all_data_array.fill(-1)

        # Prepare data to count missing pieces of information
        nfound = 0
        missing = set([])
        for mol in self.mols:
            if do_frag:
                for ifrag in xrange(mol.nfrag):
                    missing.add((mol.name, ifrag))
            else:
                missing.add((mol.name, None))

        # Go through all the data and store it in the right place in the array.
        for name, ifrag, data_array in data:
            if data_array is None:
                continue
            assert np.isfinite(data_array).all()
            ibig = self.lookup(name, ifrag)
            if kind == 'atom':
                if data_array.shape[1:] != shape:
                    raise TypeError('Shape mismatch for %s:%s. Got %s while expecting %s.' %
                                    (name, ifrag, data_array.shape[1:], shape))
                begin, end = ranges[ibig]
                if end - begin != len(data_array):
                    raise TypeError('Shape mismatch for %s:%s. Got %i atoms while expecting %i.' %
                                    (name, ifrag, len(data_array), end-begin))
                all_data_array[begin:end] = data_array
                nfound += end - begin
            elif kind == 'mol':
                all_data_array[ibig] = data_array
                nfound += 1
            else:
                raise ValueError('Uknown kind: %s' % kind)
            missing.discard((name, ifrag))

        # Add prefix in case of fragment data
        if do_frag:
            destination = 'frag/%s' % destination

        # Check the completness of the data
        fraction = float(nfound)/ntotal
        print '%50s: %7i / %7i   (%.0f%%)  %4s  frag=%s type=%s' % (
            destination, nfound, ntotal, fraction*100, kind, do_frag, dtype.__name__)
        if self.report_missing:
            for name, ifrag in sorted(missing):
                print 'Missing %70s  %4s  |  %30s' % (name, ifrag, destination)

        # Store it in the HDF5 file, only if some data was read
        if nfound > 0:
            with h5.File(self.fnh5) as f:
                if destination in f:
                    f[destination][:] = all_data_array
                else:
                    f[destination] = all_data_array

    def store_driver(self, basename, fields, do_frags=False):
        """Driver routine for loading stuff from a file"""
        data = dict((info.destination, []) for info in fields.info)
        for mol in self.mols:
            if do_frags:
                for ifrag in xrange(mol.nfrag):
                    path = os.path.join(self.root, mol.name, self.frag_path % ifrag, basename)
                    values = fields.read(path)
                    for info, value in zip(fields.info, values):
                        data[info.destination].append((mol.name, ifrag, value))
            else:
                path = os.path.join(self.root, mol.name, basename)
                values = fields.read(path)
                for info, value in zip(fields.info, values):
                    data[info.destination].append((mol.name, None, value))
        for info in fields.info:
            self.store_data(info.destination, data[info.destination], info.shape,
                            info.kind, info.dtype, do_frags)

    def store_geometries(self, basename, do_frag=False):
        """Store the geometries in the database"""
        print 'Storing geometries'
        self.store_driver(basename, XYZFields(do_frag), do_frag)

    def store_gaussian_fchk(self, basename='gaussian.fchk', do_frag=False):
        """Load all the useful stuff from a Gaussian FCHK file"""
        print 'Storing Gaussian FCHK files:', basename
        self.store_driver(basename, GaussianFCHKFields(), do_frag)

    def store_wpart(self, basename, scheme, do_frag=False):
        print 'Storing WPart output:', basename
        self.store_driver(basename, WPartFields(scheme), do_frag)

    def store_txt(self, basename, info):
        print 'Storing TXT:', basename
        self.store_driver(basename, TXTFields(info))

    def store_json(self, basename, info):
        print 'Storing JSON:', basename
        self.store_driver(basename, JSONFields(info))


FieldInfo = namedtuple('FieldInfo', 'destination shape kind dtype')


class Fields(object):
    def __init__(self, info):
        self.info = info

    def read(self, path):
        raise NotImplementedError


class XYZFields(Fields):
    def __init__(self, do_frag=False):
        Fields.__init__(self, [
            FieldInfo('geometries/atom_ranges', (2,), 'mol', int),
            FieldInfo('geometries/numbers', (), 'atom', int),
            FieldInfo('geometries/coordinates', (3,), 'atom', float),
        ])
        self.begin = 0

    def read(self, path):
        mol = IOData.from_file(path)
        end = self.begin + mol.natom
        result = [[self.begin, end], mol.numbers, mol.coordinates]
        self.begin = end
        return result


GaussianFCHKFieldInfo = namedtuple('FieldInfo', 'destination shape kind dtype fchk_name')


class GaussianFCHKFields(Fields):
    def __init__(self):
        Fields.__init__(self, [
            GaussianFCHKFieldInfo('estruct/mol_charges', (), 'mol', int, 'Charge'),
            GaussianFCHKFieldInfo('estruct/mol_dipoles', (3,), 'mol', float, 'Dipole Moment'),
            GaussianFCHKFieldInfo('estruct/atom_charges/mulliken', (), 'atom', float, 'Mulliken Charges'),
            GaussianFCHKFieldInfo('estruct/eff_core_charges', (), 'atom', float, 'Nuclear charges'),
        ])

    def read(self, path):
        fchk_names = [info.fchk_name for info in self.info]
        fchk = FCHKFile(path, fchk_names)
        result = []
        for fchk_name in fchk_names:
            result.append(fchk[fchk_name])
        # Filter out ghost atoms
        mask = result[-1] > 0
        result[2] = result[2][mask]
        result[3] = result[3][mask]
        return result


class WPartFields(Fields):
    def __init__(self, scheme):
        Fields.__init__(self, [
            FieldInfo('estruct/atom_charges/%s' % scheme, (), 'atom', float),
            FieldInfo('estruct/valence_charges/%s' % scheme, (), 'atom', float),
            FieldInfo('estruct/valence_widths/%s' % scheme, (), 'atom', float),
            FieldInfo('estruct/core_charges/%s' % scheme, (), 'atom', float),
        ])

    def read(self, path):
        with h5.File(path, 'r') as f:
            return [
                f['charges'][:],
                f['valence_charges'][:] if 'valence_charges' in f else None,
                f['valence_widths'][:] if 'valence_widths' in f else None,
                f['core_charges'][:] if 'core_charges' in f else None,
            ]


class TXTFieldInfo(object):
    def __init__(self, destination, shape, kind, dtype, line, restr):
        self.destination = destination
        self.shape = shape
        self.kind = kind
        self.dtype = dtype
        self.line = line
        self.restr = restr
        self.re = re.compile(restr)


class TXTFields(Fields):
    def read(self, path):
        with open(path) as f:
            values = [None]*len(self.info)
            for iline, line in enumerate(f):
                for iinfo, info in enumerate(self.info):
                    if info.line == iline:
                        m = info.re.match(line)
                        values[iinfo] = info.dtype(m.group(1))
            return values


JSONFieldInfo = namedtuple('JSONFieldInfo', 'destination shape kind dtype json_name')


class JSONFields(Fields):
    def read(self, path):
        values = []
        with open(path) as f:
            data = json.load(f)
        for info in self.info:
            values.append(data[info.json_name])
        return values
