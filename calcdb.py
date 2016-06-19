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


__all__ = [
    'CalcDB',
    'FieldInfo', 'Fields', 'GaussianFCHKFields',
    'HDF5FieldInfo', 'HDF5Fields', 'HDF5AtomChargeFields',
    'TXTFieldInfo', 'TXTFields',
    'cp2k_ddap_charges', 'cp2k_lowdin_charges', 'cp2k_mulliken_charges',
    'cp2k_resp_charges', 'cp2k_mol_population',
    'JSONFieldInfo', 'JSONFields',
]


Case = namedtuple('Case', 'name nfrag begin end')


def _lookup_cases(root, patterns, convert_to_frag=None):
    """Find all systems and their fragments by scanning directories

    Parameters
    ----------
    root : str
           The root directory where the calculations and the database are stored.
    patterns : list of str
               List of fnmatch strings with directories containint calculations
    convert_to_frag : function
                      Converts a full path to a fragment path.
    """
    print 'Looking up all directories (slow)'
    full_cases = []
    frag_cases = []
    for pattern in patterns:
        full_names = [match[len(root)+1:] for match in glob(os.path.join(root, pattern))]
        for full_name in full_names:
            if convert_to_frag is None:
                nfrag = 0
            else:
                nfrag = 0
                while True:
                    frag_name = convert_to_frag(full_name, nfrag)
                    frag_dirname = os.path.join(root, frag_name)
                    if not os.path.isdir(frag_dirname):
                        break
                    frag_cases.append(Case(frag_name, None, None, None))
                    nfrag += 1
            full_cases.append(Case(full_name, nfrag, None, None))
    full_cases.sort()
    frag_cases.sort()
    return full_cases, frag_cases


def _store_cases(g, cases):
    """Store a set of cases in a HDF5 group (initialization)

    Parameters
    ----------
    g : h5.Group
        The group where the cases are stored
    cases : list
            A list of Case instances
    """
    g['geometries/names'] = np.array([case.name for case in cases])
    g['geometries/names'].attrs['kind'] = 'mol'
    if any(case.nfrag is None for case in cases):
        return
    g['geometries/nfrags'] = np.array([case.nfrag for case in cases])
    g['geometries/nfrags'].attrs['kind'] = 'mol'
    frag_ranges = []
    begin = 0
    for case in cases:
        end = begin + case.nfrag
        frag_ranges.append([begin, end])
        begin = end
    g['geometries/frag_ranges'] = np.array(frag_ranges)
    g['geometries/frag_ranges'].attrs['kind'] = 'mol'



class CalcDB(object):
    def __init__(self, fnh5, root=None, report_missing=True):
        """Initialize a calculation database

        Only the first argument is needed to use the database. The rest is also needed
        when storing data in the database.


        Parameters
        ----------
        fnh5 : str
               The name of the HDF5 file.
        root : str
               The root directory where the calculations and the database are stored.
        report_missing : boolean
                         When True, every name is printed for which no data is provided.
        """
        self.fnh5 = fnh5
        self.root = root
        self.report_missing = report_missing

        with h5.File(self.fnh5, 'r') as f:
            self.full_cases = []
            self.frag_cases = []
            self._full_names = f['full/geometries/names'][:]  # only to be used by lookup
            self._frag_names = f['frag/geometries/names'][:]  # only to be used by lookup
            begin = 0
            for name, nfrag in zip(self._full_names, f['full/geometries/nfrags'][:]):
                end = begin + nfrag
                self.full_cases.append(Case(name, nfrag, begin, end))
                begin = end
            for name in self._frag_names:
                self.frag_cases.append(Case(name, None, None, None))
        print 'Number of cases', len(self.full_cases)
        print 'Number of fragments', len(self.frag_cases)

    @classmethod
    def from_scratch(cls, fnh5, root, patterns, convert_to_frag=None, report_missing=True):
        """Initialize a calculation database from scratch

        Parameters
        ----------
        fnh5 : str
               The (base)name of the HDF5 file.
        root : str
               The root directory where the calculations and the database are stored.
        patterns : list of str
                   List of fnmatch strings with directories containint calculations
        convert_to_frag : function
                          Converts a full path to a fragment path.
        report_missing : boolean
                         When True, every name is printed for which no data is provided.
        """
        if not os.path.isfile(fnh5):
            full_cases, frag_cases = _lookup_cases(root, patterns, convert_to_frag)
            with h5.File(fnh5) as f:
                _store_cases(f.create_group('full'), full_cases)
                _store_cases(f.create_group('frag'), frag_cases)
        return cls(fnh5, root, report_missing)

    def __contains__(self, h5_path):
        """Test if a path is present in the HDF5 file"""
        with h5.File(self.fnh5, 'r') as f:
            return h5_path in f

    def select(self, pattern, do_frag=False):
        """Find all the cases that match the given pattern.

        Parameters
        ----------
        pattern : str
                  See general pattern documentation above.
        do_frag : bool
                  When True, fragment names are selected.
        """
        cases = self.frag_cases if do_frag else self.full_cases
        indexes = []
        for i, case in enumerate(cases):
            if fnmatch(case.name, pattern):
                indexes.append(i)
        return np.array(indexes)

    def lookup(self, name, do_frag=False, get_frag=False):
        """Look up the index of a specific name.

        Parameters
        ----------
        name : str
               The complete name to be looked up
        do_frag : bool
                  When True, a fragment name is looked up.
        get_frag : bool
                   When True, fragment indexes are returned. Not compatible with do_frag.
        """
        names = self._frag_names if do_frag else self._full_names
        index = bisect_left(names, name)
        if index != len(names) and names[index] == name:
            if get_frag:
                assert not do_frag
                return range(self.full_cases[index].begin, self.full_cases[index].end)
            else:
                return index
        raise ValueError('Name not found: %s' % name)

    def load_data(self, source, indexes, do_frag=False):
        """Load data for all cases listed in in indexes

        Parameters
        ----------
        source : str
                 The path to the HDF5 dataset with data for all cases.
        indexes : int or list of ints
                  The cases index(es) to be loaded.
        do_frag : bool
                  Treat the data as fragment data.
        """
        with h5.File(self.fnh5, 'r') as f:
            if f[source].attrs['kind'] == 'atom':
                if do_frag:
                    atom_ranges = f['frag/geometries/atom_ranges']
                else:
                    atom_ranges = f['full/geometries/atom_ranges']
                assert f[source].shape[0] == atom_ranges[-1,1]
                if isinstance(indexes, int):
                    begin, end = atom_ranges[indexes]
                    return f[source][begin:end]
                else:
                    result = []
                    for index in indexes:
                        begin, end = atom_ranges[index]
                        result.append(f[source][begin:end])
                    return result
            elif f[source].attrs['kind'] == 'mol':
                if do_frag:
                    assert f[source].shape[0] == f['frag/geometries/names'].shape[0]
                else:
                    assert f[source].shape[0] == f['full/geometries/names'].shape[0]
                if isinstance(indexes, int):
                    return f[source][indexes]
                else:
                    result = []
                    for index in indexes:
                        result.append(f[source][index])
                    return np.array(result)
            else:
                raise TypeError('Uknown data kind: %s' % f[source].attrs['kind'])

    def store_data(self, destination, data, shape, kind, dtype, do_frag):
        """Generic function to store data in the HDF5 file.

        If some data was already present in the destination, it will be overwritten and
        permanently lost!

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
        # Select the relevant cases
        cases = self.frag_cases if do_frag else self.full_cases

        # Prepare counters for missing pieces of information
        nfound = 0
        missing = set([case.name for case in cases])

        # Prepare ranges and number of data points
        if kind == 'atom':
            with h5.File(self.fnh5, 'r') as f:
                if do_frag:
                    ranges = f['frag/geometries/atom_ranges'][:]
                else:
                    ranges = f['full/geometries/atom_ranges'][:]
            ntotal = ranges[-1,1]
        elif kind == 'mol':
            ranges = None
            ntotal = len(cases)
        else:
            raise ValueError('The argument kind should be atom or frag.')

        # Prepare the array were all data will be collected.
        all_data_array = np.empty((ntotal,) + shape, dtype=dtype)
        if issubclass(dtype, float):
            all_data_array.fill(np.nan)
        else:
            all_data_array.fill(-1)

        # Go through all the data and store it in the right place in the array.
        for name, data_array in data:
            if data_array is None:
                continue
            assert np.isfinite(data_array).all()
            ibig = self.lookup(name, do_frag)
            if kind == 'mol':
                all_data_array[ibig] = data_array
                nfound += 1
            elif kind == 'atom':
                if data_array.shape[1:] != shape:
                    raise TypeError('Shape mismatch for %s:%s. Got %s while expecting %s.' %
                                    (name, ifrag, data_array.shape[1:], shape))
                begin, end = ranges[ibig]
                if end - begin != len(data_array):
                    raise TypeError('Shape mismatch for %s:%s. Got %i %ss while expecting %i.' %
                                    (name, ifrag, len(data_array), kind, end-begin))
                all_data_array[begin:end] = data_array
                nfound += end - begin
            else:
                raise ValueError('Unknown kind: %s' % kind)
            missing.discard(name)

        # Add prefix in case of fragment data
        if do_frag:
            destination = 'frag/%s' % destination
        else:
            destination = 'full/%s' % destination

        # Check the completness of the data
        fraction = float(nfound)/ntotal
        print '    Storing %.0f%% of %s (%i/%i): kind=%s, shape=%s, type=%s, do_frag=%s' % (
            fraction*100, destination, nfound, ntotal, kind, shape, dtype.__name__, do_frag)
        if self.report_missing and nfound > 0:
            for name in sorted(missing):
                print '        Missing', os.path.join(self.root, name)

        # Store it in the HDF5 file, only if some data was read
        if nfound > 0:
            with h5.File(self.fnh5) as f:
                if destination in f:
                    f[destination][:] = all_data_array
                else:
                    f[destination] = all_data_array
                f[destination].attrs['kind'] = kind

    def store_fields(self, basename, fields, do_frag=False):
        """Driver routine for loading stuff from a file and storing it in the databse.

        Parameters
        ----------
        basename : str
                   The basename of the file to load the data from.
        fields : Fields
                 An object that can read specific fields from a data file.
        do_frag : bool
                  If True, fragment data will be loaded instead of the full system data.
        """
        print 'Loading from %s with %s (do_frag=%s)' % (basename, fields.__class__.__name__, do_frag)
        # All data will be collected here
        data = dict((info.destination, []) for info in fields.infos)

        # Loop over all cases (and fragments)
        cases = self.frag_cases if do_frag else self.full_cases
        for case in cases:
            path = os.path.join(self.root, case.name, basename)
            if os.path.isfile(path):
                values = fields.read(path)
            else:
                values = [None]*len(fields.infos)
            for info, value in zip(fields.infos, values):
                data[info.destination].append((case.name, value))

        # Call lower-level store_data
        for info in fields.infos:
            # In case of the frag kind, we have to transform it into mol kind for fragment
            # data.
            if info.kind == 'frag':
                assert not do_frag
                kind = 'mol'
                my_do_frag = True
                data_list = []
                for full_name, frag_values in data[info.destination]:
                    frag_indexes = self.lookup(full_name, get_frag=True)
                    if frag_values is None:
                        for frag_index in frag_indexes:
                            data_list.append((self.frag_cases[frag_index].name, None))
                    else:
                        for frag_index, frag_value in zip(frag_indexes, frag_values):
                            data_list.append((self.frag_cases[frag_index].name, frag_value))
            else:
                kind = info.kind
                my_do_frag = do_frag
                data_list = data[info.destination]

            # Finally store it
            self.store_data(info.destination, data_list, info.shape, kind, info.dtype, my_do_frag)
        print

    def store_geometries(self, basename, do_frag=False):
        """Store the geometries in the database."""
        self.store_fields(basename, XYZFields(do_frag), do_frag)


FieldInfo = namedtuple('FieldInfo', 'destination shape kind dtype')


class Fields(object):
    def __init__(self, infos):
        self.infos = infos

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
    def __init__(self, prefix='gaussian'):
        self.prefix = prefix
        Fields.__init__(self, [
            GaussianFCHKFieldInfo('estruct/atom_charges/%s_mulliken' % prefix, (), 'atom', float, 'Mulliken Charges'),
            GaussianFCHKFieldInfo('estruct/eff_core_charges/%s' % prefix, (), 'atom', float, 'Nuclear charges'),
            GaussianFCHKFieldInfo('estruct/mol_charges/%s' % prefix, (), 'mol', int, 'Charge'),
            GaussianFCHKFieldInfo('estruct/mol_dipoles/%s' % prefix, (3,), 'mol', float, 'Dipole Moment'),
            GaussianFCHKFieldInfo('estruct/mol_polars/%s' % prefix, (3, 3), 'mol', float, 'Polarizability'),
            GaussianFCHKFieldInfo('estruct/mol_populations/%s' % prefix, (), 'mol', int, None),
            GaussianFCHKFieldInfo('estruct/atom_populations/%s_mulliken' % prefix, (), 'atom', float, None),
        ])

    def read(self, path):
        fchk_names = [info.fchk_name for info in self.infos if info.fchk_name is not None]
        try:
            fchk = FCHKFile(path, fchk_names)
        except IOError:
            print 'Borked', path
            return [None]*len(self.infos)
        fields = {}
        for info in self.infos:
            if info.fchk_name is not None:
                fields[info.destination] = fchk.get(info.fchk_name)

        prefix = self.prefix

        # Compute Mulliken and molecular populstion(s).
        eff_charges = fields['estruct/eff_core_charges/%s' % prefix]
        fields['estruct/mol_populations/%s' % prefix] = eff_charges.sum() \
            - fields['estruct/mol_charges/%s' % prefix]
        fields['estruct/atom_populations/%s_mulliken' % prefix] = eff_charges \
            - fields['estruct/atom_charges/%s_mulliken' % prefix]

        # Filter out ghost atoms
        mask = eff_charges > 0
        for info in self.infos:
            if info.kind == 'atom':
                fields[info.destination] = fields[info.destination][mask]

        # Fix the polarizability -> 3x3 matrix
        p = fields['estruct/mol_polars/%s' % prefix]
        if p is not None:
            fields['estruct/mol_polars/%s' % prefix] = np.array([
                [p[0], p[1], p[3]],
                [p[1], p[2], p[4]],
                [p[3], p[4], p[5]],
            ])

        return [fields[info.destination] for info in self.infos]


HDF5FieldInfo = namedtuple('FieldInfo', 'destination shape kind dtype hdf5_path')


class HDF5Fields(Fields):
    def read(self, path):
        result = []
        with h5.File(path, 'r') as f:
            for info in self.infos:
                dset = f.get(info.hdf5_path)
                if dset is None:
                    result.append(None)
                elif dset.shape == ():
                    result.append(dset[()])
                else:
                    result.append(dset[:])
        return result


class HDF5AtomChargeFields(HDF5Fields):
    def __init__(self, scheme):
        Fields.__init__(self, [
            HDF5FieldInfo('estruct/atom_charges/%s' % scheme, (), 'atom', float, 'charges'),
            HDF5FieldInfo('estruct/valence_charges/%s' % scheme, (), 'atom', float, 'valence_charges'),
            HDF5FieldInfo('estruct/valence_widths/%s' % scheme, (), 'atom', float, 'valence_widths'),
            HDF5FieldInfo('estruct/core_charges/%s' % scheme, (), 'atom', float, 'core_charges'),
            HDF5FieldInfo('estruct/atom_populations/%s' % scheme, (), 'atom', float, 'populations'),
            HDF5FieldInfo('estruct/atom_self_populations/%s' % scheme, (), 'atom', float, 'self_populations'),
        ])


TXTFieldInfo = namedtuple('FieldInfo', 'destination shape kind dtype line re')


cp2k_ddap_charges = TXTFieldInfo('estruct/atom_charges/cp2k_ddap', (), 'atom', float,
                                 None, re.compile('^ ....\d  ..   (.*)$'))
restr_lowmul = '^ .{6}\d .{6} .{6}\d .{9}\d\.\d{6} *([-+0-9].*)$'
cp2k_lowdin_charges = TXTFieldInfo('estruct/atom_charges/cp2k_lowdin', (), 'atom', float,
                                   None, re.compile(restr_lowmul))
cp2k_mulliken_charges = TXTFieldInfo('estruct/atom_charges/cp2k_mulliken', (), 'atom', float,
                                     None, re.compile(restr_lowmul))
cp2k_resp_charges = TXTFieldInfo('estruct/atom_charges/cp2k_resp', (), 'atom', float,
                                 None, re.compile('^  RESP .{6}\d  ..   (.*)$'))
cp2k_mol_population = TXTFieldInfo('estruct/mol_populations/cp2k', (), 'mol', float,
                                   None, re.compile('^ Number of electrons:   (.*\d*)$'))

class TXTFields(Fields):
    def read(self, path):
        with open(path) as f:
            values = [[]]*len(self.infos)
            for iline, line in enumerate(f):
                for iinfo, info in enumerate(self.infos):
                    if info.line is None or info.line == iline:
                        m = info.re.match(line)
                        if m is not None:
                            for s in m.groups():
                                values[iinfo].append(info.dtype(s))
            for iinfo, info in enumerate(self.infos):
                if info.kind == 'mol':
                    if info.shape == ():
                        values[iinfo] = values[iinfo][0]
                    else:
                        values[iinfo] = np.array(values[iinfo]).reshape(info.shape)
                else:
                    values[iinfo] = np.array(values[iinfo]).reshape((-1,) + info.shape)
            return values


JSONFieldInfo = namedtuple('JSONFieldInfo', 'destination shape kind dtype json_name')


class JSONFields(Fields):
    def read(self, path):
        values = []
        with open(path) as f:
            data = json.load(f)
        for info in self.infos:
            values.append(data[info.json_name])
        return values
