import os
import pytest

class TestIsosurface(object):

    @classmethod
    def setup_class(cls):
        pass

    @classmethod
    def teardown_class(cls):
        pass

    def setup_method(self,method):
        pass

    def teardown_method(self,method):
        [os.remove(f) for f in os.listdir(".") if f.endswith(".pov")]
        FileList = ['isostats',
                    'proteus_default.log',
                    'proteus.inc',
                    'proteus.log']
        for file in FileList:
            if os.path.isfile(file):
                os.remove(file)
    
    # NOTE: need thorough evaluation -- unskipped this session. The
    # `povray` renderer was never the blocker (povgen.py only writes .pov
    # scene-description text, never invokes povray). The original crash was
    # `import tables` (PyTables) missing in scripts/povgen.py; rather than
    # just installing PyTables, povgen.py was switched to h5py (`h5py.File`
    # instead of `tables.open_file`), and proteus/Isosurface.pyx's
    # attachHDF5() -- its sole caller -- was switched from PyTables'
    # `h5.get_node(path)[:]` to h5py's native `h5[path][:]`, so this test no
    # longer depends on PyTables at all. The genuinely missing pieces were
    # the input fixture test/ci/comparison_files/floating_bar.h5 and the
    # expected-output archive phi_t_0.000000_000.tgz, both entirely absent
    # from this checkout (never tracked in git); real copies of both were
    # supplied.
    def test_povgen(self):
        import difflib
        import subprocess
        import os
        subprocess.check_call(['povgen.py',
                               os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            'comparison_files',
                                            'floating_bar'),
                               '-s',
                               '3'], timeout=600)
        povfiles = []
        for i in range(3):
            filename = 'phi_t_0.000000_{0:04d}.pov'.format(i)
            with open(filename, 'r') as f:
                povfiles.append(f.readlines())
        subprocess.check_call(['tar', 'xzf', os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                          'comparison_files',
                                                          'phi_t_0.000000_000.tgz')], timeout=120)
        saved_povfiles = []
        for i in range(3):
            filename = 'phi_t_0.000000_{0:04d}.pov'.format(i)
            with open(filename, 'r') as f:
                saved_povfiles.append(f.readlines())
            assert saved_povfiles[i] == povfiles[i], \
                ''.join(list(difflib.unified_diff(saved_povfiles[i],
                                                  povfiles[i],
                                                  "archived",
                                                  "test")))
