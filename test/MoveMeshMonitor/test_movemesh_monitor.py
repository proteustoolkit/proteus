#!/usr/bin/env python
import os
import sys
from proteus import NumericalSolution, Profiling, default_s
from proteus import runtime
import unittest
import numpy as np
import numpy.testing as npt
from importlib import import_module
from petsc4py import PETSc

class TestMoveMeshMonitor(unittest.TestCase):

    def teardown_method(self, method):
        """ Tear down function """
        FileList = ['movemesh_monitor.xmf', 'movemesh_monitor.ymf',
                    'movemesh_monitor.h5',
                    'mesh.ele',
                    'mesh.edge',
                    'mesh.node',
                    'mesh.neigh',
                    'mesh.face',
                    'mesh.poly',
                    'forceHistory_p.txt',
                    'forceHistory_v.txt',
                    'momentHistory.txt',
                    'wettedAreaHistory.txt',
                    'proteus.log'
                    'movemesh_monitor_so.log'
                    ]
        for file in FileList:
            if os.path.isfile(file):
                os.remove(file)
            else:
                pass

    def test_MoveMeshMonitor(self):
        # a unittest.TestCase method: pytest hands it no fixtures
        runtime.start()
        opts = runtime.RunOptions()
        from proteus import default_so
        from . import movemesh_monitor_so
        pList = []
        nList = []
        sys.path.append(os.path.dirname(__file__))
        for (p,n) in movemesh_monitor_so.pnList:
            pList.append(import_module(p))
            nList.append(import_module(n))
            if pList[-1].name == None:
                pList[-1].name = p
        sys.path.remove(os.path.dirname(__file__))
        so = movemesh_monitor_so
        so.name = "movemesh_monitor"
        if so.sList == []:
            for i in range(len(so.pnList)):
                s = default_s
                so.sList.append(s)
        Profiling.logLevel=7
        Profiling.verbose=True
        # PETSc solver configuration
        OptDB = PETSc.Options()
        dirloc = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(dirloc, "petsc.options.asm")) as f:
            all = f.read().split()
            i=0
            while i < len(all):
                if i < len(all)-1:
                    if all[i+1][0]!='-':
                        print("setting ", all[i].strip(), all[i+1])
                        OptDB.setValue(all[i].strip('-'),all[i+1])
                        i=i+2
                    else:
                        print("setting ", all[i].strip(), "True")
                        OptDB.setValue(all[i].strip('-'),True)
                        i=i+1
                else:
                    print("setting ", all[i].strip(), "True")
                    OptDB.setValue(all[i].strip('-'),True)
                    i=i+1
        ns = NumericalSolution.NS_base(so,pList,nList,so.sList,opts)
        ns.calculateSolution('movemesh_monitor')
        nodes = ns.modelList[0].levelModelList[0].mesh.nodeArray
        nodesResult = np.genfromtxt(os.path.join(dirloc, "nodesResult.csv"), delimiter=',')
        for node in range(len(nodesResult)):
            npt.assert_almost_equal(nodes[node], nodesResult[node], decimal=5)
            # npt.assert_almost_equal(nodes[node, 0], nodesResult[node, 0], decimal=5)
            # npt.assert_almost_equal(nodes[node, 1], nodesResult[node, 1], decimal=5)
            # npt.assert_almost_equal(nodes[node, 2], nodesResult[node, 2], decimal=5)
        self.teardown_method(self)

if __name__ == "__main__":
    unittest.main()
