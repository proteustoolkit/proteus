"""Auxiliary variable classes for storing solutions at points and
along lines to mimic gauges in lab and field experiments.

.. inheritance-diagram:: proteus.Gauges
   :parts: 1

"""
import os
from collections import defaultdict, OrderedDict
from itertools import product

from mpi4py import MPI
from .petsc_optional import PETSc
import numpy as np
from numpy.linalg import norm

from . import Comm
from .AuxiliaryVariables import AV_base
from .Profiling import logEvent
from proteus.MeshTools import (triangleVerticesToNormals, tetrahedronVerticesToNormals, cubeVerticesToNormals,
                               getMeshIntersections)
from proteus import Profiling
from proteus.FemTools import ReferenceCube


def _checkFields(fields):
    """Returns fields as a tuple of names.  A bare string is one field
    name: ('u') is just 'u', and iterating it per character would turn
    'u0' into the fields 'u' and '0'."""
    if isinstance(fields, str):
        return (fields,)
    if not hasattr(fields, '__iter__'):
        raise ValueError("gauge fields must be a tuple or list of field names, not %r" % (fields,))
    return tuple(fields)

def _checkPoint(point):
    """Returns point as a 3-tuple of finite floats (so lists work as
    dictionary keys), or raises ValueError."""
    try:
        coords = tuple(float(c) for c in point)
    except (TypeError, ValueError):
        raise ValueError("gauge location %r must be a sequence of 3 numbers" % (point,))
    if len(coords) != 3:
        raise ValueError("gauge location %r has %d coordinates, expected 3 "
                         "(x, y, z), also for 2D problems: use z = 0" % (point, len(coords)))
    if not all(np.isfinite(coords)):
        raise ValueError("gauge location %r must be finite" % (point,))
    return coords

def _checkLine(line):
    """Returns line as a pair of 3-tuples, or raises ValueError."""
    try:
        endpoints = tuple(line)
    except TypeError:
        raise ValueError("gauge line %r must be a pair of endpoints" % (line,))
    if len(endpoints) != 2:
        raise ValueError("gauge line %r has %d endpoints, expected 2" % (line, len(endpoints)))
    start, end = _checkPoint(endpoints[0]), _checkPoint(endpoints[1])
    if start == end:
        raise ValueError("gauge line %r has zero length" % (line,))
    return start, end


#: History files and hot starts. A run writes gauge files and the legacy RANS2P/RANS3PF histories as it goes; a run
#: continued from an archive must leave each one as the uninterrupted run would: its header and every row up to the
#: restart instant, then the continuation's rows. NumericalSolution sets the hot start before the models are built
#: (reset_history_files), every writer opens its files with open_history_file and registers how to cut them back
#: (on_hot_start), and once the restart instant is known NumericalSolution calls resume_history_files(t).
_history = {"hotStart": False, "resumers": []}


def reset_history_files(hotStart):
    """Start a run: on a hot start the history files are kept and cut back, otherwise they are rewritten."""
    _history["hotStart"] = bool(hotStart)
    _history["resumers"] = []


def open_history_file(path):
    """Open a history file for writing: (file, resumed). On a hot start an existing file is kept (resumed is True)
    and its writer cuts it back in the callback it gives on_hot_start."""
    if _history["hotStart"] and os.path.exists(path):
        return open(path, "a+"), True
    return open(path, "w"), False


def on_hot_start(callback):
    """callback(t) is called with the restart instant on a hot start (on every rank that registered it)."""
    _history["resumers"].append(callback)


def resume_history_files(t):
    """Cut every registered history file back to the restart instant t."""
    for callback in _history["resumers"]:
        callback(t)


def _rewrite(f, lines):
    f.seek(0)
    f.truncate()
    f.write("".join(lines))
    f.flush()


def _at_or_before(time, t):
    return time <= t + 1e-10 * max(1.0, abs(t))


def keep_rows_up_to(f, t, nHeader=1):
    """Keep the header and the rows whose first column (the time) is at or before t; returns their times."""
    f.seek(0)
    lines = f.readlines()
    kept, times = lines[:nHeader], []
    for line in lines[nHeader:]:
        time = float(line.split(",")[0])
        if not _at_or_before(time, t):
            break
        kept.append(line)
        times.append(time)
    _rewrite(f, kept)
    return times


def keep_steps_up_to(timeFile, members, t):
    """For histories that write one time per step to timeFile and rowsPerStep rows per step to each member file
    (members: (file, rowsPerStep) pairs, no header): keep the steps at or before t in all of them."""
    timeFile.seek(0)
    lines = timeFile.readlines()
    nSteps = 0
    while nSteps < len(lines) and _at_or_before(float(lines[nSteps]), t):
        nSteps += 1
    _rewrite(timeFile, lines[:nSteps])
    for f, rowsPerStep in members:
        f.seek(0)
        _rewrite(f, f.readlines()[:nSteps * rowsPerStep])
    return nSteps


def PointGauges(gauges, activeTime=None, sampleRate=0, fileName='point_gauges.csv'):
    """Create a set of point gauges that will automatically be serialized
    as CSV data to the requested file.

    :param gauges: An iterable of "gauges".  Each gauge is specified
                   by a 2-tuple, with the first element in the tuple a
                   set of fields to be monitored, and the second
                   element a tuple of the 3-space representations of
                   the gauge locations.

    See the Gauges class for an explanation of the other parameters.

    Example::

      p = PointGauges(gauges=((('u', 'v'), ((0.5, 0.5, 0), (1, 0.5, 0))),
                             (('p',), ((0.5, 0.5, 0),))),
                      activeTime=(0, 2.5),
                      sampleRate=0.2,
                      fileName='combined_gauge_0_0.5_sample_all.csv')

    This creates a PointGauges object that will monitor the u and v
    fields at the locations [0.5, 0.5, 0] and [1, 0.5, 0], and the p
    field at [0.5, 0.5, 0] at simulation time between = 0 and 2.5 with
    samples taken no more frequently than every 0.2 seconds.  Results
    will be saved to: combined_gauge_0_0.5_sample_all.csv.

    """

    # build up dictionary of location information from gauges
    # dictionary of dictionaries, outer dictionary is keyed by location (3-tuple)
    # inner dictionaries contain monitored fields, and closest node
    # closest_node is None if this process does not own the node
    points = OrderedDict()
    fields = list()

    for gauge in gauges:
        gauge_fields, gauge_points = gauge
        gauge_fields = _checkFields(gauge_fields)
        for field in gauge_fields:
            if field not in fields:
                fields.append(field)
        for point in gauge_points:
            point = _checkPoint(point)
            # initialize new dictionary of information at this location
            if point not in points:
                l_d = {'fields': set()}
                points[point] = l_d
            # add any currently unmonitored fields
            points[point]['fields'].update(gauge_fields)
    return Gauges(fields, activeTime, sampleRate, fileName, points=points)

def LineGauges(gauges, activeTime=None, sampleRate=0, fileName='line_gauges.csv'):
    """Create a set of line gauges that will automatically be serialized
    as CSV data to the requested file.  The line gauges will gather
    data at every element on the mesh between the two endpoints on
    each line.

    :param gauges: An iterable of "gauges".  Each gauge is specified
                   by a 2-tuple, with the first element in the tuple a
                   set of fields to be monitored, and the second
                   element a list of pairs of endpoints of the gauges
                   in 3-space representation.

    See the Gauges class for an explanation of the other parameters.

    """

    # expand the product of fields and lines for each gauge
    lines = list()
    fields = list()
    for gauge in gauges:
        gauge_fields, gauge_lines = gauge
        gauge_fields = _checkFields(gauge_fields)
        for field in gauge_fields:
            if field not in fields:
                fields.append(field)
        lines.extend(product(gauge_fields, [_checkLine(line) for line in gauge_lines]))

    return Gauges(fields, activeTime, sampleRate, fileName, lines=lines)

def LineIntegralGauges(gauges, activeTime=None, sampleRate=0, fileName='line_integral_gauges.csv'):
    """Create a set of line integral gauges that will automatically be
    serialized as CSV data to the requested file.

    :param gauges: An iterable of "gauges".  Each gauge is specified
                   by a 2-tuple, with the first element in the tuple a
                   set of fields to be monitored, and the second
                   element a list of pairs of endpoints of the gauges
                   in 3-space representation.

    See the Gauges class for an explanation of the other parameters.

    """

    # expand the product of fields and lines for each gauge
    lines = list()
    fields = list()
    for gauge in gauges:
        gauge_fields, gauge_lines = gauge
        gauge_fields = _checkFields(gauge_fields)
        for field in gauge_fields:
            if field not in fields:
                fields.append(field)
        lines.extend(product(gauge_fields, [_checkLine(line) for line in gauge_lines]))

    return Gauges(fields, activeTime, sampleRate, fileName, lines=lines, integrate=True)


class Gauges(AV_base):
    """Monitor fields at specific values.

    This class provides a generic point-wise and line-integral monitor
    that can be instantiated and attached to Proteus simulations by
    including them in the list of Auxiliary Variables in problem
    setup.

    Each Gauges instance may contain one or more fields, which may
    contain one or more locations to monitor.  The monitoring is
    defined over a given time and sample rate, and a filename is also
    supplied.  All results are serialized to a CSV file.

    Parallel Implementation Notes: After the gauge has been attached,
    all processes are partitioned into Gauge Owners and non-Gauge
    Owners.  The calculate method is a "no-op" for non-Owners.  For
    Gauge Owners, all values are computed individually, then
    collectively transmitted to the "root" process, which is the only
    process responsible for serializing gauge results to disk.  This
    code has not been aggressively vetted for parallel correctness or
    scalability.

    """
    def __init__(self, fields, activeTime=None, sampleRate=0, fileName='gauges.csv', points=None, lines=None,
                 integrate=False):
        """Create a set of gauges that will automatically be serialized as
        CSV data to the requested file.


        :param activeTime: If not None, a 2-tuple of start time and
                           end time for which the point gauge is
                           active.
        :param sampleRate: The intervals at which samples should be
                           measured.  Samples are anchored to the grid
                           t0 + k*sampleRate, where t0 is the first
                           sample, and each is taken at the first time
                           step at or after its grid time, so the mean
                           interval is sampleRate but individual
                           intervals can be up to one time step longer.
                           A sampleRate below the time step samples
                           every step.  The default value of zero
                           computes the gauge values at every time
                           step.
        :param fileName: The name of the file to serialize results to.

        Data is currently column-formatted, with 10 characters
        allotted to the time field, and 45 characters allotted to each
        point field.

        """

        AV_base.__init__(self)

        if not (np.isfinite(sampleRate) and sampleRate >= 0):
            raise ValueError("sampleRate must be a finite number >= 0, got %r" % (sampleRate,))
        if activeTime is not None:
            if len(activeTime) != 2 or not activeTime[0] <= activeTime[1]:
                raise ValueError("activeTime must be (start, end) with start <= end, got %r"
                                 % (activeTime,))

        self.fields = fields
        self.activeTime = activeTime
        self.sampleRate = sampleRate
        self.fileName = fileName
        self.points = points if points else OrderedDict()
        self.lines = lines if lines else []
        self.file = None  # only the root process should have a file open
        self.flags = {}
        self.files = {}
        self.outputWriterReady = False
        self.last_output = None
        self.first_output = None
        self.next_output = None
        self.pointGaugeMats = []
        self.field_ids = []
        self.dofsVecs = []
        self.pointGaugeVecs = []
        self.segments = []
        self.discontinuousFields = {}
        self.adapted = False
        self.resumed = False  # the file of the run this one continues is kept (a hot start)
        self.resumeTime = None  # rows at or before it are in that file already

        self.isPointGauge = bool(points)
        self.isLineGauge = bool(lines) and not integrate
        self.isLineIntegralGauge = bool(lines) and integrate

        if not (self.isPointGauge or self.isLineGauge or self.isLineIntegralGauge):
            raise ValueError("Need to provide points or lines")
        if sum((self.isPointGauge, self.isLineGauge, self.isLineIntegralGauge)) > 1:
            raise ValueError("Must be one of point or line gauge but not both")

    def getLocalNearestNode(self, location):
        # determine local nearest node distance
        nearest_node_distance_kdtree, nearest_node_kdtree = self.nodes_kdtree.query(location)
        comm = Comm.get().comm.tompi4py()
        return comm.rank, nearest_node_kdtree, nearest_node_distance_kdtree

    @staticmethod
    def referenceElementViolation(referenceElement, xi):
        """How far xi is outside the reference element, in reference coordinates
        (zero on and inside it)."""
        xi = np.asarray(xi, dtype=float)
        if referenceElement.onElement(xi):
            return 0.0
        if isinstance(referenceElement, ReferenceCube):
            return float(np.max(np.abs(xi) - 1.0))
        return float(max(-np.min(xi), np.sum(xi) - 1.0))

    def getLocalElement(self, femSpace, location, node, maxRings=5):
        """Given a location and its nearest node, determine if it is on a
        local element.

        Elements are searched in rings of node-neighbors growing out
        from the nearest node, up to maxRings deep.  The element
        containing a location is not always in the star of the nearest
        node (or of its neighbors) on graded or skewed meshes.

        .. figure:: /_generated/point_location_rings.svg
           :width: 100%

           The first three layers of the search on a 2D mesh.  Red
           nodes have their elements searched in that layer (black
           edges); their new neighbors (black dots) are the red nodes of
           the next layer, and grey edges were searched already.  The
           search stops at the first element containing the location,
           and gives up after maxRings further layers.

        Returns None if location is not on any elements owned by this
        process

        """

        mesh = femSpace.mesh
        checkedElements = set()
        checkedNodes = set()
        ring = {node}
        containing = []
        nearMiss = (np.inf, None)
        lastDepth = maxRings
        for depth in range(maxRings + 1):
            nextRing = set()
            for n in ring:
                checkedNodes.add(n)
                for eOffset in range(mesh.nodeElementOffsets[n], mesh.nodeElementOffsets[n + 1]):
                    eN = mesh.nodeElementsArray[eOffset]
                    if eN in checkedElements:
                        continue
                    checkedElements.add(eN)
                    # evaluate the inverse map for element eN
                    xi = femSpace.elementMaps.getInverseValue(eN, location)
                    # query whether xi lies within the reference element, up to
                    # roundoff: a location on a face between elements (every
                    # end of a line gauge segment) is a roundoff outside of both
                    violation = self.referenceElementViolation(femSpace.elementMaps.referenceElement, xi)
                    if violation <= 1e-9:
                        containing.append(eN)
                        lastDepth = min(lastDepth, depth + 1)
                    elif violation < nearMiss[0]:
                        nearMiss = (violation, eN)
                    nextRing.update(mesh.elementNodesArray[eN])
            if containing and depth >= lastDepth:
                # a location on a face lies in every element sharing that face,
                # all within one ring of the first one found
                break
            ring = nextRing - checkedNodes
            if not ring:
                break
        if not containing:
            # a location on a curved or tilted boundary can miss the mesh by
            # roundoff, outside every element; take the element it misses least,
            # if the miss is a roundoff of the element's size
            if nearMiss[0] <= 1e-6:
                return nearMiss[1]
            # no elements found
            return None
        # a discontinuous space has a value from each side of a face: take the
        # one from the element with the lowest global number, whatever the
        # order the search found them in
        return min(containing, key=lambda eN: (self.globalElementNumber(mesh, eN), eN))

    @staticmethod
    def globalElementNumber(mesh, eN):
        numbering = getattr(mesh, 'elementNumbering_subdomain2global', None)
        # a serial mesh may carry an empty numbering
        return int(numbering[eN]) if numbering is not None and len(numbering) > eN else int(eN)

    def findNearestNode(self, femSpace, location):
        """Given a gauge location, attempts to locate the most suitable
        process for monitoring information about this location, as
        well as the node on the process closest to the location.

        Returns a 2-tuple containing an identifier for the closest
        'owning' process as well as the local ids of the node and
        nearest element.

        """
        comm = Comm.get().comm.tompi4py()
        comm_rank, nearest_node, nearest_node_distance = self.getLocalNearestNode(location)
        local_element = self.getLocalElement(femSpace, location, nearest_node)

        # determine global nearest node
        haveElement = int(local_element is not None)
        # the owner holds the element with the lowest global number, so a point on a
        # face of a discontinuous space reads the same side on any rank layout
        elementRank = (self.globalElementNumber(femSpace.mesh, local_element)
                       if haveElement else np.iinfo(np.int32).max)
        global_element_rank, owning_proc = comm.allreduce((elementRank, comm.rank), op=MPI.MINLOC)
        global_have_element = int(global_element_rank != np.iinfo(np.int32).max)
        if global_have_element:
            logEvent("Gauges on element at location: [%g %g %g] assigned to %d" % (location[0], location[1], location[2],
                                                                    owning_proc), 3)
        else:
            # gauge isn't on any of the elements, just use nearest node
            global_min_distance, owning_proc = comm.allreduce((nearest_node_distance,comm.rank), op=MPI.MINLOC)
            logEvent("Off-element gauge location: [%g %g %g] assigned to %d" % (location[0], location[1], location[2],
                                                                 owning_proc), 3)
        if comm.rank != owning_proc:
            nearest_node = None

        assert owning_proc is not None
        return owning_proc, nearest_node

    def buildQuantityRow(self, m, femFun, quantity_id, quantity):
        """Builds up contributions to gauge operator from the underlying
        element space

        """

        location, node = quantity

        # a gauge of a discontinuous field on a line is measured in a given element
        localElement = self.points[location].get('element')
        location = location[:3]
        if localElement is None:
            # search elements that contain the nearest node
            # use nearest node if the location is not found on any elements
            localElement = self.getLocalElement(femFun.femSpace, location, node)
        if localElement is not None:
            for i, psi in enumerate(femFun.femSpace.referenceFiniteElement.localFunctionSpace.basis):
                # assign quantity weights here
                xi = femFun.femSpace.elementMaps.getInverseValue(localElement, location)
                m[quantity_id, femFun.femSpace.dofMap.l2g[localElement, i]] = psi(xi)
        else:
            # just use nearest node for now if we're given a point outside the domain.
            # the ideal thing would be to find the element with the nearest face
            logEvent("WARNING: gauge location [%g %g %g] is not on any local element; "
                     "using the value at its nearest node %d" % (location[0], location[1], location[2], node), 1)
            m[quantity_id, node] = 1


    def initOutputWriter(self):
        """Initialize communication strategy for collective output of gauge
        data.

        On the root process in this communicator, create a map of
        quantity owners and the corresponding location in their
        arrays.  This process is responsible for collecting gauge data
        and saving it to disk.

        Gauge data is globally ordered by field, then by location id
        (as ordered by globalMeasuredQuantities)

        """

        numLocalQuantities = sum([len(self.measuredQuantities[field]) for field in self.fields])
        self.localQuantitiesBuf = np.zeros(numLocalQuantities)
        if self.gaugeComm.rank != 0:
            self.globalQuantitiesBuf = None
            self.globalQuantitiesCounts = None
        else:
            if self.adapted:
              if(Profiling.logDir not in self.fileName):
                self.fileName = os.path.join(Profiling.logDir, self.fileName)
              self.file = open(self.fileName, 'a')
            else:
              self.fileName = os.path.join(Profiling.logDir, self.fileName)
              self.file, self.resumed = open_history_file(self.fileName)

            if self.isLineIntegralGauge:
                #Only need to set up mapping for point gauges
                return

            quantityIDs = [0] * self.gaugeComm.size
            numGlobalQuantities = sum([len(self.globalMeasuredQuantities[field]) for field in self.fields])
            # Assign quantity ids to processors
            for field in self.fields:
                for id in range(len(self.globalMeasuredQuantities[field])):
                    location, owningProc = self.globalMeasuredQuantities[field][id]
                    gaugeProc = self.globalGaugeRanks[owningProc]
                    quantityID = quantityIDs[gaugeProc]
                    quantityIDs[gaugeProc] += 1
                    assert gaugeProc >= 0
                    self.globalMeasuredQuantities[field][id] = location, gaugeProc, quantityID
                    logEvent("Gauge for %s[%d] at %e %e %e is at P[%d][%d]" % (field, id, location[0], location[1],
                                                                          location[2], gaugeProc, quantityID), 5)

            logEvent("Quantity IDs:\n%s" % str(quantityIDs), 5)

            # determine mapping from global measured quantities to communication buffers
            self.globalQuantitiesMap = np.zeros(numGlobalQuantities, dtype=int)
            i = 0
            for field in self.fields:
                for location, gaugeProc, quantityID in self.globalMeasuredQuantities[field]:
                    self.globalQuantitiesMap[i] = sum(quantityIDs[:gaugeProc]) + quantityID
                    assert self.globalQuantitiesMap[i] < numGlobalQuantities
                    i += 1

            # a couple consistency checks
            assert sum(quantityIDs) == numGlobalQuantities
            assert all(quantityID > 0 for quantityID in quantityIDs)

            # final ids also equal to the counts on each process
            self.globalQuantitiesCounts = quantityIDs
            self.globalQuantitiesBuf = np.zeros(numGlobalQuantities, dtype=np.double)
            logEvent("Global Quantities Map: \n%s" % str(self.globalQuantitiesMap), 5)

        self.outputWriterReady = True

    def buildGaugeComm(self):
        """Create a communicator composed only of processes that own gauge
        quantities.

        Collective over global communicator.  Builds a local
        communicator for collecting all gauge data.  This communicator
        contains only processes that will contain gauge data.

        """

        comm = Comm.get().comm.tompi4py()

        gaugeOwners = set()

        for field in self.fields:
            for location, owningProc in self.globalMeasuredQuantities[field]:
                gaugeOwners.update((owningProc,))

        self.isGaugeOwner = comm.rank in gaugeOwners
        gaugeComm = comm.Split(color=self.isGaugeOwner)

        logEvent("Gauge owner: %d" % self.isGaugeOwner, 5)
        if self.isGaugeOwner:
            self.gaugeComm = gaugeComm
            gaugeRank = self.gaugeComm.rank
        else:
            self.gaugeComm = None
            gaugeRank = -1
        self.globalGaugeRanks = comm.allgather(gaugeRank)
        logEvent("Gauge ranks: \n%s" % str(self.globalGaugeRanks), 5)


    @staticmethod
    def pointOrder(item):
        """Sort key for (location, data) items of self.points: by location,
        except the measurements of discontinuous fields, which follow in the
        order along their lines."""
        location, data = item
        if 'order' in data:
            return (1, data['order'], location)
        return (0, location, ())

    def isDiscontinuous(self, field):
        """Whether no degree of freedom of the field's space is shared between
        elements, so that its value at a point on an element face depends on
        which element it is taken from.

        Collective over the global communicator, and cached.
        """
        if field not in self.discontinuousFields:
            femSpace = self.u[self.fieldNames.index(field)].femSpace
            local = int(femSpace.dim == femSpace.dofMap.l2g.size)
            comm = Comm.get().comm.tompi4py()
            self.discontinuousFields[field] = bool(comm.allreduce(local, op=MPI.MIN))
        return self.discontinuousFields[field]

    def addLineGaugePoints(self, line, line_segments):
        """Add all gauge points from each line into self.points

        Returns the keys of self.points that the two ends of each segment
        are measured at.  Where the field is continuous the key is the
        point itself.  Where it is discontinuous (see
        :meth:`isDiscontinuous`) the field has a value from each side of
        an element face, so every segment is measured in its own element
        at both ends, and the key is the point, the side of the point the
        segment is on along the line (-1 before it, +1 after it) and the
        global number of the element.  A face crossed by the line gives
        two gauges at one location, one for each side, and the line of
        gauges jumps there.
        """
        points = self.points
        new_points = {}
        field, endpoints = line
        comm = Comm.get().comm.tompi4py()
        discontinuous = self.isDiscontinuous(field)
        segment_keys = []

        def addPoint(points, field, point):
            point = tuple(point)
            if point in points:
                if self.isLineIntegralGauge:
                    no_output = points[point]['no_output'] if 'no_output' in points[point] else set()
                    points[point]['no_output'] = no_output.union(set((field,)) - points[point]['fields'])
                points[point]['fields'].update((field,))
            else:
                ignore1, nearestNode, ignore2 = self.getLocalNearestNode(point)
                if self.isLineIntegralGauge:
                    points[point] = {'fields':set((field,)), 'no_output': set((field,)),
                                     'nearest_node': nearestNode,
                                     'owning_proc': comm.rank}
                else:
                    points[point] = {'fields':set((field,)),
                                     'nearest_node': nearestNode,
                                     'owning_proc': comm.rank}
            new_points[point] = points[point]

        lineIndex = self.lines.index(line)

        def addSidedPoint(points, field, point, side, segment):
            """Measure point in the element of segment, which is on the given
            side of it along the line"""
            midpoint = 0.5 * (np.asarray(segment[0], np.double) + np.asarray(segment[1], np.double))
            femSpace = self.u[self.fieldNames.index(field)].femSpace
            ignore1, midpointNode, ignore2 = self.getLocalNearestNode(midpoint)
            # the midpoint of a segment is inside its element, not on a face of it
            element = self.getLocalElement(femSpace, midpoint, midpointNode)
            if element is None:
                # a long thin element can be many rings of nodes from the node nearest its midpoint
                element = self.getLocalElement(femSpace, midpoint, midpointNode, maxRings=100)
            if element is None:
                raise ValueError("gauge line segment %s on a discontinuous field is in no element "
                                 "of this process" % (list(map(list, segment)),))
            key = tuple(point) + (side, self.globalElementNumber(femSpace.mesh, element))
            if key in points:
                points[key]['fields'].update((field,))
            else:
                ignore1, nearestNode, ignore2 = self.getLocalNearestNode(point)
                points[key] = {'fields': set((field,)),
                               'nearest_node': nearestNode,
                               'element': element,
                               # the place along the line: each segment's start then its end,
                               # so an end and the next start stay in this order whatever
                               # roundoff does to their coordinates
                               'order': (lineIndex, parameter(segment[0]), int(side < 0)),
                               'owning_proc': comm.rank}
                if self.isLineIntegralGauge:
                    points[key]['no_output'] = set((field,))
            new_points[key] = points[key]
            return key

        # a segment's start point is the end of the previous segment unless
        # the line has just come back into the mesh (at its start, or after a hole)
        direction = np.asarray(endpoints[1], np.double) - np.asarray(endpoints[0], np.double)
        length = norm(direction)
        def parameter(point):
            return np.dot(np.asarray(point, np.double) - endpoints[0], direction)/length**2
        # the previous segment may have been selected on another process
        end_parameters = np.sort(np.concatenate(comm.allgather(
            np.array([parameter(segment[1]) for segment in line_segments], dtype=float))))

        def continuesPreviousSegment(point):
            t = parameter(point)
            i = np.searchsorted(end_parameters, t - 1e-9)
            return i < len(end_parameters) and end_parameters[i] <= t + 1e-9

        for segment in line_segments:
            logEvent("Processing segment [ %e %e %e ] to [ %e %e %e ]" % (
                segment[0][0], segment[0][1], segment[0][2],
                segment[1][0], segment[1][1], segment[1][2]), 5)
            startPoint, endPoint = segment
            if discontinuous:
                segment_keys.append((addSidedPoint(points, field, startPoint, 1, segment),
                                     addSidedPoint(points, field, endPoint, -1, segment)))
                continue
            # line integral gauges need both sides of every segment, line gauges
            # only the points where the line enters the mesh
            if self.isLineIntegralGauge or not continuesPreviousSegment(startPoint):
                addPoint(points, field, startPoint)
            addPoint(points, field, endPoint)
            segment_keys.append((tuple(startPoint), tuple(endPoint)))

        if self.isLineGauge:
            new_points = comm.gather(new_points)
            if comm.rank == 0:
                for new_points_i in new_points:
                    points.update(new_points_i)
                # resort points
                points = OrderedDict(sorted(points.items(), key=self.pointOrder))
            self.points = comm.bcast(points)
        return segment_keys

    def identifyMeasuredQuantities(self):
        """ build measured quantities, a list of fields

        Each field in turn contains a list of gauge locations and their accompanying nearest node
        only local quantities are saved
        """

        self.measuredQuantities = defaultdict(list)
        self.globalMeasuredQuantities = defaultdict(list)
        comm = Comm.get().comm.tompi4py()

        points = self.points

        for point, l_d in points.items():
            if 'nearest_node' not in l_d:
                # TODO: Clarify assumption here about all fields sharing the same element mesh
                field_id = self.fieldNames.index(list(l_d['fields'])[0])
                femSpace = self.u[field_id].femSpace
                owningProc, nearestNode = self.findNearestNode(femSpace, point)
                l_d['nearest_node'] = nearestNode
            else:
                owningProc = l_d['owning_proc']
                # nearestNode only makes sense on owning process
                # so even if we have this information, it's not valid for this point
                if owningProc == comm.rank:
                    nearestNode = l_d['nearest_node']
                else:
                    nearestNode = None
            for field in l_d['fields']:
                self.globalMeasuredQuantities[field].append((point, owningProc))
                if nearestNode is not None:
                    point_id = len(self.measuredQuantities[field])
                    logEvent("Gauge for %s[%d] at %e %e %e is closest to node %d" % (field, point_id, point[0], point[1],
                                                                                point[2], nearestNode), 3)
                    l_d[field] = point_id
                    self.measuredQuantities[field].append((point, nearestNode))

    def buildPointGaugeOperators(self):
        """ Build the linear algebra operators needed to compute the point gauges.

        The operators are all local since the point gauge measurements are calculated locally.
        """

        for field, field_id in zip(self.fields, self.field_ids):
            m = PETSc.Mat().create(PETSc.COMM_SELF)
            m.setSizes([len(self.measuredQuantities[field]),
                        self.u[field_id].femSpace.dim])
            m.setType('aij')
            m.setUp()
            # matrices are a list in same order as fields
            self.pointGaugeMats.append(m)
            # dofs are a list in same order as fields as well
            dofs = self.u[field_id].dof
            dofsVec = PETSc.Vec().createWithArray(dofs, comm=PETSc.COMM_SELF)
            self.dofsVecs.append(dofsVec)

        for field, field_id, m in zip(self.fields, self.field_ids, self.pointGaugeMats):
            # get the FiniteElementFunction object for this quantity
            femFun = self.u[field_id]
            for quantity_id, quantity in enumerate(self.measuredQuantities[field]):
                location, node = quantity
                logEvent("Gauge for: %s at %e %e %e is on local operator row %d" % (field, location[0], location[1],
                                                                      location[2], quantity_id), 3)
                self.buildQuantityRow(m, femFun, quantity_id, quantity)
            pointGaugesVec = PETSc.Vec().create(comm=PETSc.COMM_SELF)
            pointGaugesVec.setSizes(len(self.measuredQuantities[field]))
            pointGaugesVec.setUp()
            self.pointGaugeVecs.append(pointGaugesVec)

        for m in self.pointGaugeMats:
            m.assemble()

    def pruneDuplicateSegments(self, endpoints, length_segments):
        """ prune duplicate segments across processors

        endpoints - a pair of points in 3-space defining the line
        length_segments - a pair of intersections augmented by length

        this could be optimized
        """

        eps = 1e-9

        comm = Comm.get().comm.tompi4py()

        length_segments = sorted(length_segments)
        length_segments = comm.gather(length_segments)


        selected_segments = None
        error = None
        if comm.rank == 0:
            selected_segments = [[] for i in range(len(length_segments))]
            segment_pos = 0
            covered = 0.
            while segment_pos < (1 - eps):
                # choose the longest line from those that start at segment_pos; one
                # no longer than eps is the same position (a mesh node a roundoff off
                # the line cuts slivers like that), and choosing it would leave its
                # own start within eps of the new position, so it would be chosen again
                longest_segment = eps, None, None
                next_start = None
                for proc_rank, proc_length_segments in enumerate(length_segments):
                    segment_id = 0
                    for segment_id, length_segment in enumerate(proc_length_segments):
                        # ignore segments below current position (they will be discarded)
                        start, end, segment = length_segment
                        if start < (segment_pos - eps):
                            continue
                        # equality test
                        elif start < (segment_pos + eps):
                            segment_length = end - start
                            if segment_length > longest_segment[0]:
                                longest_segment = segment_length, proc_rank, segment
                        else:
                            # first segment starting beyond segment_pos: candidate for the end of a gap
                            if next_start is None or start < next_start:
                                next_start = start
                            break
                    # discard any segments that start before our current position
                    proc_length_segments[:] = proc_length_segments[segment_id:]

                segment_length, proc_rank, segment = longest_segment
                if segment is None:
                    # the line leaves the mesh here: skip to the next segment, or stop
                    if next_start is None:
                        break
                    segment_pos = next_start
                    continue
                logEvent("Identified best segment of length %g on %d: %s" % (segment_length, proc_rank, str(segment)), 9)
                selected_segments[proc_rank].append(segment)
                segment_pos += segment_length
                covered += segment_length

            if covered == 0:
                error = "Gauge line %s is entirely outside the mesh" % (endpoints.tolist(),)
            elif covered < 1 - 1e-6:
                logEvent("WARNING: gauge line %s is partly outside the mesh; using the %.6g of it "
                         "that is inside" % (endpoints.tolist(), covered), 1)
            elif abs(segment_pos - 1) > 1e-8:
                msg = "Segmented line %s different from original length by ratio %e\n segments: %s" % (
                    str(endpoints), abs(segment_pos - 1), str(selected_segments))
                logEvent(msg, 3)
                if abs(segment_pos - 1) > 1e-6:
                    error = msg
        # every rank must see the error, or the others wait forever in scatter
        error = comm.bcast(error)
        if error is not None:
            raise ValueError(error)

        logEvent("Selected segments: %s" % str(selected_segments), 9)
        segments = comm.scatter(selected_segments)
        return segments

    def getMeshIntersections(self, line):
        field, endpoints = line
        # get Proteus mesh index for this field
        field_id = self.fieldNames.index(field)
        femFun = self.u[field_id]
        mesh = femFun.femSpace.mesh
        referenceElement = femFun.femSpace.elementMaps.referenceElement
        if referenceElement.dim == 2 and referenceElement.nNodes == 3:
            toPolyhedron = triangleVerticesToNormals
        elif referenceElement.dim == 3 and referenceElement.nNodes == 4:
            toPolyhedron = tetrahedronVerticesToNormals
        elif ((referenceElement.dim == 2 and referenceElement.nNodes == 4)
              or (referenceElement.dim == 3 and referenceElement.nNodes == 8)):
            toPolyhedron = cubeVerticesToNormals
        else:
            raise NotImplementedError("Unable to compute mesh intersections for this element type")
        # an element that cannot be clipped must stop every rank, or the others wait for it below
        comm = Comm.get().comm.tompi4py()
        try:
            intersections = np.asarray(list(getMeshIntersections(mesh, toPolyhedron, endpoints)), dtype=np.double)
            error = None
        except ValueError as e:
            error = str(e)
        errors = [e for e in comm.allgather(error) if e is not None]
        if errors:
            raise ValueError(errors[0])
        endpoints = np.asarray(endpoints, np.double)
        length = norm(endpoints[1] - endpoints[0])

        length_segments = [(norm(i[0]-endpoints[0])/length, norm(i[1]-endpoints[0])/length, i) for i in intersections]

        segments = self.pruneDuplicateSegments(endpoints, length_segments)
        return segments


    def buildLineIntegralGaugeOperators(self, lines, linesSegments, linesSegmentKeys=None):
        """ Build the linear algebra operators needed to compute the line integral gauges.

        The operators are local to each process, contributions are currently summed in the output functions.
        """

        #create lineIntegralGaugesVec to store contributions to all lines from this process
        self.lineIntegralGaugesVec = PETSc.Vec().create(comm=PETSc.COMM_SELF)
        self.lineIntegralGaugesVec.setSizes(len(lines))
        self.lineIntegralGaugesVec.setUp()

        # create lineIntegralGaugeMats to store coefficients mapping contributions from each field
        # to the line integral gauges
        self.lineIntegralGaugeMats = []

        if not self.isLineIntegralGauge:
            return

        # size of lineIntegralGaugeMats depends on number of local points for each field
        for pointGaugesVec in self.pointGaugeVecs:
            m = PETSc.Mat().create(comm=PETSc.COMM_SELF)
            m.setSizes([len(lines), pointGaugesVec.getSize()])
            m.setType('aij')
            m.setUp()
            self.lineIntegralGaugeMats.append(m)

        # Assemble contributions from each point in each line segment

        if linesSegmentKeys is None:
            linesSegmentKeys = [[(tuple(p1), tuple(p2)) for p1, p2 in segments] for segments in linesSegments]
        for lineIndex, (line, segments, segmentKeys) in enumerate(zip(self.lines, linesSegments, linesSegmentKeys)):
            field, endpoints = line
            fieldIndex = self.fields.index(field)

            # Trapezoid Rule to calculate coefficients here, from the values
            # at the ends of each segment, which are taken from inside the
            # segment's own element where the field is discontinuous
            for (p1, p2), keys in zip(segments, segmentKeys):
                segmentLength = np.linalg.norm(np.asarray(p2)-np.asarray(p1))
                for key in keys:
                    point_data = self.points[key]
                    # only assign coefficients for locally owned points
                    if field in point_data:
                        pointID = point_data[field]
                        self.lineIntegralGaugeMats[fieldIndex].setValue(lineIndex, pointID, segmentLength/2, addv=True)

        for m in self.lineIntegralGaugeMats:
            m.assemble()

    def snapGaugesToBoundingBox(self, relativeTolerance=1e-9):
        """ Moves gauge points and line endpoints that lie outside the mesh's
        bounding box by no more than relativeTolerance times its largest
        extent onto the box.

        A gauge "on the bottom" is typically computed differently from the
        mesh's own bottom and lands outside it by roundoff, where it would
        otherwise miss every element.  Only boundaries aligned with the
        bounding box are caught; further outside, points use their nearest
        node and lines are restricted to the part inside the mesh.

        .. figure:: /_generated/line_gauge_cases.svg
           :width: 100%

           Line gauges on a 2D mesh.  (1) A line gauge places points
           (red) where the line crosses element boundaries.  (2) A line
           outside the mesh by roundoff is snapped onto it.  (3) A line
           partly outside, and (4) a line over a hole in the mesh, warn
           and are restricted to the part inside (solid); the dashed
           part is not gauged and a line integral does not interpolate
           across it.
        """
        comm = Comm.get().comm.tompi4py()
        nodes = self.vertices
        # Allreduce on buffers: the pickle-based allreduce applies MIN/MAX
        # to the arrays as Python objects and fails for more than one entry
        low = np.empty(3)
        high = np.empty(3)
        comm.Allreduce(np.min(nodes, axis=0).astype(float), low, op=MPI.MIN)
        comm.Allreduce(np.max(nodes, axis=0).astype(float), high, op=MPI.MAX)
        tolerance = relativeTolerance * max(np.max(high - low), np.finfo(float).tiny)

        def snap(point):
            p = np.asarray(point, dtype=float)
            snapped = np.where((p < low) & (p >= low - tolerance), low, p)
            snapped = np.where((snapped > high) & (snapped <= high + tolerance), high, snapped)
            if np.any(snapped != p):
                logEvent("Gauge location [%g %g %g] moved onto the mesh boundary" % tuple(p), 3)
            return tuple(float(c) for c in snapped)

        if self.points:
            points = OrderedDict()
            for point, l_d in self.points.items():
                snapped = snap(point)
                if snapped in points:
                    points[snapped]['fields'].update(l_d['fields'])
                else:
                    points[snapped] = l_d
            self.points = points
        self.lines = [(field, (snap(endpoints[0]), snap(endpoints[1]))) for field, endpoints in self.lines]

    def attachModel(self, model, ar):
        """ Attach this gauge to the given simulation model.
        """
        from scipy import spatial
        self.model = model
        self.fieldNames = model.levelModelList[-1].coefficients.variableNames
        self.vertexFlags = model.levelModelList[-1].mesh.nodeMaterialTypes
        self.vertices = model.levelModelList[-1].mesh.nodeArray
        self.num_owned_nodes = model.levelModelList[-1].mesh.nNodes_global
        self.u = model.levelModelList[-1].u
        self.timeIntegration = model.levelModelList[-1].timeIntegration
        for field in self.fields:
            if field not in self.fieldNames:
                raise ValueError("gauge field %r is not a variable of this model; "
                                 "available fields: %s" % (field, list(self.fieldNames)))
            field_id = self.fieldNames.index(field)
            self.field_ids.append(field_id)
        self.nodes_kdtree = spatial.cKDTree(model.levelModelList[-1].mesh.nodeArray)
        self.snapGaugesToBoundingBox()
        linesSegments = []
        linesSegmentKeys = []
        for line in self.lines:
            lineSegments = self.getMeshIntersections(line)
            linesSegmentKeys.append(self.addLineGaugePoints(line, lineSegments))
            linesSegments.append(lineSegments)

        self.identifyMeasuredQuantities()

        self.buildGaugeComm()

        if self.isGaugeOwner:
            self.initOutputWriter()
            self.buildPointGaugeOperators()
            self.buildLineIntegralGaugeOperators(self.lines, linesSegments, linesSegmentKeys)
            if self.adapted or self.resumed:
              pass
            else:
              self.outputHeader()
            on_hot_start(self.resumeAt)
        return self

    def resumeAt(self, t):
        """Hot start from t: keep the file's header and rows up to t, and sample on the grid of the run it continues
        (first_output is that run's first sample), writing nothing more at or before t. Collective over gaugeComm."""
        times = None
        if self.gaugeComm.rank == 0 and self.resumed:
            header = self.headerLine()
            self.file.seek(0)
            if self.file.readline() == header:
                times = keep_rows_up_to(self.file, t)
            else:
                # the file was written for other gauges (locations, lines or fields changed): its rows would sit
                # under the wrong labels, so start it over with this run's header
                logEvent("Gauges: %s holds other gauges than this run's; starting it over" % (self.fileName,), 1)
                _rewrite(self.file, [header])
                times = []
        times = self.gaugeComm.bcast(times, root=0)
        if times is None:
            return
        self.resumeTime = t
        if times:
            self.first_output = times[0]
            self.last_output = times[-1]
            self.advanceSampleGrid(times[-1])

    def get_time(self):
        """ Returns the current model time"""

        return self.timeIntegration.tLast

    def headerLine(self):
        """The CSV header line of this gauge's file (on gaugeComm rank 0)."""
        parts = ["%10s" % ('time',)]
        if self.isPointGauge or self.isLineGauge:
            for field in self.fields:
                for quantity in self.globalMeasuredQuantities[field]:
                    location, gaugeProc, quantityID = quantity
                    parts.append(",%12s [%9.5g %9.5g %9.5g]" % (field, location[0], location[1], location[2]))
                    if len(location) > 3:
                        # a discontinuous field: the value in the element before or after the location
                        parts.append(" (%s)" % ("before" if location[3] < 0 else "after",))
        elif self.isLineIntegralGauge:
            for line in self.lines:
                parts.append(",%12s [%9.5g %9.5g %9.5g] - [%9.5g %9.5g %9.5g]" % (
                    line[0], line[1][0][0], line[1][0][1], line[1][0][2],
                             line[1][1][0], line[1][1][1], line[1][1][2]))
        return "".join(parts) + "\n"

    def outputHeader(self):
        """ Outputs a single header for a CSV style file to self.file"""

        assert self.isGaugeOwner
        if self.gaugeComm.rank == 0:
            self.file.write(self.headerLine())

    def outputRow(self, time):
        """ Outputs a single row of currently calculated gauge data to self.file"""

        assert self.isGaugeOwner

        if self.isPointGauge or self.isLineGauge:
            self.localQuantitiesBuf = np.concatenate([gaugesVec.getArray() for gaugesVec in
                                                      self.pointGaugeVecs]).astype(np.double)
            logEvent("Sending local array of type %s and shape %s to root on comm %s" % (
                str(self.localQuantitiesBuf.dtype), str(self.localQuantitiesBuf.shape), str(self.gaugeComm)), 9)
            if self.gaugeComm.rank == 0:
                logEvent("Receiving global array of type %s and shape %s on comm %s" % (
                str(self.localQuantitiesBuf.dtype), str(self.globalQuantitiesBuf.shape), str(self.gaugeComm)), 9)
            self.gaugeComm.Gatherv(sendbuf=[self.localQuantitiesBuf, MPI.DOUBLE],
                                   recvbuf=[self.globalQuantitiesBuf, (self.globalQuantitiesCounts, None),
                                            MPI.DOUBLE], root=0)
            self.gaugeComm.Barrier()
        if self.isLineIntegralGauge:
            lineIntegralGaugeBuf = self.lineIntegralGaugesVec.getArray()
            globalLineIntegralGaugeBuf = lineIntegralGaugeBuf.copy()
            self.gaugeComm.Reduce(lineIntegralGaugeBuf, globalLineIntegralGaugeBuf, op=MPI.SUM)
        else:
            globalLineIntegralGaugeBuf = []

        if self.gaugeComm.rank == 0:
            self.file.write("%25.15e" % time)
            if self.isPointGauge or self.isLineGauge:
                for id in self.globalQuantitiesMap:
                    self.file.write(", %43.18e" % (self.globalQuantitiesBuf[id],))
            if self.isLineIntegralGauge:
                for lineIntegralGauge in globalLineIntegralGaugeBuf:
                    self.file.write(", %80.18e" % (lineIntegralGauge))
            self.file.write('\n')
            # disable this for better performance, but risk of data loss on crashes
            self.file.flush()
        self.last_output = time
        self.advanceSampleGrid(time)

    def sampleDue(self, time):
        """ Returns True if a sample should be written at the given time.

        Samples are anchored to the grid first_output + k*sampleRate and
        each is taken at the first step at or after its grid time, so the
        mean sampling interval is sampleRate however it relates to the
        time step.  Comparing against last_output + sampleRate instead
        quantizes the interval to a multiple of the step and, through
        roundoff in the sum, skips samples whose grid time equals a step.
        """

        if self.next_output is None:
            return True
        return time >= self.next_output - 1e-9 * max(self.sampleRate, abs(time))

    def advanceSampleGrid(self, time):
        """ Moves next_output to the first grid time after the sample taken at time"""

        if self.first_output is None:
            self.first_output = time
        if self.sampleRate > 0:
            k = np.floor((time - self.first_output) / self.sampleRate
                         + 1e-9) + 1
            self.next_output = self.first_output + k * self.sampleRate

    def calculate_init(self):
        """Nothing: the initial conditions are not set when this is called, and
        the gauges would record zeros.  See calculate_initial_solution."""
        pass

    def calculate_initial_solution(self):
        """Records the initial time, now that the initial conditions are set."""
        self.calculate()

    def calculate(self):
        """ Computes current gauge values, updates open output files
        """

        if not self.isGaugeOwner:
            return

        time = self.get_time()
        logEvent("Gauges calculate called at time %g" % time, 4)
        # a hot start: the rows up to the restart instant are in the file already
        if self.resumeTime is not None and _at_or_before(time, self.resumeTime):
            return
        # check that gauge is in its active time region
        if self.activeTime is not None and (self.activeTime[0] > time or self.activeTime[1] < time):
            return

        # check that gauge is ready to be sampled again
        if not self.sampleDue(time):
            return

        for m, dofsVec, gaugesVec in zip(self.pointGaugeMats, self.dofsVecs, self.pointGaugeVecs):
            m.mult(dofsVec, gaugesVec)

        # this could be optimized out... but why?
        self.lineIntegralGaugesVec.zeroEntries()
        for m, dofsVec in zip(self.lineIntegralGaugeMats, self.pointGaugeVecs):
            m.multAdd(dofsVec, self.lineIntegralGaugesVec, self.lineIntegralGaugesVec)

        self.outputRow(time)

class BoundaryIntegralGauges(AV_base):
    """Boundary integrals of a flow model (RANS2P, RANS3PF) per boundary tag, written to one CSV file.

    The flow model's kernel integrates, per boundary flag and over owned faces, the wetted area, the pressure and
    viscous forces and the moment about a point, and reduces them over the ranks (they need the in-step fluxes, which
    an archive cannot reproduce). This gauge only reads them, after each accepted system step, so a failed and
    retried step writes no row, and a hot start keeps the rows up to the restart instant (open_history_file).

    :param tags: {name: boundary flag} (a ymf region tag that maps to a flag > 0), or a list of flags (named by
        their number).
    :param quantities: any of wetted_area, force_pressure, force_viscous, force (their sum), moment.
    :param moment_points: {name: point} about which the moment is reported; a tag without one uses the flow
        coefficients' barycenter of its flag (where the kernel takes moments). M_point = M_b + (b - point) x F.
    :param sampleRate: as for PointGauges (0: every step; otherwise the grid first sample + k * sampleRate).
    :param units: {quantity: units} overriding the SI defaults (per unit depth in 2D).
    :param fileName: the CSV file, in Profiling.logDir: a header row, then one row per sample, tag, quantity and
        component: time, tag, quantity, component, units, value.
    """
    QUANTITIES = ("wetted_area", "force_pressure", "force_viscous", "force", "moment")

    def __init__(self, tags, quantities=QUANTITIES, moment_points=None, sampleRate=0, activeTime=None, units=None,
                 fileName="boundary_integral_gauges.csv"):
        AV_base.__init__(self)
        if not isinstance(tags, dict):
            tags = OrderedDict((str(flag), flag) for flag in tags)
        for name, flag in tags.items():
            if int(flag) <= 0:
                raise ValueError("boundary integral gauge tag %r: flag %r is not a boundary condition flag (> 0)"
                                 % (name, flag))
        for quantity in quantities:
            if quantity not in self.QUANTITIES:
                raise ValueError("unknown boundary integral quantity %r; available: %s" % (quantity, self.QUANTITIES))
        self.tags = OrderedDict((name, int(flag)) for name, flag in tags.items())
        self.quantities = tuple(quantities)
        self.moment_points = {name: np.array(p, 'd') for name, p in (moment_points or {}).items()}
        unknown = set(self.moment_points) - set(self.tags)
        if unknown:
            raise ValueError("moment points for tags that are not gauged: %s" % sorted(unknown))
        self.sampleRate = sampleRate
        self.activeTime = activeTime
        self.units = dict(units or {})
        self.fileName = fileName
        self.file = None
        self.resumed = False
        self.resumeTime = None
        self.first_output = None
        self.last_output = None
        self.next_output = None

    def attachModel(self, model, ar):
        self.model = model
        lm = model.levelModelList[-1]
        self.coefficients = lm.coefficients
        for name in ("wettedAreas", "netForces_p", "netForces_v", "netMoments"):
            if not hasattr(self.coefficients, name):
                raise ValueError("BoundaryIntegralGauges needs a flow model with per-flag boundary integrals "
                                 "(RANS2P, RANS3PF), not %s" % type(lm).__name__)
        nFlags = self.coefficients.wettedAreas.shape[0]
        for name, flag in self.tags.items():
            if flag >= nFlags:
                raise ValueError("boundary integral gauge tag %r: no boundary has flag %d (largest %d)"
                                 % (name, flag, nFlags - 1))
        self.nd = lm.nSpace_global
        self.timeIntegration = lm.timeIntegration
        self.comm = Comm.get()
        default = {"wetted_area": "m^2" if self.nd == 3 else "m",
                   "force_pressure": "N" if self.nd == 3 else "N/m",
                   "force_viscous": "N" if self.nd == 3 else "N/m",
                   "force": "N" if self.nd == 3 else "N/m",
                   "moment": "N*m" if self.nd == 3 else "N"}
        default.update(self.units)
        self.units = default
        if self.comm.isMaster():
            self.fileName = os.path.join(Profiling.logDir, self.fileName)
            self.file, self.resumed = open_history_file(self.fileName)
            if not self.resumed:
                self.file.write("time,tag,quantity,component,units,value\n")
                self.file.flush()
        on_hot_start(self.resumeAt)
        return self

    def resumeAt(self, t):
        """Hot start from t: keep the rows up to t, sample on the grid of the run this one continues."""
        times = keep_rows_up_to(self.file, t) if self.comm.isMaster() and self.resumed else None
        times = self.comm.comm.tompi4py().bcast(times, root=0)
        if times is None:
            return
        self.resumeTime = t
        if times:
            self.first_output = times[0]
            self.last_output = times[-1]
            Gauges.advanceSampleGrid(self, times[-1])

    def calculate_init(self):
        """Nothing: the boundary integrals are computed by the flow model's residual, so none exist at t0."""
        pass

    def calculate_initial_solution(self):
        pass

    def values(self, name, flag):
        """(quantity, components, values) of one tag, as the flow model left them in its last residual."""
        c = self.coefficients
        Fp, Fv = c.netForces_p[flag, :self.nd], c.netForces_v[flag, :self.nd]
        components = ("x", "y", "z")[:self.nd]
        out = []
        for quantity in self.quantities:
            if quantity == "wetted_area":
                out.append((quantity, ("",), [c.wettedAreas[flag]]))
            elif quantity == "force_pressure":
                out.append((quantity, components, Fp))
            elif quantity == "force_viscous":
                out.append((quantity, components, Fv))
            elif quantity == "force":
                out.append((quantity, components, Fp + Fv))
            elif quantity == "moment":
                M = np.array(c.netMoments[flag, :], 'd')
                if name in self.moment_points:
                    b = np.zeros(3)
                    b[:len(c.barycenters[flag])] = c.barycenters[flag]
                    F = np.zeros(3)
                    F[:self.nd] = Fp + Fv
                    point = np.zeros(3)
                    point[:len(self.moment_points[name])] = self.moment_points[name]
                    M = M + np.cross(b - point, F)
                if self.nd == 3:
                    out.append((quantity, components, M))
                else:
                    out.append((quantity, ("z",), M[2:3]))
        return out

    def calculate(self):
        time = self.timeIntegration.tLast
        if self.resumeTime is not None and _at_or_before(time, self.resumeTime):
            return
        if self.activeTime is not None and (self.activeTime[0] > time or self.activeTime[1] < time):
            return
        if not Gauges.sampleDue(self, time):
            return
        if self.comm.isMaster():
            rows = []
            for name, flag in self.tags.items():
                for quantity, components, values in self.values(name, flag):
                    for component, value in zip(components, values):
                        rows.append("%25.15e,%s,%s,%s,%s,%25.15e\n" % (time, name, quantity, component,
                                                                         self.units[quantity], value))
            self.file.write("".join(rows))
            self.file.flush()
        self.last_output = time
        Gauges.advanceSampleGrid(self, time)
