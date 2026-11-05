"""
Classes for archiving numerical solution data

.. inheritance-diagram:: proteus.Archiver
   :parts: 1
"""
from . import Profiling
from .Profiling import logEvent
from . import Comm
import numpy
import os
import h5py

memory = Profiling.memory

#: The dtype of the index arrays the XDMF writers number themselves -- the
#: point numbering of quadrature, DG and particle meshes -- as opposed to a
#: mesh's own connectivity, whose dtype comes from the mesh. The writers
#: build those arrays with it and declare them with it, so the two cannot
#: disagree. It is the mesh index type, proteus_index_t.
from .cmeshTools import INDEX_DTYPE as GENERATED_INDEX_DTYPE

def xdmf_number_type(*arrays):
    """``{"DataType": ..., "Precision": ...}`` for a DataItem holding ``arrays``.

    Pass the array a DataItem describes, or its dtype. Pass several when the
    data is computed from them, as in ``offset + l2g``: the type is the one
    numpy gives that expression (``numpy.result_type``). The XDMF writers
    build a DataItem at every step but write its data only when the mesh
    changes, so they name what the data is made from, which always exists.

    The map from dtype to XDMF type is :func:`ymf.archive.data_item_for`'s,
    so the XDMF writers and the ymf writers cannot describe one array two
    ways, and a dtype with no XDMF type is refused rather than guessed.
    Before this, the mesh writers declared ``Int`` (4 bytes) and
    ``Float``/``8`` whatever they wrote.
    """
    from ymf.archive import data_item_for
    item = data_item_for(numpy.empty(0, numpy.result_type(*arrays)), inline=True)
    return {"DataType": item["data_type"], "Precision": str(item["precision"])}


class ArchiveCollection(object):
    """One temporal collection of the running archive.

    A collection is one finite-element space's sequence of grids, one per
    archived instant, and it holds only the *current* step's grid: the
    writers build it, :meth:`AR_base.sync` stores it in the HDF5 file and
    clears it, so the running document is never more than one step deep.
    The grid is a ymf grid dict (``topology``, ``geometry``, ``attributes``
    and an optional ``name``), built by :meth:`AR_base.write_grid`,
    :meth:`AR_base.write_topology`, :meth:`AR_base.write_geometry` and
    :meth:`AR_base.write_field`, and stored as it is.

    Before the XDMF retirement this was a ``<Grid GridType="Collection"
    CollectionType="Temporal">`` element whose last child was the current
    step; the writers compared that child's ``<Time>`` with the time being
    written to decide whether to start a new grid. :meth:`is_current` is
    that test.
    """
    def __init__(self, name):
        self.name = name
        self.grid = None
        self.time = None
        self.tCount = None

    def new_grid(self, t, tCount, name=None):
        """Start the grid of a new step at time ``t``. Returns it."""
        grid = {}
        if name is not None:
            grid["name"] = name
        grid["topology"] = None
        grid["geometry"] = None
        grid["attributes"] = []
        self.grid, self.time, self.tCount = grid, float(t), tCount
        return grid

    def is_current(self, t):
        """Whether the grid held is the one for time ``t``.

        False after :meth:`clear`, so the first writer of a step builds
        the grid and the others at the same time attach fields to it.
        """
        return self.grid is not None and self.time == float(t)

    def clear(self):
        """Drop the stored step's grid; the next writer starts a new one."""
        self.grid = None


class ArchiveDomain(object):
    """The running archive's temporal collections, in the order created.

    What the writers used to reach as ``ar.domain``, the ``<Domain>``
    element of the in-memory XDMF tree. Collections are found by name
    (:meth:`collection`) or by position (``domain[0]``, as
    ``MeshTools.Mesh.writeMeshXdmf`` does for the base mesh and its
    element-boundary mesh), and are never removed: the order they were
    created in is the order the archive records them.
    """
    def __init__(self):
        self.collections = []

    def __iter__(self):
        return iter(self.collections)

    def __len__(self):
        return len(self.collections)

    def __getitem__(self, index):
        return self.collections[index]

    def names(self):
        return [c.name for c in self.collections]

    def get(self, name):
        """The collection called ``name``, or None."""
        for c in self.collections:
            if c.name == name:
                return c
        return None

    def collection(self, name, create=True):
        """The collection called ``name``, created at the end if missing.

        With ``create=False`` a missing collection is None rather than
        made: a writer continuing a run (``init=False``) expects its
        collection to exist.
        """
        c = self.get(name)
        if c is None and create:
            c = ArchiveCollection(name)
            self.collections.append(c)
        return c

    def clear_step(self):
        for c in self.collections:
            c.clear()

class ArchiveFlags(object):
    EVERY_MODEL_STEP     = 0
    EVERY_USER_STEP      = 1
    EVERY_SEQUENCE_STEP  = 2
    UNDEFINED            =-1
#
class AR_base(object):
    """An archive: field data in ``<filename>.h5``, a ymf document beside it.

    Proteus writes no ``.xmf``. Each step's grid metadata is stored in the
    HDF5 file (:meth:`allGatherIncremental`), and :meth:`close` assembles
    ``<filename>.ymf`` from it. For ParaView or VisIt, ``ymf2xmf
    <filename>.ymf`` derives an ``.xmf`` that references the same HDF5
    datasets; ``gatherTimes`` rebuilds both from the ``.h5`` alone.

    ``self.domain`` is the running document: an :class:`ArchiveDomain` of
    named collections, each holding the ymf grid dict of the current step.
    The writers build those dicts through :meth:`write_grid`,
    :meth:`write_topology`, :meth:`write_geometry` and :meth:`write_field`,
    and :meth:`allGatherIncremental` stores them. Nothing in proteus builds
    or parses XML any more: ``ymf2xmf`` derives XDMF from the ``.ymf``, and
    the one XDMF proteus still reads -- the ``.xmf`` beside a <= 1.9.x
    archive, for the order of its collections -- goes through ``ymf.xdmf``.

    ``useGlobalXMF`` and ``gatherAtClose`` are accepted and ignored. They
    chose between one ``.xmf`` and one per rank, gathered at close; the
    HDF5 metadata makes that choice for a reader, and ``global_sync`` is
    what decides the layout of the data.

    **Modes.** One writer, with switches; a ``mode`` names a combination
    of them (:data:`MODES`), and a switch given explicitly overrides what
    the mode says:

    ``incremental``
        append each step to the ``.ymf`` as it is archived (the default),
        or write the ``.ymf`` only at close, from the HDF5 metadata.
    ``driver``
        how the ``.h5`` is used: ``"mpio"`` (the default), every rank
        writing its part of one file through parallel HDF5; ``"serial"``,
        one file that master alone opens and writes, the other ranks
        sending it their arrays (no parallel HDF5 needed, the same file,
        at the cost of the gather; such an archive cannot be hot started
        on several ranks); or ``"per-rank"``, every rank writing its own
        ``<filename>_p<rank>.h5`` with plain HDF5 and no communication at
        all. Per-rank files hold per-rank datasets, so this driver sets
        ``global_sync`` off.
    ``document``
        with per-rank files, where the grids are described: ``"global"``
        (the default), one ``<filename>.ymf`` whose every step is a spatial
        collection of the ranks' grids, each referencing its rank's file
        (the roster's layout B: one thing to open, built by master from
        the gathered grids); or ``"per-rank"``, one ``<filename>_p<rank>.ymf``
        per rank, each step a single grid, each rank writing its own with
        no collective step and no single point of failure (layout C). The
        single-file drivers take ``"global"`` only.
    ``includeLog``
        carry the run's log, and proteus's version and command line,
        in the ``.ymf`` under ``extra: proteus``.
    ``inline``
        no ``.h5`` at all: the arrays are copied into the ``.ymf``
        (``values`` on every data item), which is then self-contained.
        For problems small enough to want that.

    ======================  ===========  ======  ========  ==========  ======
    mode                    incremental  driver  document  includeLog  inline
    ======================  ===========  ======  ========  ==========  ======
    ``"without-a-net"``     off          mpio    global    off         off
    ``"split"`` (default)   on           mpio    global    off         off
    ``"kitchen-sink"``      on           mpio    global    on          off
    ``"ymf-only"``          on           --      global    off         on
    ======================  ===========  ======  ========  ==========  ======

    *Without a net* is the ``.h5`` alone during the run, the ``.ymf``
    derived from it at close -- and derivable again by ``gatherTimes``,
    which is the net. *Split* is the ``.h5`` plus a ``.ymf`` that is
    current at every step; the problem's own ``.ymf`` specification, the
    third file of that mode, is the ``pro`` command's. *Kitchen sink* adds
    the log and the run's provenance to the document. *Ymf only* is one
    file. The driver and document switches combine with any mode:
    ``driver="per-rank"`` with ``document="global"`` or ``"per-rank"`` are
    the two per-rank-file layouts, and ``driver="serial"`` the one for a
    build without parallel HDF5.
    """
    #: Names for combinations of the switches; see the class docstring.
    MODES = {
        "without-a-net": {"incremental": False},
        "split":         {},
        "kitchen-sink":  {"includeLog": True},
        "ymf-only":      {"inline": True},
    }
    #: The HDF5 drivers an archive can be opened with.
    DRIVERS = ("mpio", "serial", "per-rank")
    #: Where the grids of per-rank files are described.
    DOCUMENTS = ("global", "per-rank")
    #: HDF5 root attribute recording the document layout (:data:`DOCUMENTS`), so
    #: a reader of a per-rank file knows whether its steps are single grids.
    DOCUMENT_ATTR = "ymf_archive_document"

    def __init__(self,dataDir,filename,
                 useTextArchive=False,
                 gatherAtClose=True,
                 useGlobalXMF=True,
                 hotStart=False,
                 readOnly=False,
                 global_sync=True,
                 mode=None,
                 incremental=None,
                 driver=None,
                 document=None,
                 includeLog=None,
                 inline=None):
        comm=Comm.get()
        self.comm=comm
        self.rank = comm.rank()
        self.size = comm.size()
        self.dataDir=dataDir
        self.filename=filename
        self.readOnly = readOnly
        self.n_datasets = 0
        self.archived_domain = None
        #the running document: appended to a step at a time, never rewritten
        self._liveYmfStarted = False
        self._liveYmfWarned = False
        self.global_sync = global_sync
        switches = self.resolve_mode(mode, incremental=incremental, driver=driver,
                                     document=document, includeLog=includeLog,
                                     inline=inline)
        self.mode = switches["mode"]
        self.writeIncrementalYmf = switches["incremental"]
        self.driver = switches["driver"]
        self.document = switches["document"]
        self.includeLog = switches["includeLog"]
        self.inline = switches["inline"]
        #the arrays of an inline archive, by dataset name, on master; and
        #the document they are copied into, a step at a time
        self._inlineArrays = {}
        self._inlineDomain = None
        comm_world = self.comm.comm.tompi4py()
        self.domain = ArchiveDomain()
        if self.inline and (hotStart or readOnly):
            raise ValueError("an inline (ymf-only) archive has no .h5 to hot start from or read")
        if self.driver == "serial" and self.size > 1 and (hotStart or readOnly):
            raise ValueError(
                "a serial-driver archive is written by master alone, so the other "
                "ranks cannot read it: hot start or read it with the mpio driver")
        if self.driver == "per-rank" and self.global_sync:
            #a global array cannot span files: per-rank files hold per-rank
            #datasets, which is what global_sync off means
            logEvent("Archive %s: per-rank files hold per-rank datasets; "
                     "global_sync is off" % (filename,), level=2)
            self.global_sync = False
        if self.inline:
            #no file: the "reference" every data item carries is the dataset
            #name, which allGatherIncremental replaces with the values
            self.hdfFilename=filename+".h5"
            self.hdfFile=None
            self.dataItemFormat="XML"
        elif not useTextArchive:
            if hotStart:
                mode = "a"
            elif readOnly:
                mode = "r"
            else:
                mode = "w"
            if self.driver == "per-rank":
                self.hdfFilename = "%s_p%d.h5" % (filename, self.rank)
            else:
                self.hdfFilename=filename+".h5"
            path = os.path.join(self.dataDir,self.hdfFilename)
            if self.driver == "mpio":
                self.hdfFile=h5py.File(path, mode, driver="mpio", comm = comm_world)
            elif self.driver == "per-rank" or self.comm.isMaster():
                self.hdfFile=h5py.File(path, mode)
            else:
                self.hdfFile=None
            self.dataItemFormat="HDF"
            if mode == "w":
                self.comm.barrier()
        else:
            self.textDataDir=filename+"_Data"
            if readOnly:
                assert(os.path.exists(self.textDataDir))
            elif not os.path.exists(self.textDataDir):
                try:
                    os.mkdir(self.textDataDir)
                except:
                    self.textDataDir=""
            self.hdfFile=None
            self.dataItemFormat="XML"
        if hotStart and self.hdfFile is not None:
            self._seed_domain()

    @classmethod
    def resolve_mode(cls, mode=None, **switches):
        """The switch values a ``mode`` and explicit switches combine to.

        Returns ``{"mode", "incremental", "driver", "includeLog", "inline"}``.
        A switch passed as something other than None wins over the mode;
        the mode wins over the defaults (``"split"``'s). An unknown mode
        or driver is refused by name.
        """
        if mode is None:
            mode = "split"
        if mode not in cls.MODES:
            raise ValueError("archive mode %r is not one of %s"
                             % (mode, ", ".join(sorted(cls.MODES))))
        resolved = {"incremental": True, "driver": "mpio", "document": "global",
                    "includeLog": False, "inline": False}
        resolved.update(cls.MODES[mode])
        for name, value in switches.items():
            if name not in resolved:
                raise TypeError("unknown archive switch %r" % (name,))
            if value is not None:
                resolved[name] = value
        if resolved["driver"] not in cls.DRIVERS:
            raise ValueError("archive driver %r is not one of %s"
                             % (resolved["driver"], ", ".join(cls.DRIVERS)))
        if resolved["document"] not in cls.DOCUMENTS:
            raise ValueError("archive document %r is not one of %s"
                             % (resolved["document"], ", ".join(cls.DOCUMENTS)))
        if resolved["document"] == "per-rank" and resolved["driver"] != "per-rank":
            raise ValueError(
                "a per-rank document describes a per-rank file: document='per-rank' "
                "needs driver='per-rank', not %r" % (resolved["driver"],))
        if resolved["inline"] and resolved["driver"] != "mpio":
            raise ValueError("an inline (ymf-only) archive has no .h5, so it takes no driver")
        resolved["mode"] = mode
        return resolved

    @property
    def writesDocument(self):
        """Whether this rank writes a ``.ymf``: master, or every rank for per-rank documents."""
        return self.document == "per-rank" or self.comm.isMaster()

    @property
    def uniformSteps(self):
        """Whether a step of this rank's document is one grid (not a spatial collection)."""
        return self.global_sync or self.document == "per-rank"

    def documentPath(self):
        """This rank's ``.ymf``: ``<filename>.ymf``, or ``<filename>_p<rank>.ymf`` per rank."""
        if self.document == "per-rank":
            return os.path.join(self.dataDir, "%s_p%d.ymf" % (self.filename, self.rank))
        return os.path.join(self.dataDir, self.filename + ".ymf")

    @property
    def writesDatasets(self):
        """Whether the writers hand their arrays to the archive as datasets.

        True for every HDF5 driver and for an inline archive (whose
        "datasets" become the values in the document); False only for a
        text archive, which writes sidecar files itself. The writers used
        to test ``ar.hdfFile is not None``, which is wrong on the ranks
        of a serial-driver archive that hold no file.
        """
        return self.dataItemFormat == "HDF" or getattr(self, "inline", False)

    def _document_extra(self, with_log):
        """What the ``.ymf`` carries beside the domain: the run's provenance.

        Always proteus's version, the rank count and the mode -- what any
        two runs of the same computation share, so the documents of two
        such runs stay identical. With ``includeLog`` the command line as
        well, and the run's log (``with_log`` chooses, since the log is
        only complete at close).
        """
        import sys
        provenance = {"ranks": self.size, "archive_mode": self.mode,
                      "archive_driver": self.driver}
        if self.document == "per-rank":
            provenance["rank"] = self.rank
        try:
            from proteus import __version__
            provenance["version"] = str(__version__)
        except ImportError:
            pass
        if self.includeLog:
            provenance["command"] = " ".join(sys.argv)
        if with_log and self.includeLog:
            logFile = Profiling.logFile
            if logFile is not None and not logFile.closed:
                logFile.flush()
                provenance["log_file"] = os.path.basename(logFile.name)
                try:
                    with open(logFile.name, "r", errors="replace") as f:
                        provenance["log"] = f.read()
                except OSError as error:
                    provenance["log"] = "could not read %s: %s" % (logFile.name, error)
        return {"proteus": provenance}

    def _seed_domain(self):
        """Give a hot-started run the collections its writers append to.

        The writers find their temporal collection by name, or as the
        first one in the domain (``MeshTools.writeMeshXdmf``), so a
        continuing run needs the collections the archive already has, in
        the order they were written, and nothing else: the grids of earlier
        steps are stored by :meth:`sync` and not kept. This used to come
        from parsing the ``.xmf``; the HDF5 file records the names and
        their order itself.
        """
        for name in self._collection_names():
            self.domain.collection(name)
    def _collection_names(self):
        """The archive's temporal-collection names, in the order written.

        A version 2 archive records them in :data:`COLLECTIONS_ATTR`.
        Proteus <= 1.9.x recorded neither the names nor their order, and
        discovering them from the dataset names gives the names but not
        the order, which a hot start needs (``EBMesh`` would sort ahead of
        ``Mesh``). Those archives always came with an ``.xmf``, so the
        order is read from it, and only there.
        """
        raw = self.hdfFile.attrs.get(self.COLLECTIONS_ATTR)
        if raw is not None:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            return [name for name in raw.split("\n") if name]
        xmf = os.path.join(self.dataDir, self.filename + ".xmf")
        if not os.path.exists(xmf):
            raise IOError(
                "%s was written by proteus <= 1.9.x, which did not record "
                "the order of its time collections; hot starting from it "
                "needs the %s written beside it" % (self.hdfFilename, xmf))
        from ymf.xdmf import collection_names
        return collection_names(xmf)
    #: Bumped when the in-archive metadata layout changes incompatibly.
    #: Written as an attribute on the HDF5 root by
    #: :meth:`allGatherIncremental` and checked by
    #: :meth:`gatherAndWriteTimes`, so reading an archive written by an
    #: incompatible proteus fails with a clear message rather than a
    #: parse error deep inside the reader.
    METADATA_FORMAT_VERSION = 2
    #: HDF5 root attribute holding :data:`METADATA_FORMAT_VERSION`.
    METADATA_VERSION_ATTR = "ymf_archive_metadata_version"
    #: HDF5 root attribute holding the time-collection names, newline
    #: separated, in the order they were written.
    COLLECTIONS_ATTR = "ymf_archive_collections"
    #: HDF5 root attribute recording whether the archive was written as
    #: global arrays (1) or one grid per subdomain (0). Recorded so a
    #: reader need not be told: at one rank the two layouts produce
    #: metadata of the same shape but different XDMF, so it cannot be
    #: inferred from the data.
    GLOBAL_SYNC_ATTR = "ymf_archive_global_sync"

    def _metadata_version(self):
        """Which metadata layout this archive uses.

        Version 1 is XDMF ``<Grid>`` fragments, written by proteus up to
        and including 1.9.x. Version 2 is YAML describing a ymf grid dict.
        Both live in datasets named ``<collection>_<step>``, and a version 1
        archive carries no version attribute at all -- its absence is the
        marker.

        Version 1 is **read**, not refused: an existing XDMF archive must
        stay hot-startable and stay usable with the scripts. Only version 2
        is written.
        """
        found = self.hdfFile.attrs.get(self.METADATA_VERSION_ATTR)
        if found is None:
            return 1
        version = int(found)
        if version > self.METADATA_FORMAT_VERSION:
            raise ValueError(
                "%s holds grid metadata in format version %d; this proteus "
                "reads up to version %d"
                % (self.hdfFilename, version, self.METADATA_FORMAT_VERSION))
        return version

    def _check_metadata_version(self):
        """Kept for callers that only want the refusal on a future version."""
        self._metadata_version()

    def _discover_metadata_datasets(self):
        """Group the metadata datasets by collection, ordered by step.

        Needed for a version 1 archive, which records no collection names.
        Rather than pattern-matching names -- a collection name can itself
        contain underscores and digits, as Mesh_c0p2_Lagrange does -- a
        candidate is confirmed by what it *is*: a one-dimensional array of
        byte strings. Field data is numeric, so nothing else in the file
        looks like this.

        Returns ``{collection_name: [dataset_name_by_step, ...]}``.
        """
        import re

        pattern = re.compile(r"^(.+)_(\d+)$")
        found = {}
        for key, value in self.hdfFile.items():
            if getattr(value, "ndim", None) != 1 or value.dtype.kind != "S":
                continue
            match = pattern.match(key)
            if match is None:
                continue
            found.setdefault(match.group(1), []).append(
                (int(match.group(2)), key))
        return {name: [key for _, key in sorted(steps)]
                for name, steps in found.items()}

    @property
    def archived_times(self):
        """The times recorded in the archive, in the order written.

        Reads the assembled ymf domain rather than the in-memory XML tree.
        Before Phase 2 of the YMF campaign, ``gatherAndWriteTimes`` filled
        ``self.treeGlobal`` from the HDF5 metadata and callers counted
        ``<Time>`` elements in it; the domain dict is the representation
        now, and that tree is gone.

        Empty until :meth:`gatherAndWriteTimes` has run, i.e. until the
        archive is closed.
        """
        domain = self.load_archived_domain() if self.hdfFile is not None \
            else getattr(self, "archived_domain", None)
        if domain is None:
            return []
        if "times" in domain:
            return [instant["time"] for instant in domain["times"]]
        times = []
        for collection in archive_collections(domain):
            for step in collection["data"]:
                times.append(step["time"])
            #every collection covers the same instants, so one is enough
            break
        return times

    def assemble_domain(self, n_steps=None):
        """Build the ymf domain for the whole archive from the HDF5 metadata.

        Used by both ends of the archive's life: :meth:`gatherAndWriteTimes`
        calls it at close to produce the document it writes, and readers
        call it to get the archive's structure without walking XML. That
        shared use is the point -- there is one definition of what the
        archive contains, rather than a writer's idea and a reader's idea
        that can drift.

        ``n_steps`` bounds the search when the caller knows it (the writer
        does, from ``self.n_datasets``). A reader does not, so it is
        discovered: steps are written consecutively from 0, so counting up
        until a step is missing finds them all.

        Returns ``None`` when there is no HDF5 metadata to assemble from,
        which is the text-archive case.
        """
        from ymf.archive import (add_spatial_step, add_uniform_step,
                                 load_grid, new_domain)

        if self.hdfFile is None:
            return None
        version = self._metadata_version()

        raw = self.hdfFile.attrs.get(self.COLLECTIONS_ATTR)
        if raw is not None:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            collection_names = [name for name in raw.split("\n") if name]
            datasets = {name: None for name in collection_names}
        else:
            #A version 1 archive records no names; find them from the file.
            datasets = self._discover_metadata_datasets()
            collection_names = sorted(datasets)

        #Prefer the mode the archive recorded. A version 2 reader has no way
        #to infer it: at one rank a global write and a per-subdomain write
        #both leave metadata of shape (1,), but the first means a uniform
        #step and the second a spatial collection holding one grid. Version 1
        #records nothing, so there it *is* inferred -- see _v1_grids.
        recorded = self.hdfFile.attrs.get(self.GLOBAL_SYNC_ATTR)
        global_sync = self.global_sync if recorded is None else bool(int(recorded))
        #a per-rank document's steps are single grids whatever the dataset
        #layout: its file describes one subdomain
        document = self.hdfFile.attrs.get(self.DOCUMENT_ATTR)
        if isinstance(document, bytes):
            document = document.decode("utf-8")
        if document == "per-rank":
            global_sync = True

        #Time outside, space inside: walk the steps outermost, and let each
        #collection's grid join the instant at its time (add_*_step with a
        #collection name does that). A collection that is missing a step --
        #one first archived late -- is simply absent from that instant.
        domain = new_domain()
        live = {name: True for name in collection_names}
        seen = set()
        step = 0
        while n_steps is None or step < n_steps:
            found = False
            for collection_name in collection_names:
                if not live[collection_name]:
                    continue
                keys = datasets.get(collection_name)
                if keys is not None:
                    if step >= len(keys):
                        live[collection_name] = False
                        continue
                    dataset_name = keys[step]
                else:
                    dataset_name = self._metadata_dataset_name(
                        collection_name, step)
                    if dataset_name not in self.hdfFile:
                        #Discovering the end of a collection that has had steps
                        #ends it; one that has not had any yet is waiting for
                        #its first (a space first archived late).
                        if n_steps is None and collection_name in seen:
                            live[collection_name] = False
                        continue       # writer knows the count; tolerate a gap
                found = True
                seen.add(collection_name)
                grid_array = self.hdfFile[dataset_name]
                if version == 1:
                    t, grids, step_is_global = self._v1_grids(grid_array)
                else:
                    t = float(grid_array.attrs['Time'])
                    grids = [load_grid(grid_array[j].decode("utf-8"))
                             for j in range(grid_array.shape[0])]
                    step_is_global = global_sync
                if step_is_global:
                    #one already-global grid: its pieces are the archive
                    g = grids[0]
                    space = add_uniform_step(domain, t, g["topology"], g["geometry"],
                                             g.get("attributes", []),
                                             collection=collection_name)
                    #a mesh's own keys (its tagged facets, its provenance with
                    #the mesh fingerprint) travel with it
                    space.update((k, g[k]) for k in ("facets", "provenance") if k in g)
                else:
                    #one grid per rank, presented as a single instant
                    add_spatial_step(domain, t, grids,
                                     collection=collection_name)
            if n_steps is None and not found:
                break
            step += 1
        return domain if collection_names else None

    def _v1_grids(self, grid_array):
        """Read one step out of a version 1 (XDMF fragment) archive.

        Returns ``(time, grids, global_sync)``. Both the time and the
        layout are recovered from the fragments themselves, because
        proteus <= 1.9.x recorded neither reliably:

        * The dataset's ``Time`` attribute was set only on the
          per-subdomain path, so for a global archive the time has to come
          from the ``<Time>`` element inside the fragment.
        * That same asymmetry identifies the layout. The per-subdomain
          writer moved ``<Time>`` out of each grid before storing it, so a
          fragment that still has one was written globally. Shape alone is
          not enough -- both layouts give shape (1,) at a single rank.
        """
        from ymf.xdmf import read_grid_fragment

        parsed = [read_grid_fragment(grid_array[j])
                  for j in range(grid_array.shape[0])]
        grids = [grid for grid, _ in parsed]
        first_time = parsed[0][1]
        global_sync = first_time is not None
        if first_time is not None:
            t = first_time
        else:
            t = float(grid_array.attrs["Time"])
        return t, grids, global_sync

    def load_archived_domain(self):
        """The archive's domain, assembling it from HDF5 on first use.

        For readers -- hot start, mesh readers -- which open an existing
        archive and need its structure. Cached, since assembling walks
        every step's metadata.
        """
        if getattr(self, "archived_domain", None) is None:
            self.archived_domain = self.assemble_domain()
        return self.archived_domain

    def field_dataset(self, name, tCount):
        """The HDF5 dataset holding one field at one step, for reading back.

        Hot start supports exactly the two modes the archive can write:

        * **global** -- one assembled array per field, named
          ``<name>_t<step>``. Readable at any number of MPI tasks, since
          the array does not encode the decomposition.
        * **per-rank** -- one array per subdomain, named
          ``<name>_p<rank>_t<step>``. Readable only with the same number
          of tasks that wrote it, because rank *i* reads subdomain *i*.

        A third path used to exist: a separate ``<name>global.h5`` opened
        as ``hdfFileGlb`` and read with ``get_node``, which is a PyTables
        method that ``h5py.File`` does not have. It had been dead since the
        PyTables-to-h5py migration, and silently so -- the open was wrapped
        in a bare ``except: pass``, so it only surfaced when the file
        actually existed. Removed rather than repaired: a user writing
        either supported mode can hot start from it.

        Raises ``KeyError`` naming the mismatch rather than letting a
        missing dataset surface as a bare key error, since the usual cause
        is a per-rank archive being read at the wrong task count.
        """
        if self.global_sync:
            key = "{0:s}_t{1:d}".format(name, tCount)
        else:
            key = "{0:s}_p{1:s}_t{2:d}".format(
                name, repr(self.comm.rank()), tCount)
        try:
            return self.hdfFile["/" + key]
        except KeyError:
            pass
        if self.global_sync:
            raise KeyError(
                "%s has no dataset %r. This run is hot starting in global "
                "mode; the archive may have been written per-subdomain "
                "instead, in which case it must be read with global_sync "
                "off and the same number of MPI tasks."
                % (self.hdfFilename, key))
        import re
        pattern = re.compile(r"^%s_p(\d+)_t%d$" % (re.escape(name), tCount))
        written_by = sorted(int(m.group(1)) for m in
                            (pattern.match(k) for k in self.hdfFile) if m)
        raise KeyError(
            "%s has no dataset %r. A per-subdomain hot start needs the same "
            "number of MPI tasks that wrote the archive: this one holds %d "
            "subdomain(s) %s and this run has %d task(s). Either run with %d "
            "tasks or write the archive in global mode, which any task count "
            "can read."
            % (self.hdfFilename, key, len(written_by),
               written_by if len(written_by) < 8 else
               "0..%d" % (written_by[-1],),
               self.size, len(written_by) or self.size))

    def continue_from(self, tCount):
        """Make the archive continue after step ``tCount``, dropping later ones.

        A hot start resumes from the latest step at or before the requested
        time, which need not be the last archived. The steps after it
        belong to the run being continued, and the new run is about to
        write its own under the same numbers; left in the file, a shorter
        new run would leave the old tail to be read as part of the
        archive, and a later hot start could resume from it. Their grid
        metadata is deleted (the field arrays are replaced as the new run
        reaches them, or left unreferenced). Collective: every rank calls
        it with the same step.
        """
        self.n_datasets = tCount + 1
        self.archived_domain = None
        if self.hdfFile is not None:
            for name in self._collection_names():
                index = self.n_datasets
                while True:
                    key = self._metadata_dataset_name(name, index)
                    if key not in self.hdfFile:
                        break
                    del self.hdfFile[key]
                    index += 1
        self.comm.barrier()

    def _metadata_dataset_name(self, collection_name, index):
        """Name of the HDF5 dataset holding one collection's step metadata."""
        return (collection_name + "_" + str(index)).replace(" ", "_")

    #: Whether each step is appended to the ``.ymf`` as it is produced, so
    #: that the archive of record is current during the run rather than only
    #: after a clean close. Appending is constant work per step -- see
    #: :meth:`_appendLiveStep` -- so this is on by default. Set False to
    #: restore the write-at-close-only behaviour.
    writeIncrementalYmf = True

    def _appendLiveStep(self, live_steps):
        """Append this timestep to the running document.

        One record for the instant, appended and flushed. Master already
        holds the grids, so this costs no HDF5 read and no YAML parse, and
        because it appends rather than rewrites it costs the same whether
        the archive holds ten steps or ten thousand.

        The first call writes the header document -- the archive version and
        an empty domain -- which the records are then appended to. The
        header is written to a temporary file and renamed so that a reader
        arriving mid-write never sees half of it; the records after it need
        no such care, since each ends with the document-end marker that
        tells a reader it is complete.

        Failures are logged and swallowed, the import among them: an ymf
        too old to have ``append_step`` should cost the run its live
        document, not the run. The authoritative document is still written
        at close from the HDF5 file, and the steps already in the archive
        matter more than this one.
        """
        path = self.documentPath()
        try:
            from ymf.archive import append_step, new_domain, write_ymf
            if not self._liveYmfStarted:
                tmp = path + ".writing"
                write_ymf(new_domain(), tmp,
                          extra=self._document_extra(with_log=False))
                os.replace(tmp, path)
                self._liveYmfStarted = True
            #One record per instant: every space archived at this time, so a
            #run killed here has the whole instant or none of it.
            instants = []
            for _, name, t, grids in live_steps:
                if self.uniformSteps:
                    g = grids[0]
                    space = {"topology": g["topology"],
                             "geometry": g["geometry"],
                             "attributes": g.get("attributes", [])}
                else:
                    space = {"spatial_collection": grids}
                if instants and instants[-1]["time"] == t \
                        and name not in instants[-1]["spaces"]:
                    instants[-1]["spaces"][name] = space
                else:
                    instants.append({"time": t, "spaces": {name: space}})
            for instant in instants:
                append_step(path, instant)
        except Exception as error:
            if not self._liveYmfWarned:
                self._liveYmfWarned = True
                logEvent("Could not update %s during the run (%s); the "
                         "archive is unaffected and the document will be "
                         "written at close"
                         % (os.path.basename(path), error), level=1)

    def gatherAndWriteTimes(self):
        """Assemble the whole archive document and write it out.

        Reads back the per-step, per-rank grid metadata that
        :meth:`allGatherIncremental` stashed in the HDF5 file as YAML,
        builds a single ymf domain from it, and writes that domain as the
        ``.ymf`` archive of record. An ``.xmf`` for viewers is derived from
        it on demand by ``ymf2xmf``, rather than written by every run.
        """
        from ymf.archive import write_ymf

        if self.inline:
            #an inline archive has no HDF5 metadata: the document was built
            #a step at a time, values and all
            domain = self._inlineDomain
        else:
            domain = self.assemble_domain()
        #keep the assembled domain: it is the archive's representation, and
        #callers that used to inspect the XML tree want this instead
        self.archived_domain = domain
        if domain is None:
            #a text archive keeps its arrays in files and its grid metadata
            #nowhere: there is no HDF5 file to have stored it in
            logEvent("Text archive %s: no grid metadata to write"
                     % (self.filename,))
            return
        ymf_path = self.documentPath()
        write_ymf(domain, ymf_path, extra=self._document_extra(with_log=True))
        logEvent("Wrote YMF archive %s (ymf2xmf %s writes an .xmf for viewers)"
                 % (ymf_path, os.path.basename(ymf_path)))

    def close(self):
        logEvent("Closing Archive")
        if self.writesDocument and not self.readOnly:
            self.gatherAndWriteTimes()
        if self.hdfFile is not None:
            self.hdfFile.close()
        logEvent("Done Closing Archive")
    def allGatherIncremental(self):
        """Stash this timestep's grid metadata in the HDF5 file.

        Each rank turns its own ``<Grid>`` element into a ymf grid dict,
        serializes it to YAML, and the collection of them is written into a
        dataset named ``<collection>_<step>``.
        :meth:`gatherAndWriteTimes` reads them back at close and assembles
        the whole document.

        Three things changed here relative to the XML version:

        * The payload is YAML describing a ymf grid dict, not an XDMF
          fragment. Parsing an element into a dict at this boundary is a
          bridge -- the writers still build elements (Phase 1 of the
          campaign is unfinished), and when they build dicts directly this
          parse disappears.
        * Grids cross MPI as dicts rather than as pickled
          ``ElementTree.Element`` objects.
        * The collective agreement on the dataset width is an
          ``allreduce(MAX)`` over each rank's own encoded length, not a
          ``Bcast`` of a maximum only master could compute. Master no
          longer needs every grid in hand before the dataset can be sized.
          A fixed width is still required: parallel HDF5 rejects
          variable-length datatypes outright ("Parallel IO does not support
          writing VL or region reference datatypes yet"), so
          ``h5py.string_dtype()`` is not an option here.
        """
        from mpi4py import MPI
        from ymf.archive import dump_grid

        logEvent("Gathering Archive Time step")
        self.comm.barrier()
        Domain = self.domain

        comm_world = self.comm.comm.tompi4py()
        if self.hdfFile is not None:
            self.hdfFile.attrs[self.METADATA_VERSION_ATTR] = \
                self.METADATA_FORMAT_VERSION
            #Record the collection names, in order, so a reader can find the
            #metadata without inferring names from dataset spellings. A
            #collection name may itself contain underscores and digits
            #(Mesh_c0p2_Lagrange), so pattern-matching <name>_<step> against
            #the file's keys is guessy where this is not. Written by every
            #rank with the same value, since attribute writes are collective.
            self.hdfFile.attrs[self.COLLECTIONS_ATTR] = "\n".join(
                c.name for c in Domain)
            self.hdfFile.attrs[self.GLOBAL_SYNC_ATTR] = \
                1 if self.global_sync else 0
            self.hdfFile.attrs[self.DOCUMENT_ATTR] = self.document

        #who holds this step's grids, and writes them: with one file, master
        #gathers the ranks' grids (or holds the one global grid); with
        #per-rank files every rank records its own grid in its own file, and
        #for a global document master gathers them as well, to describe the
        #spatial collection the ranks' files make up
        gather = (not self.global_sync) and (
            self.driver != "per-rank" or self.document == "global")
        live_steps = []
        for i, TemporalGridCollection in enumerate(Domain):
            local_grid = TemporalGridCollection.grid
            if local_grid is None:
                raise RuntimeError(
                    "archive collection %r has no grid for this step: every "
                    "collection is written at every archived instant"
                    % (TemporalGridCollection.name,))
            if local_grid["topology"] is None or local_grid["geometry"] is None:
                raise RuntimeError(
                    "archive collection %r: the grid for t=%r has no %s"
                    % (TemporalGridCollection.name, TemporalGridCollection.time,
                       "topology" if local_grid["topology"] is None else "geometry"))
            t = TemporalGridCollection.time
            TimeAttrib = repr(t)

            #the grids this rank records: the step's, on master (one per
            #rank, or the one global grid); its own, on a rank with its own
            #file
            if gather:
                gathered = comm_world.gather(local_grid)
            else:
                gathered = None
            if self.comm.isMaster():
                grid_dicts = gathered if gather else [local_grid]
            else:
                grid_dicts = [local_grid] if self.driver == "per-rank" else None
            if self.inline:
                #no file: master replaces every dataset reference with the
                #values it was handed and keeps the step in the document
                if self.comm.isMaster():
                    grid_dicts = [self._inlineGrid(g) for g in grid_dicts]
                    self._inlineStep(TemporalGridCollection.name, t, grid_dicts)
                    live_steps.append((i, TemporalGridCollection.name,
                                       t, grid_dicts))
                continue
            payloads = ([dump_grid(g).encode("utf-8") for g in grid_dicts]
                        if grid_dicts is not None else [])

            #every rank must create the dataset with the same width, so agree
            #on it collectively. Each rank contributes the length it knows.
            #(Per-rank files need no agreement; the allreduce is harmless.)
            local_width = max((len(pl) for pl in payloads), default=0)
            width = comm_world.allreduce(local_width, op=MPI.MAX)
            width = max(int(width), 1)

            dataset_name = self._metadata_dataset_name(
                TemporalGridCollection.name, self.n_datasets)
            if self.hdfFile is not None:
                n_rows = len(payloads) if self.driver == "per-rank" else (
                    self.size if gather else 1)
                grid_data = self._dataset(dataset_name, (n_rows,),
                                          '|S'+str(width))
                grid_data.attrs['Time'] = TimeAttrib
                for j, payload in enumerate(payloads):
                    grid_data[j] = payload
            if self.writesDocument and self.hdfFile is not None:
                #the writer of a document already holds its step's grids, so
                #the running document costs no extra read and no extra parse
                live_steps.append((i, TemporalGridCollection.name,
                                   t, grid_dicts))
        self.n_datasets += 1
        if live_steps and self.writeIncrementalYmf:
            self._appendLiveStep(live_steps)
        logEvent("Done Gathering Archive Time Step")
    def sync(self):
        logEvent("Syncing Archive",level=3)
        memory()
        self.allGatherIncremental()
        #the step is stored; drop its grids so the domain stays one step deep
        self.domain.clear_step()
        #the barriers are collective: every rank takes them, whether or not
        #it holds the file (with the serial driver only master does)
        self.comm.barrier()
        if self.hdfFile is not None:
            self.hdfFile.flush()
        self.comm.barrier()
        logEvent("Done Syncing Archive",level=3)
        logEvent(memory("Syncing Archive"),level=4)
    def create_dataset_async(self,name,data):
        """Write each rank's own array under its own name.

        With the mpio driver every rank creates every dataset (creation is
        collective) and fills its own. With the serial driver, or inline,
        the arrays are gathered to master, which writes (or keeps) them.
        """
        comm_world = self.comm.comm.tompi4py()
        if self.driver == "per-rank":
            self._store(name, numpy.asarray(data))
            return
        if self.driver == "mpio" and not self.inline:
            metadata = comm_world.allgather((name,data.shape,data.dtype))
            for i,m in enumerate(metadata):
                dataset = self._dataset(m[0], m[1], m[2])
                if i == self.rank:
                    dataset[:] = data
            return
        pieces = comm_world.gather((name, data))
        if self.comm.isMaster():
            for piece_name, piece in pieces:
                self._store(piece_name, numpy.asarray(piece))
    def create_dataset_sync(self,name,offsets,data):
        """Write one global array, each rank contributing the slice it owns.

        With the mpio driver the dataset is created collectively and every
        rank writes its slice. With the serial driver, or inline, the
        slices are gathered to master and assembled there. Per-rank files
        cannot hold a global array, which is why that driver turns
        global_sync off.
        """
        if self.driver == "per-rank":
            raise RuntimeError(
                "a global array (%s) cannot be written across per-rank files; "
                "the per-rank driver turns global_sync off, so this is a writer "
                "that ignored ar.global_sync" % (name,))
        if self.driver == "mpio" and not self.inline:
            dataset = self._dataset(name, tuple([offsets[-1]]+list(data.shape[1:])),
                                    data.dtype)
            dataset[offsets[self.rank]:offsets[self.rank+1]] = data
            return
        comm_world = self.comm.comm.tompi4py()
        pieces = comm_world.gather(data)
        if self.comm.isMaster():
            full = numpy.empty(tuple([int(offsets[-1])]+list(data.shape[1:])), data.dtype)
            for r, piece in enumerate(pieces):
                full[int(offsets[r]):int(offsets[r+1])] = piece
            self._store(name, full)
    def _store(self, name, array):
        """Master keeps an assembled array: in the file, or for the inline document."""
        if self.inline:
            self._inlineArrays[name] = array
        else:
            self._dataset(name, array.shape, array.dtype)[...] = array

    def _dataset(self, name, shape, dtype):
        """The dataset ``name`` with this shape and type, creating it if need be.

        A hot start continues from a step that may not be the last one
        archived, and the steps after it are written again under the names
        they had. The dataset is reused when it fits and replaced when it
        does not (a different mesh, rank count, or metadata width), instead
        of failing as ``create_dataset`` does or, as the metadata writer
        did, writing a longer payload into a shorter fixed-width string
        and truncating it. Creation and deletion are collective under the
        mpio driver, and every rank calls this with the same arguments.
        """
        dtype = numpy.dtype(dtype)
        if name in self.hdfFile:
            existing = self.hdfFile[name]
            if existing.shape == tuple(shape) and existing.dtype == dtype:
                return existing
            del self.hdfFile[name]
        return self.hdfFile.create_dataset(name, shape=tuple(shape), dtype=dtype)

    def _inlineGrid(self, grid):
        """A copy of ``grid`` with every dataset reference replaced by its values.

        The writers describe each array by the dataset name it would have
        had in the ``.h5``; an inline archive has no ``.h5``, so master
        swaps the values it was handed (:meth:`_store`) into the document.
        A reference with no values behind it is a writer that described an
        array it never wrote, and is refused rather than left dangling.
        """
        import copy
        from ymf.archive import data_item_for
        g = copy.deepcopy(grid)
        prefix = self.hdfFilename + ":/"

        def swap(owner):
            item = owner["data_item"]
            reference = item.get("data")
            if reference is None or not reference.startswith(prefix):
                return
            name = reference[len(prefix):]
            if name not in self._inlineArrays:
                raise RuntimeError(
                    "inline archive: the document refers to dataset %r but no "
                    "values were written for it" % (name,))
            owner["data_item"] = data_item_for(self._inlineArrays[name], inline=True,
                                               dimensions=item["dimensions"])
        swap(g["topology"])
        swap(g["geometry"])
        for a in g.get("attributes", []):
            swap(a)
        return g

    def _inlineStep(self, name, t, grids):
        """Append one step to the inline document, creating its collection at need."""
        from ymf.archive import add_spatial_step, add_uniform_step, new_domain
        if self._inlineDomain is None:
            self._inlineDomain = new_domain()
        if self.global_sync:
            g = grids[0]
            add_uniform_step(self._inlineDomain, t, g["topology"], g["geometry"],
                             g.get("attributes", []), collection=name)
        else:
            add_spatial_step(self._inlineDomain, t, grids, collection=name)

    def write_field(self, grid, name, data, tCount,
                    center="Node", rank="Scalar", dimensions=None,
                    sync_offsets=None, sync_data=None, dataset=None,
                    text_stem=None, write_data=True):
        """Attach one field to ``grid`` and write its array. Returns the data item.

        This is the single place an attribute and its data item get built.
        Before it existed, that eight-line block was written out by hand at
        roughly fifty call sites across Archiver.py, FemTools.py and
        MeshTools.py, each repeating the same four-way branch on
        ``global_sync`` and ``hdfFile``. Two consequences of that
        duplication worth knowing about:

        * ``DataType`` and ``Precision`` were hardcoded per call site --
          usually ``Float``/``8``. A ``float32`` field was therefore
          described to consumers as 8-byte, which a viewer reads as
          garbage. Here they come from the array's own dtype via
          :func:`ymf.archive.data_item_for`, so they cannot disagree with
          the data.
        * The sidecar reference for the text fallback had to be attached
          to the right DataItem by hand, and at ``Archiver.py:1475`` it
          wasn't -- the reference for one field landed on another field's
          DataItem. Here it is part of the data item this call creates,
          so that class of bug is not expressible.

        Parameters
        ----------
        grid : dict
            The ymf grid dict to attach the attribute to, from
            :meth:`write_grid`.
        name : str
            Field name, used for both the attribute and the dataset.
        data : numpy.ndarray
            The values to write. In the ``global_sync`` case this is the
            full local array and ``sync_data`` is the owned slice actually
            written; ``data`` is still used for its dtype.
        dimensions : sequence of int, optional
            The logical shape to declare. Required when it differs from
            ``data.shape``, which it does for ``global_sync`` writes: the
            DataItem describes the *global* array while each rank
            contributes only the part it owns.
        sync_offsets, sync_data :
            Passed to :meth:`create_dataset_sync` for ``global_sync``
            writes.
        dataset : str, optional
            Overrides the HDF5 dataset name. The default follows the
            convention most call sites use -- ``<name>_t<tCount>`` when
            synchronized, ``<name>_p<rank>_t<tCount>`` otherwise -- but a
            few writers predate it and use their own. Those pass their name
            explicitly so that converting them does not rename datasets
            inside existing archives.
        text_stem : str, optional
            Overrides the sidecar filename stem for the text fallback,
            which defaults to ``<name><tCount>``.
        write_data : bool
            False attaches the attribute without writing its array, for a
            field whose array was written at an earlier step and is
            referenced again (the mesh writers' local-to-global maps).
        """
        from ymf.archive import attribute, data_item_for

        if dataset is not None:
            dataset_name = dataset
        elif self.global_sync:
            dataset_name = "{0:s}_t{1:d}".format(name, tCount)
        else:
            dataset_name = "{0:s}_p{1:s}_t{2:d}".format(
                name, repr(self.comm.rank()), tCount)

        if text_stem is None:
            text_stem = "{0:s}{1:d}".format(name, tCount)
        #textDataDir only exists when the archive was opened with
        #useTextArchive=True, so it must not be touched on the HDF5 path
        text_path = ("{0:s}/{1:s}.txt".format(self.textDataDir, text_stem)
                     if not self.writesDatasets else None)

        item = data_item_for(
            data,
            data="{0:s}:/{1:s}".format(self.hdfFilename, dataset_name)
            if self.writesDatasets else None,
            include="./" + text_path if not self.writesDatasets else None,
            dimensions=list(data.shape) if dimensions is None else dimensions,
            #On a genuinely parallel collective write the DataItem
            #describes the assembled global array while `data` is this
            #rank's slice, present only for its dtype, so the declared
            #dimensions are larger than the array by design and checking
            #them would reject a correct write.
            #
            #The check stays on for size==1, where global and local
            #coincide: that is the case where a declared size that does not
            #match the data means a genuinely malformed archive, and it is
            #how the phi_s corruption was found. Narrowing rather than
            #disabling keeps that safety net for the common serial run.
            check=not (self.global_sync and self.size > 1),
        )
        grid["attributes"].append(
            attribute(name, item, center=center, attribute_type=rank))

        if not write_data:
            return item
        if self.writesDatasets:
            if self.global_sync:
                self.create_dataset_sync(dataset_name,
                                         offsets=sync_offsets,
                                         data=sync_data)
            else:
                self.create_dataset_async(dataset_name, data=data)
        else:
            assert not self.global_sync, \
                "global_sync is not supported with text heavy data"
            numpy.savetxt(text_path, data)
        return item

    def write_grid(self, collection, grid_name, t, tCount):
        """Start the grid of one timestep of a mesh. Returns the grid dict.

        ``collection`` is the :class:`ArchiveCollection` of the space; the
        grid becomes its current one, and the collection records the time
        (what used to be the ``<Time>`` element). The writers then fill
        it with :meth:`write_topology`, :meth:`write_geometry` and
        :meth:`write_field`.
        """
        return collection.new_grid(t, tCount, grid_name)

    def write_topology(self, grid, topology_type, n_elements, dimensions,
                       dataset, text_stem, nodes_per_element=None, *, dtype):
        """Set the grid's topology, naming its array. Returns the data item.

        The array reference is filled in here, but the array itself is
        **not** written: mesh datasets are only created when ``init or
        meshChanged``, while the reference is written on every pass. The
        caller keeps that decision, and the data preparation that goes with
        it, because it differs at every call site (element-to-node maps,
        DG node duplication, particle index ranges).

        ``dtype`` is the type of the array the reference names: the array
        itself, its dtype, or a tuple of what it is computed from (see
        :func:`xdmf_number_type`). It has no default, so a caller cannot
        describe an array without saying what it holds.
        """
        from ymf.archive import topology
        item = self._mesh_data_item(dimensions, dataset, text_stem, dtype)
        grid["topology"] = topology(topology_type, n_elements, item,
                                    nodes_per_element=nodes_per_element)
        return item

    def write_geometry(self, grid, dimensions, dataset, text_stem,
                       geometry_type="XYZ", *, dtype):
        """Set the grid's geometry, naming its array. Returns the data item.

        Same division of labour, and the same ``dtype``, as
        :meth:`write_topology`.
        """
        from ymf.archive import geometry
        item = self._mesh_data_item(dimensions, dataset, text_stem, dtype)
        grid["geometry"] = geometry(item, geometry_type=geometry_type)
        return item

    def _mesh_data_item(self, dimensions, dataset, text_stem, dtype):
        """The data item shared by :meth:`write_topology`/:meth:`write_geometry`.

        Unlike :meth:`write_field` this cannot read the dtype off the array,
        because the array often does not exist yet at this point -- the
        reference is written on every pass and the data only when the mesh
        changed. So the caller names what the array is made from.
        """
        from ymf.archive import data_item
        dtypes = dtype if isinstance(dtype, tuple) else (dtype,)
        kind = xdmf_number_type(*dtypes)
        if self.writesDatasets:
            reference = {"data": "{0:s}:/{1:s}".format(self.hdfFilename, dataset)}
        else:
            reference = {"include": "./{0:s}/{1:s}.txt".format(self.textDataDir,
                                                               text_stem)}
        return data_item([int(d) for d in dimensions],
                         data_type=kind["DataType"],
                         precision=int(kind["Precision"]), **reference)

def archive_collections(domain):
    """A domain's spaces as ``[{"name", "data": [step, ...]}]``, in either format.

    ymf's archive format 1 holds one time collection per space
    (``time_collections``); format 2 holds one list of instants with the
    spaces inside each (``times``). Readers that think one space at a time
    use this and work with a ymf of either kind. A format 1 domain is
    returned as it is; a format 2 one is turned into the same shape, the
    grids shared, not copied. Spaces come out in the order they first
    appear.
    """
    if not domain:
        return []
    if "times" not in domain:
        return domain.get("time_collections", [])
    order, steps = [], {}
    for instant in domain["times"]:
        for name, space in instant["spaces"].items():
            if name not in steps:
                order.append(name)
                steps[name] = []
            steps[name].append(dict(space, time=instant["time"]))
    return [{"name": name, "data": steps[name]} for name in order]


def readArchiveDomain(filename, dataDir='.'):
    """The ymf domain of an existing archive, read from its HDF5 file.

    For tools that hold a filename rather than a live archive object. The
    ``.h5`` is self-describing -- it carries the metadata format version,
    the time-collection names and whether the run wrote global arrays or
    one grid per subdomain -- so no ``.xmf`` or ``.ymf`` sidecar is needed
    and no flags have to be supplied.

    That is what makes the sidecars recoverable: an archive whose ``.ymf``
    or ``.xmf`` was lost or truncated can be rebuilt from the ``.h5``
    alone.
    """
    import h5py

    base = os.path.join(dataDir, filename)
    hdfFilename = filename + ".h5"
    if not os.path.exists(base + ".h5") and os.path.exists(base + "_p0.h5"):
        #per-rank files with a global document: master's file holds the
        #gathered grids of every rank. With a per-rank document it holds
        #rank 0's subdomain only, so it cannot stand for the archive.
        with h5py.File(base + "_p0.h5", "r") as hdfFile:
            document = hdfFile.attrs.get(AR_base.DOCUMENT_ATTR)
        if isinstance(document, bytes):
            document = document.decode("utf-8")
        if document == "per-rank" and os.path.exists(base + "_p1.h5"):
            raise ValueError(
                "%s_p<rank>.h5 is a per-rank archive with one document per rank: "
                "%s_p0.h5 describes rank 0's subdomain only, and no file holds the "
                "whole domain; read each rank's file on its own" % (base, base))
        base, hdfFilename = base + "_p0", filename + "_p0.h5"
    archive = AR_base.__new__(AR_base)
    archive.hdfFilename = hdfFilename
    archive.global_sync = True          # overridden by the recorded value
    archive.archived_domain = None
    archive._liveYmfStarted = False
    archive._liveYmfWarned = False
    with h5py.File(base + ".h5", "r") as hdfFile:
        archive.hdfFile = hdfFile
        domain = archive.assemble_domain()
    archive.hdfFile = None
    return domain


XdmfArchive=AR_base

########################################################################
#writers for the function spaces whose nodes are not the mesh's own
class XdmfWriter(object):
    """
    collect functionality for writing data to the archive

    Writer is supposed to keep track of grid collection (temporal collection)

    as well as current grid under grid collection where data belong,
    since data are associated with a grid of specific type
    (e.g., P1 Lagrange, P2 Lagrange, elementQuadrature dictionary, ...)

    Each ``writeMeshXdmf_*`` method describes one space's mesh as a ymf
    grid dict through the archive's builders (``write_grid``,
    ``write_topology``, ``write_geometry``) and writes the arrays behind it
    when ``init or meshChanged``; the ``write*Xdmf_*`` methods attach fields
    to the current grid through ``write_field``. The names survive from
    when these methods built XDMF elements; what they build now is the
    archive's own representation, which ``ymf2xmf`` turns into XDMF.
    """
    def __init__(self,shareSingleGrid=True,arGridCollection=None,arGrid=None,arTime=None):
        self.arGridCollection = arGridCollection #collection of "grids" (at least one per time level)
        self.arGrid           = arGrid #grid in collection that data should be associated with
        self.arTime           = arTime #kept for callers that read it; the collection records the time
        self.shareSingleGrid  = shareSingleGrid

    def setGridCollectionAndGridElements(self,init,ar,arGrid,t,spaceSuffix):
        """Find or create this space's collection; returns the grid name.

        On ``init`` the collection ``Mesh<spaceSuffix>`` is created (or
        found, if another writer of the same space made it); otherwise it
        is looked up, and a writer that already holds one keeps it. With
        ``arGrid`` given -- another writer's grid for this space at this
        time -- that grid becomes the current one, so a second component
        attaches its fields to the grid the first one built.
        """
        found = ar.domain.collection("Mesh"+spaceSuffix, create=init)
        if found is not None:
            self.arGridCollection = found
        assert self.arGridCollection is not None, \
            "no collection Mesh%s in the archive; it is created on init" % (spaceSuffix,)
        if arGrid is not None:
            self.arGrid = arGrid
        return "Grid"+spaceSuffix

    def _needsGrid(self, t):
        """Whether this space still lacks a grid for time ``t``.

        The first writer of a step builds the grid; the others at the same
        time reuse it. Replaces the comparison of the current grid's
        ``<Time>`` with ``"{0:e}".format(t)`` that every writer made.
        """
        if self.arGridCollection.is_current(t):
            self.arGrid = self.arGridCollection.grid
            return False
        return True

    def writeMeshXdmf_elementQuadrature(self,ar,mesh,spaceDim,x,t=0.0,
                                       init=False,meshChanged=False,arGrid=None,tCount=0):
        return self.writeMeshXdmf_quadraturePoints(ar,mesh,spaceDim,x,t=t,quadratureType="q",
                                                   init=init,meshChanged=meshChanged,arGrid=arGrid,tCount=tCount)
    def writeMeshXdmf_elementBoundaryQuadrature(self,ar,mesh,spaceDim,x,t=0.0,
                                                init=False,meshChanged=False,arGrid=None,tCount=0):
        return self.writeMeshXdmf_quadraturePoints(ar,mesh,spaceDim,x,t=t,quadratureType="ebq_global",
                                                   init=init,meshChanged=meshChanged,arGrid=arGrid,tCount=tCount)
    def writeMeshXdmf_exteriorElementBoundaryQuadrature(self,ar,mesh,spaceDim,x,t=0.0,
                                                        init=False,meshChanged=False,arGrid=None,tCount=0):
        return self.writeMeshXdmf_quadraturePoints(ar,mesh,spaceDim,x,t=t,quadratureType="ebqe",
                                                   init=init,meshChanged=meshChanged,arGrid=arGrid,tCount=tCount)

    def writeScalarXdmf_elementQuadrature(self,ar,u,name,tCount=0,init=True):
        qualifiedName = name.replace(' ','_')+"_elementQuadrature"
        return self.writeScalarXdmf_quadrature(ar,u,qualifiedName,tCount=tCount,init=init)
    def writeVectorXdmf_elementQuadrature(self,ar,u,name,tCount=0,init=True):
        qualifiedName = name.replace(' ','_')+"_elementQuadrature"
        return self.writeVectorXdmf_quadrature(ar,u,qualifiedName,tCount=tCount,init=init)
    def writeTensorXdmf_elementQuadrature(self,ar,u,name,tCount=0,init=True):
        qualifiedName = name.replace(' ','_')+"_elementQuadrature"
        return self.writeTensorXdmf_quadrature(ar,u,qualifiedName,tCount=tCount,init=init)
    #
    def writeScalarXdmf_elementBoundaryQuadrature(self,ar,u,name,tCount=0,init=True):
        qualifiedName = name.replace(' ','_')+"_elementBoundaryQuadrature"
        return self.writeScalarXdmf_quadrature(ar,u,qualifiedName,tCount=tCount,init=init)
    def writeVectorXdmf_elementBoundaryQuadrature(self,ar,u,name,tCount=0,init=True):
        qualifiedName = name.replace(' ','_')+"_elementBoundaryQuadrature"
        return self.writeVectorXdmf_quadrature(ar,u,qualifiedName,tCount=tCount,init=init)
    def writeTensorXdmf_elementBoundaryQuadrature(self,ar,u,name,tCount=0,init=True):
        qualifiedName = name.replace(' ','_')+"_elementBoundaryQuadrature"
        return self.writeTensorXdmf_quadrature(ar,u,qualifiedName,tCount=tCount,init=init)
    #
    def writeScalarXdmf_exteriorElementBoundaryQuadrature(self,ar,u,name,tCount=0,init=True):
        qualifiedName = name.replace(' ','_')+"_exteriorElementBoundaryQuadrature"
        return self.writeScalarXdmf_quadrature(ar,u,qualifiedName,tCount=tCount,init=init)
    def writeVectorXdmf_exteriorElementBoundaryQuadrature(self,ar,u,name,tCount=0,init=True):
        qualifiedName = name.replace(' ','_')+"_exteriorElementBoundaryQuadrature"
        return self.writeVectorXdmf_quadrature(ar,u,qualifiedName,tCount=tCount,init=init)
    def writeTensorXdmf_exteriorElementBoundaryQuadrature(self,ar,u,name,tCount=0,init=True):
        qualifiedName = name.replace(' ','_')+"_exteriorElementBoundaryQuadrature"
        return self.writeTensorXdmf_quadrature(ar,u,qualifiedName,tCount=tCount,init=init)


    def writeMeshXdmf_quadraturePoints(self,ar,mesh,spaceDim,x,t=0.0,quadratureType="q",
                                       init=False,meshChanged=False,arGrid=None,tCount=0):
        """
        write out quadrature point mesh for quadrature points that are unique to
        either elements or elementBoundaries

        quadrature type
           q  --> element
           ebq_global --> element boundary
           ebqe       --> exterior element boundary
        """
        if quadratureType == "q":
            spaceSuffix = "_elementQuadrature"
        elif quadratureType == "ebq_global":
            spaceSuffix = "_elementBoundaryQuadrature"
        elif quadratureType == "ebqe":
            spaceSuffix = "_exteriorElementBoundaryQuadrature"
        else:
            raise RuntimeError("quadratureType = %s not recognized" % quadratureType)

        #write out basic geometry if not already done?
        mesh.writeMeshXdmf(ar,"Spatial_Domain",t,init,meshChanged,tCount=tCount)
        #now try to write out a mesh that is a collection of points per element

        gridName = self.setGridCollectionAndGridElements(init,ar,arGrid,t,spaceSuffix)

        assert len(x.shape) == 3 #make sure have right type of dictionary
        self.mesh = mesh
        if self._needsGrid(t):
            Xdmf_NodesPerElement = x.shape[1]
            if ar.global_sync:
                assert ar.writesDatasets, "global_sync not supported with text  heavy data"
                Xdmf_NumberOfElements= mesh.globalMesh.nElements_global
                elements_dataset = 'elements'+spaceSuffix+str(tCount)
                nodes_dataset    = 'nodes'+spaceSuffix+str(tCount)
            else:
                Xdmf_NumberOfElements= x.shape[0]
                elements_dataset = 'elements'+str(ar.rank)+spaceSuffix+str(tCount)
                nodes_dataset    = 'nodes'+str(ar.rank)+spaceSuffix+str(tCount)
            Xdmf_NodesGlobal     = Xdmf_NumberOfElements*Xdmf_NodesPerElement
            elements_stem = 'elements'+spaceSuffix+str(tCount)
            nodes_stem    = 'nodes'+spaceSuffix+str(tCount)

            self.arGrid = ar.write_grid(self.arGridCollection, gridName, t, tCount)
            ar.write_topology(self.arGrid, "Polyvertex", Xdmf_NumberOfElements,
                              [Xdmf_NumberOfElements, Xdmf_NodesPerElement],
                              elements_dataset, elements_stem,
                              nodes_per_element=Xdmf_NodesPerElement,
                              dtype=GENERATED_INDEX_DTYPE)
            ar.write_geometry(self.arGrid, [Xdmf_NodesGlobal, 3],
                              nodes_dataset, nodes_stem, dtype=x)
            if init or meshChanged:
                if ar.global_sync:
                    q_l2g = numpy.arange(mesh.globalMesh.elementOffsets_subdomain_owned[ar.rank]*Xdmf_NodesPerElement,
                                         mesh.globalMesh.elementOffsets_subdomain_owned[ar.rank+1]*Xdmf_NodesPerElement,
                                         dtype=GENERATED_INDEX_DTYPE).reshape((mesh.nElements_owned,Xdmf_NodesPerElement))
                    ar.create_dataset_sync(elements_dataset,
                                           offsets=mesh.globalMesh.elementOffsets_subdomain_owned,
                                           data = q_l2g)
                    ar.create_dataset_sync(nodes_dataset,
                                           offsets=mesh.globalMesh.elementOffsets_subdomain_owned*Xdmf_NodesPerElement,
                                           data = x[:mesh.nElements_owned].reshape((mesh.nElements_owned*Xdmf_NodesPerElement,3)))
                else:
                    q_l2g = numpy.arange(Xdmf_NumberOfElements*Xdmf_NodesPerElement,dtype=GENERATED_INDEX_DTYPE).reshape((Xdmf_NumberOfElements,Xdmf_NodesPerElement))
                    if ar.writesDatasets:
                        ar.create_dataset_async(elements_dataset, data = q_l2g)
                        ar.create_dataset_async(nodes_dataset, data = x.flat[:])
                    else:
                        numpy.savetxt(ar.textDataDir+"/"+elements_stem+".txt",q_l2g,fmt='%d')
                        numpy.savetxt(ar.textDataDir+"/"+nodes_stem+".txt",x.flat[:])
        return self.arGrid
    #def
    def writeScalarXdmf_quadrature(self,ar,u,name,tCount=0,init=True):
        assert len(u.shape) == 2
        if ar.global_sync:
            #the DataItem covers every element's quadrature points globally;
            #this rank contributes only the elements it owns
            n_owned = self.mesh.nElements_owned
            ar.write_field(self.arGrid, name, u, tCount,
                           dimensions=[self.mesh.globalMesh.nElements_global*u.shape[1]],
                           sync_offsets=self.mesh.globalMesh.elementOffsets_subdomain_owned*u.shape[1],
                           sync_data=u[:n_owned].reshape((n_owned*u.shape[1],)))
        else:
            ar.write_field(self.arGrid, name, u.flat[:], tCount,
                           dimensions=[u.shape[0]*u.shape[1]])

    def writeVectorXdmf_quadrature(self,ar,u,name,tCount=0,init=True):
        assert len(u.shape) == 3
        #XDMF vectors are 3-component because the points are 3D, so the
        #components the field doesn't have stay zero
        Xdmf_StorageDim = 3
        Xdmf_NumberOfComponents = u.shape[2]
        if ar.global_sync:
            n_owned = self.mesh.nElements_owned
            n_local = n_owned*u.shape[1]
            tmp = numpy.zeros((n_local,Xdmf_StorageDim),'d')
            tmp[:,:Xdmf_NumberOfComponents] = numpy.reshape(
                u[:n_owned].flat,(n_local,Xdmf_NumberOfComponents))
            ar.write_field(self.arGrid, name, tmp, tCount, rank="Vector",
                           dimensions=[self.mesh.globalMesh.nElements_global*u.shape[1],
                                       Xdmf_StorageDim],
                           sync_offsets=self.mesh.globalMesh.elementOffsets_subdomain_owned*u.shape[1],
                           sync_data=tmp)
        else:
            Xdmf_NodesGlobal = u.shape[0]*u.shape[1]
            tmp = numpy.zeros((Xdmf_NodesGlobal,Xdmf_StorageDim),'d')
            tmp[:,:Xdmf_NumberOfComponents] = numpy.reshape(
                u.flat,(Xdmf_NodesGlobal,Xdmf_NumberOfComponents))
            ar.write_field(self.arGrid, name, tmp, tCount, rank="Vector",
                           dimensions=[Xdmf_NodesGlobal,Xdmf_StorageDim])

    def writeTensorXdmf_quadrature(self,ar,u,name,tCount=0,init=True):
        """
        TODO make faster tmp creation
        """
        assert len(u.shape) == 4
        #XDMF tensors are 9-component; a 2x2 tensor occupies the leading
        #corner of a 3x3 and the rest stays zero
        Xdmf_NumberOfComponents = u.shape[2]*u.shape[3]

        def pack(rows, source):
            tmp = numpy.zeros((rows,9),'d')
            for k in range(rows):
                for i in range(u.shape[2]):
                    for j in range(u.shape[3]):
                        tmp.flat[k*9 + i*3 + j] = source.flat[
                            k*Xdmf_NumberOfComponents + i*u.shape[2] + j]
            return tmp

        if ar.global_sync:
            #sized by the elements this rank owns, matching
            #writeVectorXdmf_quadrature. The previous version sized tmp by
            #the *global* element count while reading from the local array,
            #which indexes past the end of u.
            n_owned = self.mesh.nElements_owned
            tmp = pack(n_owned*u.shape[1], u[:n_owned])
            ar.write_field(self.arGrid, name, tmp, tCount, rank="Tensor",
                           dimensions=[self.mesh.globalMesh.nElements_global*u.shape[1],9],
                           #.globalMesh was missing here; every other call
                           #site in this file reads the offsets off the
                           #global mesh, and a subdomain mesh has no such
                           #attribute
                           sync_offsets=self.mesh.globalMesh.elementOffsets_subdomain_owned*u.shape[1]*9,
                           sync_data=tmp)
        else:
            Xdmf_NodesGlobal = u.shape[0]*u.shape[1]
            tmp = pack(Xdmf_NodesGlobal, u)
            ar.write_field(self.arGrid, name, tmp, tCount, rank="Tensor",
                           dimensions=[Xdmf_NodesGlobal,9])


    def writeMeshXdmf_DGP1Lagrange(self,ar,name,mesh,spaceDim,dofMap,CGDOFMap,t=0.0,
                                   init=False,meshChanged=False,arGrid=None,tCount=0):
        """
        TODO Not tested yet
        """
        comm = Comm.get()
        #assert False, "Not tested"
        #write out basic geometry if not already done?
        mesh.writeMeshXdmf(ar,"Spatial_Domain",t,init,meshChanged,tCount=tCount)
        #now try to write out a mesh that matches DG P1 layout

        spaceSuffix = "_dgp1_Lagrange"
        gridName = self.setGridCollectionAndGridElements(init,ar,arGrid,t,spaceSuffix)

        if self._needsGrid(t):
            if spaceDim == 1:
                Xdmf_ElementTopology = "Polyline"
            elif spaceDim == 2:
                Xdmf_ElementTopology = "Triangle"
            elif spaceDim == 3:
                Xdmf_ElementTopology = "Tetrahedron"
            if ar.global_sync:
                assert ar.writesDatasets, "Global sync  not implemented for text heavy data"
                n_elements       = mesh.globalMesh.nElements_global
                elements_dims    = [n_elements, mesh.nNodes_element]
                nodes_dims       = [dofMap.nDOF_all_processes, 3]
                elements_dataset = 'elements'+spaceSuffix+str(tCount)
                nodes_dataset    = 'nodes'+spaceSuffix+str(tCount)
                elements_dtype   = (dofMap.dof_offsets_subdomain_owned, dofMap.l2g)
            else:
                n_elements       = mesh.nElements_global
                elements_dims    = [n_elements, mesh.nNodes_element]
                nodes_dims       = [dofMap.nDOF, 3]
                elements_dataset = 'elements'+str(ar.rank)+spaceSuffix+str(tCount)
                nodes_dataset    = 'nodes'+str(ar.rank)+spaceSuffix+str(tCount)
                elements_dtype   = dofMap.l2g
            elements_stem = 'elements'+spaceSuffix+str(tCount)
            nodes_stem    = 'nodes'+spaceSuffix+str(tCount)

            self.arGrid = ar.write_grid(self.arGridCollection, gridName, t, tCount)
            ar.write_topology(self.arGrid, Xdmf_ElementTopology, n_elements,
                              elements_dims, elements_dataset, elements_stem,
                              dtype=elements_dtype)
            ar.write_geometry(self.arGrid, nodes_dims, nodes_dataset, nodes_stem,
                              dtype=mesh.nodeArray)
            if init or meshChanged:
                if ar.global_sync:
                    nElements_owned = (mesh.globalMesh.elementOffsets_subdomain_owned[ar.rank+1]
                                       -mesh.globalMesh.elementOffsets_subdomain_owned[ar.rank])
                    #this will fail if elements_dgp1 already exists
                    ar.create_dataset_sync(elements_dataset,
                                           offsets = mesh.globalMesh.elementOffsets_subdomain_owned,
                                           data = dofMap.dof_offsets_subdomain_owned[ar.rank]+dofMap.l2g[:nElements_owned])
                    dgnodes = numpy.zeros((dofMap.dof_offsets_subdomain_owned[ar.rank+1]-dofMap.dof_offsets_subdomain_owned[ar.rank],3),mesh.nodeArray.dtype)
                    for eN in range(nElements_owned):
                        for nN in range(mesh.nNodes_element):
                            dgnodes[dofMap.l2g[eN,nN],:]=mesh.nodeArray[mesh.elementNodesArray[eN,nN]]
                    ar.create_dataset_sync(nodes_dataset,
                                           offsets=dofMap.dof_offsets_subdomain_owned,
                                           data = dgnodes)
                else:
                    dgnodes = numpy.zeros((dofMap.nDOF,3),mesh.nodeArray.dtype)
                    for eN in range(mesh.nElements_global):
                        for nN in range(mesh.nNodes_element):
                            dgnodes[dofMap.l2g[eN,nN],:]=mesh.nodeArray[mesh.elementNodesArray[eN,nN]]
                    if ar.writesDatasets:
                        #this will fail if elements_dgp1 already exists
                        ar.create_dataset_async(elements_dataset, data = dofMap.l2g)
                        ar.create_dataset_async(nodes_dataset, data = dgnodes)
                    else:
                        numpy.savetxt(ar.textDataDir+"/"+elements_stem+".txt",dofMap.l2g,fmt='%d')
                        numpy.savetxt(ar.textDataDir+"/"+nodes_stem+".txt",dgnodes)
        return self.arGrid
    #def
    def writeMeshXdmf_DGP2Lagrange(self,ar,name,mesh,spaceDim,dofMap,CGDOFMap,t=0.0,
                                   init=False,meshChanged=False,arGrid=None,tCount=0):
        #write out basic geometry if not already done?
        mesh.writeMeshXdmf(ar,"Spatial_Domain",t,init,meshChanged,tCount=tCount)
        #now try to write out a mesh that matches DG P1 layout
        #first duplicate geometry points etc then try to save space
        spaceSuffix = "_dgp2_Lagrange"
        gridName = self.setGridCollectionAndGridElements(init,ar,arGrid,t,spaceSuffix)

        if self._needsGrid(t):
            if spaceDim == 1:
                Xdmf_ElementTopology = "Edge_3"
            elif spaceDim == 2:
                Xdmf_ElementTopology = "Tri_6"
            elif spaceDim == 3:
                Xdmf_ElementTopology = "Tet_10"
            if ar.global_sync:
                assert ar.writesDatasets, "global_sync not implemented for  text heavy data"
                n_elements       = mesh.globalMesh.nElements_global
                elements_dims    = [n_elements, dofMap.l2g.shape[-1]]
                nodes_dims       = [dofMap.nDOF_all_processes, 3]
                elements_dataset = 'elements'+spaceSuffix+str(tCount)
                nodes_dataset    = 'nodes'+spaceSuffix+str(tCount)
                elements_dtype   = (dofMap.dof_offsets_subdomain_owned, dofMap.l2g)
            else:
                n_elements       = mesh.nElements_global
                elements_dims    = list(dofMap.l2g.shape)
                nodes_dims       = [dofMap.nDOF, 3]
                elements_dataset = 'elements'+str(ar.rank)+spaceSuffix+str(tCount)
                nodes_dataset    = 'nodes'+str(ar.rank)+spaceSuffix+str(tCount)
                elements_dtype   = dofMap.l2g
            elements_stem = 'elements'+spaceSuffix+str(tCount)
            nodes_stem    = 'nodes'+spaceSuffix+str(tCount)

            self.arGrid = ar.write_grid(self.arGridCollection, gridName, t, tCount)
            ar.write_topology(self.arGrid, Xdmf_ElementTopology, n_elements,
                              elements_dims, elements_dataset, elements_stem,
                              dtype=elements_dtype)
            ar.write_geometry(self.arGrid, nodes_dims, nodes_dataset, nodes_stem,
                              dtype=CGDOFMap.lagrangeNodesArray)
            if init or meshChanged:
                if ar.global_sync:
                    nElements_owned = (mesh.globalMesh.elementOffsets_subdomain_owned[ar.rank+1]
                                       -mesh.globalMesh.elementOffsets_subdomain_owned[ar.rank])
                    ar.create_dataset_sync(elements_dataset,
                                           offsets = mesh.globalMesh.elementOffsets_subdomain_owned,
                                           data = dofMap.dof_offsets_subdomain_owned[ar.rank]+dofMap.l2g[:nElements_owned])
                    dgnodes = numpy.zeros((dofMap.dof_offsets_subdomain_owned[ar.rank+1]-dofMap.dof_offsets_subdomain_owned[ar.rank],3),CGDOFMap.lagrangeNodesArray.dtype)
                    for eN in range(nElements_owned):
                        for i in range(dofMap.l2g.shape[1]):
                            #lagrange nodes hold all nodes
                            dgnodes[dofMap.l2g[eN,i],:]= CGDOFMap.lagrangeNodesArray[CGDOFMap.l2g[eN,i],:]
                    ar.create_dataset_sync(nodes_dataset,
                                           offsets = dofMap.dof_offsets_subdomain_owned,
                                           data = dgnodes)
                else:
                    dgnodes = numpy.zeros((dofMap.nDOF,3),CGDOFMap.lagrangeNodesArray.dtype)
                    for eN in range(mesh.nElements_global):
                        for i in range(dofMap.l2g.shape[1]):
                            dgnodes[dofMap.l2g[eN,i],:]= CGDOFMap.lagrangeNodesArray[CGDOFMap.l2g[eN,i],:]
                    if ar.writesDatasets:
                        ar.create_dataset_async(elements_dataset, data = dofMap.l2g)
                        ar.create_dataset_async(nodes_dataset, data = dgnodes)
                    else:
                        numpy.savetxt(ar.textDataDir+"/"+elements_stem+".txt",dofMap.l2g,fmt='%d')
                        numpy.savetxt(ar.textDataDir+"/"+nodes_stem+".txt",dgnodes)
        return self.arGrid

    def writeMeshXdmf_C0P2Lagrange(self,ar,name,mesh,spaceDim,dofMap,t=0.0,
                                   init=False,meshChanged=False,arGrid=None,tCount=0):
        """
        TODO: test new lagrangeNodes convention for 2d,3d
        """
        #write out basic geometry if not already done?
        mesh.writeMeshXdmf(ar,"Spatial_Domain",t,init,meshChanged,tCount=tCount)
        spaceSuffix = "_c0p2_Lagrange"
        gridName = self.setGridCollectionAndGridElements(init,ar,arGrid,t,spaceSuffix)
        if self._needsGrid(t):
            if spaceDim == 1:
                Xdmf_ElementTopology = "Edge_3"
            elif spaceDim == 2:
                Xdmf_ElementTopology = "Tri_6"
            elif spaceDim == 3:
                Xdmf_ElementTopology = "Tet_10"
            lagrangeNodesArray = dofMap.lagrangeNodesArray

            #the synchronized case describes the assembled global arrays and
            #names its datasets without a rank; the per-rank case describes
            #this subdomain and carries the rank in the dataset name. Sidecar
            #names omit the rank in both.
            if ar.global_sync:
                n_elements       = mesh.globalMesh.nElements_global
                elements_dims    = [n_elements, dofMap.l2g.shape[-1]]
                nodes_dims       = [dofMap.nDOF_all_processes, 3]
                elements_dataset = 'elements'+spaceSuffix+str(tCount)
                nodes_dataset    = 'nodes'+spaceSuffix+str(tCount)
            else:
                n_elements       = mesh.nElements_global
                elements_dims    = list(dofMap.l2g.shape)
                nodes_dims       = [dofMap.nDOF, 3]
                elements_dataset = 'elements'+str(ar.rank)+spaceSuffix+str(tCount)
                nodes_dataset    = 'nodes'+str(ar.rank)+spaceSuffix+str(tCount)
            elements_stem = 'elements'+spaceSuffix+str(tCount)
            nodes_stem    = 'nodes'+spaceSuffix+str(tCount)

            self.arGrid = ar.write_grid(self.arGridCollection, gridName, t, tCount)
            ar.write_topology(self.arGrid, Xdmf_ElementTopology, n_elements,
                              elements_dims, elements_dataset, elements_stem,
                              dtype=(dofMap.subdomain2global if ar.global_sync
                                     else dofMap.l2g))
            ar.write_geometry(self.arGrid, nodes_dims,
                              nodes_dataset, nodes_stem,
                              dtype=lagrangeNodesArray)

            if not ar.writesDatasets:
                assert not ar.global_sync, \
                    "global_sync is not implemented for text heavy data"
                if init or meshChanged:
                    numpy.savetxt(ar.textDataDir+"/"+elements_stem+".txt",
                                  dofMap.l2g,fmt='%d')
                    numpy.savetxt(ar.textDataDir+"/"+nodes_stem+".txt",
                                  lagrangeNodesArray)
                return self.arGrid

            if spaceDim == 3:
                #proteus stores 3d dof as
                #|n0,n1,n2,n3|(n0,n1),(n1,n2),(n2,n3)|(n0,n2),(n1,n3)|(n0,n3)|
                #xdmf wants them as
                #|n0,n1,n2,n3|(n0,n1),(n1,n2),(n0,n2) (n0,n3),(n1,n3) (n2,n3)|
                import copy
                element_nodes = copy.deepcopy(dofMap.l2g)
                for eN in range(mesh.nElements_global):
                    element_nodes[eN,4+2] = dofMap.l2g[eN,4+3]
                    element_nodes[eN,4+3] = dofMap.l2g[eN,4+5]
                    element_nodes[eN,4+5] = dofMap.l2g[eN,4+2]
            else:
                element_nodes = dofMap.l2g

            #references were written above on every pass; the arrays only when
            #the mesh actually changed
            if init or meshChanged:
                if ar.global_sync:
                    owned = dofMap.dof_offsets_subdomain_owned
                    ar.create_dataset_sync(elements_dataset,
                                           offsets = mesh.globalMesh.elementOffsets_subdomain_owned,
                                           data = dofMap.subdomain2global[element_nodes[:mesh.nElements_owned]])
                    ar.create_dataset_sync(nodes_dataset,
                                           offsets = owned,
                                           data = lagrangeNodesArray[:owned[ar.rank+1]-owned[ar.rank]])
                else:
                    ar.create_dataset_async(elements_dataset, data = element_nodes)
                    ar.create_dataset_async(nodes_dataset, data = lagrangeNodesArray)
        return self.arGrid

    def writeMeshXdmf_C0Q2Lagrange(self,ar,name,mesh,spaceDim,dofMap,t=0.0,init=False,meshChanged=False,arGrid=None,tCount=0):
        """
        TODO: test new lagrangeNodes convention for 2d,3d
        """
        #write out basic geometry if not already done?
        #mesh.writeMeshXdmf(ar,"Spatial_Domain",t,init,meshChanged,tCount=tCount)
        spaceSuffix = "_c0q2_Lagrange"
        gridName = self.setGridCollectionAndGridElements(init,ar,arGrid,t,spaceSuffix)
        if self._needsGrid(t):
            if spaceDim == 1:
                print("No writeMeshXdmf_C0Q2Lagrange for 1D")
                return 0
            elif spaceDim == 2:
                Xdmf_ElementTopology = "Quadrilateral"
                e2s=[ [0,4,8,7], [7,8,6,3],  [4,1,5,8], [8,5,2,6] ]
                nsubelements=4

                l2g = numpy.zeros((4*mesh.nElements_global,4),dofMap.l2g.dtype)
                for eN in range(mesh.nElements_global):
                    dofs=dofMap.l2g[eN,:]
                    for i in range(4): #loop over subelements
                        for j in range(4): # loop over nodes of subelements
                            l2g[4*eN+i,j] = dofs[e2s[i][j]]
            elif spaceDim == 3:
                Xdmf_ElementTopology = "Hexahedron"

                e2s=[[0,8,20,11,12,21,26,24], [8,1,9,20,21,13,22,26], [11,20,10,3,24,26,23,15], [20,9,2,10,26,22,14,23],
                     [12,21,26,24,4,16,25,19], [21,13,22,26,16,5,17,25], [24,26,23,15,19,25,18,7], [26,22,14,23,25,17,6,18] ]

                l2g = numpy.zeros((8*mesh.nElements_global,8),dofMap.l2g.dtype)
                nsubelements=8
                for eN in range(mesh.nElements_global):
                    dofs=dofMap.l2g[eN,:]
                    for i in range(8): #loop over subelements
                        for j in range(8): # loop over nodes of subelement
                            l2g[8*eN+i,j] = dofs[e2s[i][j]]
            lagrangeNodesArray = dofMap.lagrangeNodesArray
            if ar.global_sync:
                assert ar.writesDatasets, "not implemented  for text heavy data"
                n_elements       = mesh.globalMesh.nElements_global*nsubelements
                elements_dims    = [n_elements, l2g.shape[-1]]
                nodes_dims       = [dofMap.nDOF_all_processes, lagrangeNodesArray.shape[-1]]
                elements_dataset = 'elements'+spaceSuffix+str(tCount)
                nodes_dataset    = 'nodes'+spaceSuffix+str(tCount)
                elements_dtype   = dofMap.subdomain2global
            else:
                n_elements       = l2g.shape[0]
                elements_dims    = list(l2g.shape)
                nodes_dims       = list(lagrangeNodesArray.shape)
                elements_dataset = 'elements'+str(ar.rank)+spaceSuffix+str(tCount)
                nodes_dataset    = 'nodes'+str(ar.rank)+spaceSuffix+str(tCount)
                elements_dtype   = l2g
            elements_stem = 'elements'+spaceSuffix+str(tCount)
            nodes_stem    = 'nodes'+spaceSuffix+str(tCount)

            self.arGrid = ar.write_grid(self.arGridCollection, gridName, t, tCount)
            ar.write_topology(self.arGrid, Xdmf_ElementTopology, n_elements,
                              elements_dims, elements_dataset, elements_stem,
                              dtype=elements_dtype)
            ar.write_geometry(self.arGrid, nodes_dims, nodes_dataset, nodes_stem,
                              dtype=lagrangeNodesArray)
            if init or meshChanged:
                if ar.global_sync:
                    ar.create_dataset_sync(elements_dataset,
                                           offsets = [x*nsubelements for x in mesh.globalMesh.elementOffsets_subdomain_owned],
                                           data = dofMap.subdomain2global[l2g[:mesh.nElements_owned*nsubelements]])
                    ar.create_dataset_sync(nodes_dataset,
                                           offsets = dofMap.dof_offsets_subdomain_owned,
                                           data = lagrangeNodesArray[:dofMap.dof_offsets_subdomain_owned[ar.rank+1]-dofMap.dof_offsets_subdomain_owned[ar.rank]])
                elif ar.writesDatasets:
                    ar.create_dataset_async(elements_dataset, data = l2g)
                    ar.create_dataset_async(nodes_dataset, data = lagrangeNodesArray)
                else:
                    numpy.savetxt(ar.textDataDir+"/"+elements_stem+".txt",l2g,fmt='%d')
                    numpy.savetxt(ar.textDataDir+"/"+nodes_stem+".txt",lagrangeNodesArray)
        return self.arGrid

    def _writeDiscontinuousVertexMesh(self,ar,gridName,mesh,spaceDim,t,tCount,
                                      spaceSuffix,init,meshChanged,
                                      nodes_every_pass=False):
        """The element-by-element vertex mesh several spaces display on.

        Each element gets its own copy of its vertices, numbered
        consecutively, so a field with one value per element vertex (a
        non-conforming P1 function written as discontinuous P1, an RT0
        velocity) can be shown. Shared by the Crouzeix-Raviart and
        lowest-order mixed writers, which differed only in dataset naming
        and in whether the nodes were rewritten on every pass.
        """
        if spaceDim == 1:
            Xdmf_ElementTopology = "Polyline"
        elif spaceDim == 2:
            Xdmf_ElementTopology = "Triangle"
        elif spaceDim == 3:
            Xdmf_ElementTopology = "Tetrahedron"
        Xdmf_NodesPerElement = spaceDim+1
        if ar.global_sync:
            assert ar.writesDatasets, "global_sync not implemented for text heavy data"
            Xdmf_NumberOfElements= mesh.globalMesh.nElements_global
            elements_dataset = 'elements'+spaceSuffix+str(tCount)
            nodes_dataset    = 'nodes'+spaceSuffix+str(tCount)
        else:
            Xdmf_NumberOfElements= mesh.nElements_global
            elements_dataset = 'elements'+str(ar.rank)+spaceSuffix+str(tCount)
            nodes_dataset    = 'nodes'+str(ar.rank)+spaceSuffix+str(tCount)
        elements_stem = 'elements'+spaceSuffix+str(tCount)
        nodes_stem    = 'nodes'+spaceSuffix+str(tCount)

        self.arGrid = ar.write_grid(self.arGridCollection, gridName, t, tCount)
        ar.write_topology(self.arGrid, Xdmf_ElementTopology, Xdmf_NumberOfElements,
                          [Xdmf_NumberOfElements, Xdmf_NodesPerElement],
                          elements_dataset, elements_stem,
                          dtype=GENERATED_INDEX_DTYPE)
        ar.write_geometry(self.arGrid, [Xdmf_NumberOfElements*Xdmf_NodesPerElement, 3],
                          nodes_dataset, nodes_stem, dtype=mesh.nodeArray)
        if ar.global_sync:
            if init or meshChanged:
                #simple dg l2g mapping
                dg_l2g = numpy.arange(mesh.nElements_owned*Xdmf_NodesPerElement,dtype=GENERATED_INDEX_DTYPE).reshape((mesh.nElements_owned,Xdmf_NodesPerElement))
                ar.create_dataset_sync(elements_dataset,
                                       offsets = mesh.globalMesh.elementOffsets_subdomain_owned,
                                       data = dg_l2g)
            if init or meshChanged or nodes_every_pass:
                dgnodes = numpy.reshape(mesh.nodeArray[mesh.elementNodesArray[:mesh.nElements_owned]],
                                        (mesh.nElements_owned*Xdmf_NodesPerElement,3))
                ar.create_dataset_sync(nodes_dataset,
                                       offsets = mesh.globalMesh.elementOffsets_subdomain_owned*Xdmf_NodesPerElement,
                                       data = dgnodes)
        elif ar.writesDatasets:
            if init or meshChanged:
                dg_l2g = numpy.arange(Xdmf_NumberOfElements*Xdmf_NodesPerElement,dtype=GENERATED_INDEX_DTYPE).reshape((Xdmf_NumberOfElements,Xdmf_NodesPerElement))
                ar.create_dataset_async(elements_dataset, data = dg_l2g)
            if init or meshChanged or nodes_every_pass:
                dgnodes = numpy.reshape(mesh.nodeArray[mesh.elementNodesArray],(Xdmf_NumberOfElements*Xdmf_NodesPerElement,3))
                ar.create_dataset_async(nodes_dataset, data = dgnodes)
        elif init or meshChanged:
            dg_l2g = numpy.arange(Xdmf_NumberOfElements*Xdmf_NodesPerElement,dtype=GENERATED_INDEX_DTYPE).reshape((Xdmf_NumberOfElements,Xdmf_NodesPerElement))
            numpy.savetxt(ar.textDataDir+"/"+elements_stem+".txt",dg_l2g,fmt='%d')
            dgnodes = numpy.reshape(mesh.nodeArray[mesh.elementNodesArray],(Xdmf_NumberOfElements*Xdmf_NodesPerElement,3))
            numpy.savetxt(ar.textDataDir+"/"+nodes_stem+".txt",dgnodes)
        return self.arGrid

    def writeMeshXdmf_CrouzeixRaviartP1(self,ar,mesh,spaceDim,dofMap,t=0.0,
                                        init=False,meshChanged=False,arGrid=None,tCount=0):
        """
        Write out nonconforming P1 approximation
        Write out as a (discontinuous) Lagrange P1 function to make visualization easier
        and dof's as face centered data on original grid
        """
        #write out basic geometry if not already done?
        mesh.writeMeshXdmf(ar,"Spatial_Domain",t,init,meshChanged,tCount=tCount)
        #now try to write out a mesh that matches DG P1 layout

        spaceSuffix = "_ncp1_CrouzeixRaviart"
        gridName = self.setGridCollectionAndGridElements(init,ar,arGrid,t,spaceSuffix)

        if self._needsGrid(t):
            self._writeDiscontinuousVertexMesh(ar,gridName,mesh,spaceDim,t,tCount,
                                               spaceSuffix,init,meshChanged)
        return self.arGrid
    #def
    def writeFunctionXdmf_DGP1Lagrange(self,ar,u,tCount=0,init=True, dofMap=None):
        if ar.global_sync:
            assert(dofMap)
            owned = dofMap.dof_offsets_subdomain_owned
            ar.write_field(self.arGrid, u.name, u.dof, tCount,
                           dimensions=[dofMap.nDOF_all_processes],
                           sync_offsets=owned,
                           sync_data=u.dof[:owned[ar.rank+1] - owned[ar.rank]])
        else:
            #Dimensions comes from the array rather than u.nDOF_global: the
            #DataItem describes exactly what is written, and the two differ
            #when the caller passed a foreign array through the residual
            #adapter (phi_s is a vertex field, not a DOF vector of this space)
            ar.write_field(self.arGrid, u.name, u.dof, tCount)

    def writeFunctionXdmf_DGP2Lagrange(self,ar,u,tCount=0,init=True, dofMap=None):
        #this writer predates the <name>_p<rank>_t<tCount> dataset
        #convention and names its dataset <name><tCount> in both the
        #synchronized and per-rank cases. Passed explicitly so converting
        #it does not rename datasets inside existing archives.
        dataset = u.name + str(tCount)
        if ar.global_sync:
            owned = dofMap.dof_offsets_subdomain_owned
            ar.write_field(self.arGrid, u.name, u.dof, tCount,
                           dataset=dataset,
                           dimensions=[dofMap.nDOF_all_processes],
                           sync_offsets=owned,
                           sync_data=u.dof[:owned[ar.rank+1] - owned[ar.rank]])
        else:
            #Dimensions from the array, not u.nDOF_global -- see
            #writeFunctionXdmf_DGP1Lagrange
            ar.write_field(self.arGrid, u.name, u.dof, tCount,
                           dataset=dataset)

    def writeFunctionXdmf_CrouzeixRaviartP1(self,ar,u,tCount=0,init=True, dofMap=None):
        """A non-conforming P1 function, two ways.

        As a discontinuous P1 function on the element-by-element vertex
        mesh (the vertex value is the sum over the element's faces minus
        the opposite face's), which viewers can draw; and as the dofs
        themselves, face-centred on the base mesh.
        """
        mesh = u.femSpace.mesh
        Xdmf_NodesPerElement  = mesh.nNodes_element
        name = u.name.replace(' ','_')
        if ar.global_sync:
            Xdmf_NumberOfElements = mesh.globalMesh.nElements_global
            nElements_owned = (mesh.globalMesh.elementOffsets_subdomain_owned[ar.rank+1]
                               - mesh.globalMesh.elementOffsets_subdomain_owned[ar.rank])
        else:
            Xdmf_NumberOfElements = mesh.nElements_global
            nElements_owned = Xdmf_NumberOfElements
        u_tmp = numpy.zeros((nElements_owned*Xdmf_NodesPerElement,),u.dof.dtype)
        l2g = u.femSpace.dofMap.l2g
        if u.femSpace.nSpace_global == 1:
            for eN in range(nElements_owned):
                #dof associated with face id, so opposite usual C0P1 ordering here
                u_tmp[eN*Xdmf_NodesPerElement + 0] = u.dof[l2g[eN,1]]
                u_tmp[eN*Xdmf_NodesPerElement + 1] = u.dof[l2g[eN,0]]
        elif u.femSpace.nSpace_global == 2:
            for eN in range(nElements_owned):
                #assume vertex associated with face across from it
                u_tmp[eN*Xdmf_NodesPerElement + 0] = u.dof[l2g[eN,1]]
                u_tmp[eN*Xdmf_NodesPerElement + 0]+= u.dof[l2g[eN,2]]
                u_tmp[eN*Xdmf_NodesPerElement + 0]-= u.dof[l2g[eN,0]]

                u_tmp[eN*Xdmf_NodesPerElement + 1] = u.dof[l2g[eN,0]]
                u_tmp[eN*Xdmf_NodesPerElement + 1]+= u.dof[l2g[eN,2]]
                u_tmp[eN*Xdmf_NodesPerElement + 1]-= u.dof[l2g[eN,1]]

                u_tmp[eN*Xdmf_NodesPerElement + 2] = u.dof[l2g[eN,0]]
                u_tmp[eN*Xdmf_NodesPerElement + 2]+= u.dof[l2g[eN,1]]
                u_tmp[eN*Xdmf_NodesPerElement + 2]-= u.dof[l2g[eN,2]]
        else:
            for eN in range(nElements_owned):
                for i in range(Xdmf_NodesPerElement):
                    u_tmp[eN*Xdmf_NodesPerElement + i] = u.dof[l2g[eN,i]]*(1.0-float(u.femSpace.nSpace_global)) + \
                                                         sum([u.dof[l2g[eN,j]] for j in range(Xdmf_NodesPerElement) if j != i])
        #as dgp1, on this writer's grid
        if ar.global_sync:
            ar.write_field(self.arGrid, name, u_tmp, tCount,
                           dimensions=[Xdmf_NumberOfElements*Xdmf_NodesPerElement],
                           sync_offsets=mesh.globalMesh.elementOffsets_subdomain_owned*Xdmf_NodesPerElement,
                           sync_data=u_tmp)
        else:
            ar.write_field(self.arGrid, name, u_tmp, tCount)
        #as face centered dofs (true nc p1), on the mesh's grid
        grid = mesh.arGrid
        if ar.global_sync:
            owned = dofMap.dof_offsets_subdomain_owned
            ar.write_field(grid, name+"_dof", u.dof, tCount, center="Face",
                           dimensions=[dofMap.nDOF_all_processes],
                           sync_offsets=owned,
                           sync_data=u.dof[:owned[ar.rank+1] - owned[ar.rank]])
        else:
            ar.write_field(grid, name+"_dof", u.dof, tCount, center="Face",
                           dimensions=[u.nDOF_global])

    def _nodalVector(self,uList,components,n_rows=None):
        """A 3-component array from up to three dof vectors; missing components are zero."""
        u_dof = uList[components[0]].dof
        rows = slice(None) if n_rows is None else slice(0, n_rows)
        u_dof = u_dof[rows]
        if len(components) < 2:
            v_dof = numpy.zeros(u_dof.shape,dtype='d')
        else:
            v_dof = uList[components[1]].dof[rows]
        if len(components) < 3:
            w_dof = numpy.zeros(u_dof.shape,dtype='d')
        else:
            w_dof = uList[components[2]].dof[rows]
        return numpy.column_stack((u_dof,v_dof,w_dof))

    def writeVectorFunctionXdmf_nodal(self,ar,uList,components,vectorName,spaceSuffix,tCount=0,init=True):
        nDOF_global = uList[components[0]].nDOF_global
        velocity = self._nodalVector(uList,components)
        if ar.global_sync:
            dofMap = uList[components[0]].femSpace.dofMap
            nDOF_owned = (dofMap.dof_offsets_subdomain_owned[ar.rank+1] -
                          dofMap.dof_offsets_subdomain_owned[ar.rank] )
            ar.write_field(self.arGrid, vectorName, velocity, tCount, rank="Vector",
                           dimensions=[dofMap.nDOF_all_processes,3],
                           sync_offsets=dofMap.dof_offsets_subdomain_owned,
                           sync_data=velocity[:nDOF_owned])
        else:
            ar.write_field(self.arGrid, vectorName, velocity, tCount, rank="Vector",
                           dimensions=[nDOF_global,3])

    def writeVectorFunctionXdmf_CrouzeixRaviartP1(self,ar,uList,components,spaceSuffix,tCount=0,init=True):
        return self.writeVectorFunctionXdmf_nodal(ar,uList,components,"_ncp1_CrouzeixRaviart",
                                                  tCount=tCount,init=init)

    def writeMeshXdmf_MonomialDGPK(self,ar,mesh,spaceDim,interpolationPoints,t=0.0,
                                   init=False,meshChanged=False,arGrid=None,tCount=0):
        """
        as a first cut, archive DG monomial spaces using same approach as for element quadrature
        arrays using x = interpolation points (which are just Gaussian quadrature points) as values
        """

        #write out basic geometry if not already done?
        mesh.writeMeshXdmf(ar,"Spatial_Domain",t,init,meshChanged,tCount=tCount)
        #now try to write out a mesh that is a collection of points per element
        spaceSuffix = "_dgpk_Monomial"

        gridName = self.setGridCollectionAndGridElements(init,ar,arGrid,t,spaceSuffix)

        assert len(interpolationPoints.shape) == 3 #make sure have right type of dictionary
        self.mesh = mesh
        if self._needsGrid(t):
            Xdmf_NodesPerElement = interpolationPoints.shape[1]
            if ar.global_sync:
                assert ar.writesDatasets, "global_sync not implementedf or text heavy data"
                Xdmf_NumberOfElements= mesh.globalMesh.nElements_global
                elements_dataset = 'elements'+spaceSuffix+str(tCount)
                nodes_dataset    = 'nodes'+spaceSuffix+str(tCount)
            else:
                Xdmf_NumberOfElements= interpolationPoints.shape[0]
                elements_dataset = 'elements'+str(ar.rank)+spaceSuffix+str(tCount)
                nodes_dataset    = 'nodes'+str(ar.rank)+spaceSuffix+str(tCount)
            Xdmf_NodesGlobal     = Xdmf_NumberOfElements*Xdmf_NodesPerElement
            elements_stem = 'elements'+spaceSuffix+str(tCount)
            nodes_stem    = 'nodes'+spaceSuffix+str(tCount)

            self.arGrid = ar.write_grid(self.arGridCollection, gridName, t, tCount)
            ar.write_topology(self.arGrid, "Polyvertex", Xdmf_NumberOfElements,
                              [Xdmf_NumberOfElements, Xdmf_NodesPerElement],
                              elements_dataset, elements_stem,
                              nodes_per_element=Xdmf_NodesPerElement,
                              dtype=GENERATED_INDEX_DTYPE)
            ar.write_geometry(self.arGrid, [Xdmf_NodesGlobal, 3],
                              nodes_dataset, nodes_stem, dtype=interpolationPoints)
            if init or meshChanged:
                if ar.global_sync:
                    q_l2g = numpy.arange(mesh.nElements_owned*Xdmf_NodesPerElement,dtype=GENERATED_INDEX_DTYPE).reshape((mesh.nElements_owned,Xdmf_NodesPerElement))
                    ar.create_dataset_sync(elements_dataset,
                                           offsets = mesh.globalMesh.elementOffsets_subdomain_owned,
                                           data = q_l2g)
                    ar.create_dataset_sync(nodes_dataset,
                                           offsets = mesh.globalMesh.elementOffsets_subdomain_owned,
                                           data = interpolationPoints[:mesh.nElements_owned])
                else:
                    q_l2g = numpy.arange(Xdmf_NumberOfElements*Xdmf_NodesPerElement,dtype=GENERATED_INDEX_DTYPE).reshape((Xdmf_NumberOfElements,Xdmf_NodesPerElement))
                    if ar.writesDatasets:
                        ar.create_dataset_async(elements_dataset, data = q_l2g)
                        ar.create_dataset_async(nodes_dataset, data = interpolationPoints.flat[:])
                    else:
                        numpy.savetxt(ar.textDataDir+"/"+elements_stem+".txt",q_l2g,fmt='%d')
                        numpy.savetxt(ar.textDataDir+"/"+nodes_stem+".txt",interpolationPoints.flat[:])
        return self.arGrid
    #def
    def writeFunctionXdmf_MonomialDGPK(self,ar,interpolationValues,name,tCount=0,init=True, mesh=None):
        """
        Different than usual FemFunction Write routines since saves values at interpolation points
        need to check on way to save dofs as well
        """
        assert len(interpolationValues.shape) == 2
        if ar.global_sync:
            ar.write_field(self.arGrid, name, interpolationValues, tCount,
                           dimensions=[mesh.globalMesh.nElements_global
                                       *interpolationValues.shape[1]],
                           sync_offsets=mesh.globalMesh.elementOffsets_subdomain_owned,
                           sync_data=interpolationValues[:mesh.nElements_owned])
        else:
            ar.write_field(self.arGrid, name, interpolationValues.flat[:], tCount,
                           dimensions=[interpolationValues.shape[0]
                                       *interpolationValues.shape[1]])

    def writeVectorFunctionXdmf_MonomialDGPK(self,ar,interpolationValues,name,tCount=0,init=True):
        """
        Different than usual FemFunction Write routines since saves values at interpolation points
        need to check on way to save dofs as well
        """
        assert len(interpolationValues.shape) == 3
        Xdmf_NumberOfComponents = interpolationValues.shape[2]
        n_points = interpolationValues.shape[1]
        if ar.global_sync:
            #this rank contributes the elements it owns; the DataItem covers
            #every element's points globally (the previous version sized the
            #temporary by the global count and filled it from the local
            #array, which only agrees on one rank)
            n_owned = self.mesh.nElements_owned
            tmp = numpy.zeros((n_owned*n_points,3),interpolationValues.dtype)
            tmp[:,:Xdmf_NumberOfComponents]=numpy.reshape(interpolationValues[:n_owned].flat,
                                                          (n_owned*n_points,Xdmf_NumberOfComponents))
            ar.write_field(self.arGrid, name, tmp, tCount, rank="Vector",
                           dimensions=[self.mesh.globalMesh.nElements_global*n_points,3],
                           sync_offsets=self.mesh.globalMesh.elementOffsets_subdomain_owned*n_points,
                           sync_data=tmp)
        else:
            Xdmf_NodesGlobal = interpolationValues.shape[0]*n_points
            tmp = numpy.zeros((Xdmf_NodesGlobal,3),interpolationValues.dtype)
            tmp[:,:Xdmf_NumberOfComponents]=numpy.reshape(interpolationValues.flat,(Xdmf_NodesGlobal,Xdmf_NumberOfComponents))
            ar.write_field(self.arGrid, name, tmp, tCount, rank="Vector",
                           dimensions=[Xdmf_NodesGlobal,3])


    def writeMeshXdmf_DGP0(self,ar,mesh,spaceDim,
                           t=0.0,init=False,meshChanged=False,arGrid=None,tCount=0):
        """
        as a first cut, archive piecewise constant space
        """

        #write out basic geometry if not already done?
        meshSpaceSuffix = "Spatial_Domain"
        mesh.writeMeshXdmf(ar,meshSpaceSuffix,t,init,meshChanged,tCount=tCount)
        #now try to write out a mesh that a constant per element
        spaceSuffix = "_dgp0"
        gridName = self.setGridCollectionAndGridElements(init,ar,arGrid,t,spaceSuffix)

        if self._needsGrid(t):
            #mwf hack
            #allow for other types of topologies if the mesh has specified one
            if 'elementTopologyName' in dir(mesh):
                Xdmf_ElementTopology = mesh.elementTopologyName
            else:
                if spaceDim == 1:
                    Xdmf_ElementTopology = "Polyline"
                elif spaceDim == 2:
                    Xdmf_ElementTopology = "Triangle"
                elif spaceDim == 3:
                    Xdmf_ElementTopology = "Tetrahedron"
            #the spatial mesh's own datasets are reused; nothing is written here
            if ar.global_sync:
                assert ar.writesDatasets, "global_sync not implemented  for text heavy data"
                elements_dataset = 'elements'+meshSpaceSuffix+str(tCount)
                nodes_dataset    = 'nodes'+meshSpaceSuffix+str(tCount)
                elements_dtype   = mesh.globalMesh.nodeNumbering_subdomain2global
            else:
                elements_dataset = 'elements'+str(ar.rank)+meshSpaceSuffix+str(tCount)
                nodes_dataset    = 'nodes'+str(ar.rank)+meshSpaceSuffix+str(tCount)
                elements_dtype   = mesh.elementNodesArray
            #mwf hack, allow for a mixed element mesh
            if mesh.nNodes_element is None:
                assert 'xdmf_topology' in dir(mesh)
                elements_dims = [len(mesh.xdmf_topology)]
            else:
                elements_dims = [mesh.nElements_global, mesh.nNodes_element]
            self.arGrid = ar.write_grid(self.arGridCollection, gridName, t, tCount)
            ar.write_topology(self.arGrid, Xdmf_ElementTopology, mesh.nElements_global,
                              elements_dims, elements_dataset,
                              'elements'+meshSpaceSuffix, dtype=elements_dtype)
            ar.write_geometry(self.arGrid, [mesh.nNodes_global, 3],
                              nodes_dataset, 'nodes'+meshSpaceSuffix,
                              dtype=mesh.nodeArray)
        return self.arGrid
    #def
    def writeFunctionXdmf_DGP0(self,ar,u,tCount=0,init=True):
        name = u.name.replace(' ','_')
        mesh = u.femSpace.elementMaps.mesh
        if ar.global_sync:
            ar.write_field(self.arGrid, name, u.dof, tCount, center="Cell",
                           dimensions=[mesh.globalMesh.nElements_global],
                           sync_offsets=mesh.globalMesh.elementOffsets_subdomain_owned,
                           sync_data=u.dof[:mesh.nElements_owned])
        else:
            ar.write_field(self.arGrid, name, u.dof, tCount, center="Cell",
                           dimensions=[mesh.nElements_global])

    def writeVectorFunctionXdmf_DGP0(self,ar,uList,components,vectorName,tCount=0,init=True):
        #this referred to a bare `u` that is not a parameter of this
        #function, so every call raised NameError -- including in the
        #default global_sync configuration. The mesh comes from the first
        #component, matching u_dof below.
        mesh = uList[components[0]].femSpace.elementMaps.mesh
        velocity = self._nodalVector(uList,components)

        if ar.global_sync:
            ar.write_field(self.arGrid, vectorName, velocity, tCount,
                           center="Cell", rank="Vector",
                           dimensions=[mesh.globalMesh.nElements_global,3],
                           sync_offsets=mesh.globalMesh.elementOffsets_subdomain_owned,
                           sync_data=velocity[:mesh.nElements_owned])
        else:
            #the non-HDF path had no else branch at all, so with
            #--useTextArchive this Attribute got a DataItem containing no
            #data reference. write_field writes a sidecar instead.
            ar.write_field(self.arGrid, vectorName, velocity, tCount,
                           center="Cell", rank="Vector",
                           dimensions=[mesh.nElements_global,3])

    def writeMeshXdmf_P1Bubble(self,ar,mesh,spaceDim,dofMap,t=0.0,
                               init=False,meshChanged=False,arGrid=None,tCount=0):
        """
        represent P1 bubble space using just vertices for now
        """
        mesh.writeMeshXdmf(ar,"Spatial_Domain",t,init,meshChanged,tCount=tCount)
        spaceSuffix = "_c0p1_Bubble%s" % tCount
        gridName = self.setGridCollectionAndGridElements(init,ar,arGrid,t,spaceSuffix)

        if self._needsGrid(t):
            if spaceDim == 1:
                Xdmf_ElementTopology = "Polyline"
            elif spaceDim == 2:
                Xdmf_ElementTopology = "Triangle"
            elif spaceDim == 3:
                Xdmf_ElementTopology = "Tetrahedron"
            Xdmf_NodesPerElement = spaceDim+1 #just handle vertices for now
            Xdmf_NumberOfNodes   = mesh.nNodes_global
            if  ar.global_sync:
                assert ar.writesDatasets, "global_sync not implemented for text heavy data"
                Xdmf_NumberOfElements= mesh.globalMesh.nElements_global
                elements_dataset = 'elements'+spaceSuffix+str(tCount)
                nodes_dataset    = 'nodes'+spaceSuffix+str(tCount)
            else:
                Xdmf_NumberOfElements= mesh.nElements_global
                elements_dataset = 'elements'+str(ar.rank)+spaceSuffix+str(tCount)
                nodes_dataset    = 'nodes'+str(ar.rank)+spaceSuffix+str(tCount)
            elements_stem = 'elements'+spaceSuffix+str(tCount)
            nodes_stem    = 'nodes'+spaceSuffix+str(tCount)

            self.arGrid = ar.write_grid(self.arGridCollection, gridName, t, tCount)
            ar.write_topology(self.arGrid, Xdmf_ElementTopology, Xdmf_NumberOfElements,
                              [Xdmf_NumberOfElements, Xdmf_NodesPerElement],
                              elements_dataset, elements_stem,
                              dtype=mesh.elementNodesArray)
            ar.write_geometry(self.arGrid, [Xdmf_NumberOfNodes, 3],
                              nodes_dataset, nodes_stem, dtype=mesh.nodeArray)
            if init or meshChanged:
                if ar.global_sync:
                    #c0p1 mapping for now
                    ar.create_dataset_sync(elements_dataset,
                                           offsets = mesh.globalMesh.elementOffsets_subdomain_owned,
                                           data = mesh.elementNodesArray)
                    ar.create_dataset_sync(nodes_dataset,
                                           offsets = mesh.globalMesh.nodeOffsets_subdomain_owned,
                                           data = mesh.nodeArray)
                elif ar.writesDatasets:
                    ar.create_dataset_async(elements_dataset, data = mesh.elementNodesArray)
                    ar.create_dataset_async(nodes_dataset, data = mesh.nodeArray)
                else:
                    numpy.savetxt(ar.textDataDir+"/"+elements_stem+".txt",mesh.elementNodesArray,fmt='%d')
                    numpy.savetxt(ar.textDataDir+"/"+nodes_stem+".txt",mesh.nodeArray)
        return self.arGrid
    #def
    def writeFunctionXdmf_P1Bubble(self,ar,u,tCount=0,init=True):
        #this read ar.sync_global, which does not exist -- the attribute
        #is global_sync -- so every call raised AttributeError
        name = u.name.replace(' ','_')
        mesh = u.femSpace.mesh
        #just write out nodal part right now
        if ar.global_sync:
            ar.write_field(self.arGrid, name, u.dof, tCount,
                           dimensions=[mesh.globalMesh.nNodes_global],
                           sync_offsets=mesh.globalMesh.nodeOffsets_subdomain_owned,
                           sync_data=u.dof[0:mesh.nNodes_owned])
        else:
            ar.write_field(self.arGrid, name, u.dof[0:mesh.nNodes_global], tCount,
                           dimensions=[mesh.nNodes_global])

    def writeFunctionXdmf_C0P2Lagrange(self,ar,u,tCount=0,init=True):
        #the text branch here read `assert Fasle` -- a typo for False, so it
        #raised NameError rather than the intended AssertionError.
        #write_field raises the assertion itself.
        if ar.global_sync:
            owned = u.femSpace.dofMap.dof_offsets_subdomain_owned
            ar.write_field(self.arGrid, u.name, u.dof, tCount,
                           dimensions=[u.femSpace.dofMap.nDOF_all_processes],
                           sync_offsets=owned,
                           sync_data=u.dof[:(owned[ar.rank+1] - owned[ar.rank])])
        else:
            #Dimensions comes from the array rather than u.nDOF_global: the
            #DataItem describes exactly what is written, and the two differ
            #when the caller passed a foreign array through the residual
            #adapter (phi_s is a vertex field, not a DOF vector of this space)
            ar.write_field(self.arGrid, u.name, u.dof, tCount)

    def writeVectorFunctionXdmf_P1Bubble(self,ar,uList,components,vectorName,spaceSuffix,tCount=0,init=True):
        mesh = uList[components[0]].femSpace.mesh
        if ar.global_sync:
            nDOF_global = mesh.globalMesh.nNodes_global
            nDOF_local = (mesh.globalMesh.nodeOffsets_subdomain_owned[ar.rank+1] -
                          mesh.globalMesh.nodeOffsets_subdomain_owned[ar.rank])
            velocity = self._nodalVector(uList,components,n_rows=nDOF_local)
            ar.write_field(self.arGrid, vectorName, velocity, tCount, rank="Vector",
                           dimensions=[nDOF_global,3],
                           sync_offsets=mesh.globalMesh.nodeOffsets_subdomain_owned,
                           sync_data=velocity)
        else:
            nDOF_global = mesh.nNodes_global
            velocity = self._nodalVector(uList,components,n_rows=nDOF_global)
            ar.write_field(self.arGrid, vectorName, velocity, tCount, rank="Vector",
                           dimensions=[nDOF_global,3])

    def writeMeshXdmf_particles(self,ar,mesh,spaceDim,x,t=0.0,
                                init=False,meshChanged=False,arGrid=None,tCount=0,
                                spaceSuffix = "_particles"):
        """
        write out arbitrary set of points on a mesh
        """
        #write out basic geometry if not already done?
        mesh.writeMeshXdmf(ar,"Spatial_Domain",t,init,meshChanged,tCount=tCount)
        #now try to write out a mesh that is a collection of points per element

        gridName = self.setGridCollectionAndGridElements(init,ar,arGrid,t,spaceSuffix)
        nPoints = numpy.cumprod(x.shape)[-2]
        if self._needsGrid(t):
            Xdmf_NodesPerElement = 1
            #the dataset names carry the rank before the space suffix while the
            #sidecar names omit it; both predate write_field's convention
            elements_dataset = 'elements'+str(ar.rank)+spaceSuffix+str(tCount)
            nodes_dataset    = 'nodes'+str(ar.rank)+spaceSuffix+str(tCount)
            elements_stem    = 'elements'+spaceSuffix+str(tCount)
            nodes_stem       = 'nodes'+spaceSuffix+str(tCount)

            self.arGrid = ar.write_grid(self.arGridCollection, gridName, t, tCount)
            ar.write_topology(self.arGrid, "Polyvertex", nPoints,
                              [nPoints, Xdmf_NodesPerElement],
                              elements_dataset, elements_stem,
                              nodes_per_element=Xdmf_NodesPerElement,
                              dtype=GENERATED_INDEX_DTYPE)
            ar.write_geometry(self.arGrid, [nPoints, 3],
                              nodes_dataset, nodes_stem, dtype=x)

            #the references above are written every pass; the arrays only when
            #the mesh actually changed
            if init or meshChanged:
                q_l2g = numpy.arange(nPoints*Xdmf_NodesPerElement,
                                     dtype=GENERATED_INDEX_DTYPE).reshape((nPoints,Xdmf_NodesPerElement))
                if ar.writesDatasets:
                    ar.create_dataset_async(elements_dataset, data = q_l2g)
                    ar.create_dataset_async(nodes_dataset, data = x.flat[:])
                else:
                    numpy.savetxt(ar.textDataDir+"/"+elements_stem+".txt",q_l2g,fmt='%d')
                    numpy.savetxt(ar.textDataDir+"/"+nodes_stem+".txt",x.flat[:])
        return self.arGrid
    #def
    def writeScalarXdmf_particles(self,ar,u,name,tCount=0,init=True):
        nPoints = numpy.cumprod(u.shape)[-1]
        ar.write_field(self.arGrid, name, u.flat[:], tCount,
                       dimensions=[nPoints])

    def writeVectorXdmf_particles(self,ar,u,name,tCount=0,init=True):
        nPoints = numpy.cumprod(u.shape)[-2]
        Xdmf_NumberOfComponents = u.shape[-1]
        #force a 3-component vector since the points are 3D
        tmp = numpy.zeros((nPoints,3),'d')
        tmp[:,:Xdmf_NumberOfComponents] = numpy.reshape(
            u.flat,(nPoints,Xdmf_NumberOfComponents))
        ar.write_field(self.arGrid, name, tmp, tCount, rank="Vector",
                       dimensions=[nPoints,3])

    def writeMeshXdmf_LowestOrderMixed(self,ar,mesh,spaceDim,t=0.0,init=False,meshChanged=False,arGrid=None,tCount=0,
                                       spaceSuffix = "_RT0"):
        #write out basic geometry if not already done?
        mesh.writeMeshXdmf(ar,"Spatial_Domain",t,init,meshChanged,tCount=tCount)
        #now try to write out a mesh that matches RT0 velocity as dgp1 lagrange
        gridName = self.setGridCollectionAndGridElements(init,ar,arGrid,t,spaceSuffix)
        self.mesh = mesh
        if self._needsGrid(t):
            #this writer rewrote the nodes on every pass of the HDF5 path
            self._writeDiscontinuousVertexMesh(ar,gridName,mesh,spaceDim,t,tCount,
                                               spaceSuffix,init,meshChanged,
                                               nodes_every_pass=True)
        return self.arGrid
    #def
    def writeVectorFunctionXdmf_LowestOrderMixed(self,ar,u,tCount=0,init=True,spaceSuffix="_RT0",name="velocity"):
        Xdmf_NumberOfComponents = u.shape[2]
        Xdmf_StorageDim = 3
        n_points = u.shape[1]
        if ar.global_sync:
            #this rank contributes the elements it owns; the previous version
            #wrote the whole local array under the global name, which only
            #agrees on one rank
            n_owned = self.mesh.nElements_owned
            tmp = numpy.zeros((n_owned*n_points,Xdmf_StorageDim),u.dtype)
            tmp[:,:Xdmf_NumberOfComponents]=numpy.reshape(u[:n_owned].flat,(n_owned*n_points,Xdmf_NumberOfComponents))
            ar.write_field(self.arGrid, name, tmp, tCount, rank="Vector",
                           dimensions=[self.mesh.globalMesh.nElements_global*n_points,Xdmf_StorageDim],
                           sync_offsets=self.mesh.globalMesh.elementOffsets_subdomain_owned*n_points,
                           sync_data=tmp)
        else:
            Xdmf_NodesGlobal = u.shape[0]*n_points
            tmp = numpy.zeros((Xdmf_NodesGlobal,Xdmf_StorageDim),u.dtype)
            tmp[:,:Xdmf_NumberOfComponents]=numpy.reshape(u.flat,(Xdmf_NodesGlobal,Xdmf_NumberOfComponents))
            ar.write_field(self.arGrid, name, tmp, tCount, rank="Vector",
                           dimensions=[Xdmf_NodesGlobal,Xdmf_StorageDim])
