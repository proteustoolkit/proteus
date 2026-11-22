"""Declaring extra fields to write into the archive.

The problem this solves
-----------------------
Adding an auxiliary field to the archive used to mean editing the generic
driver, ``NumericalSolution.py``, and the established way to do it was::

    try:
        phi_s = {}
        phi_s[0] = model.levelModelList[-1].coefficients.phi_s
        model.levelModelList[-1].archiveFiniteElementResiduals(
            self.ar[index], t, self.tCount, phi_s, res_name_base='phi_s')
        logEvent("Writing initial phi_s ...")
    except:
        pass

Eight fields were written that way, each duplicated across
``archiveInitialSolution`` and ``archiveSolution``. The pattern has four
problems:

1. ``archiveFiniteElementResiduals`` is not a residual writer. It is the
   only entry point that accepts an arbitrary DOF array, so everything
   got routed through it -- bathymetry, free-surface elevation, VOF, a
   solid-phase indicator. Its name misleads, and its ``{0: array}``
   argument is pure ceremony.
2. ``try/except: pass`` is being used as feature detection ("skip this
   field if the model doesn't have it"), but it discards *every* error in
   the block -- a shape mismatch, a failed HDF5 write, a renamed
   attribute.
3. That masking hid real bugs. ``archiveInitialSolution`` has no ``t``
   parameter, yet one of its blocks called ``str(t)``; the resulting
   ``NameError`` fired on every model that had ``coefficients.phi_s`` and
   was never seen. Worse, ``coefficients.phi_sp`` stopped existing when
   commit ``eed483b3`` replaced the nodal ``phi_sp`` field with
   quadrature-point ``q_phi_porous``. Both ``phi_sp`` blocks have been
   dead ever since -- archives written before that change contain a
   ``phi_sp0`` field and archives written after silently do not.
4. It put model-specific knowledge in the generic driver, down to
   ``if 'clsvof' in model.name:``.

How this module fixes it
------------------------
Fields are *declared where they live*, by the coefficients class that owns
them, and the driver writes them in one loop:

.. code-block:: python

    # in proteus/kernels/SW2DCV.py, class Coefficients
    def archiveFields(self, lm):
        yield ArchiveField("bathymetry", self.b.dof)
        yield ArchiveField("eta", self.b.dof + lm.u[0].dof)

.. code-block:: python

    # in NumericalSolution.py -- replaces all sixteen blocks
    model.levelModelList[-1].archiveDeclaredFields(self.ar[index], t, self.tCount)

Consequences:

- Absence is expressed as a normal ``if`` in the hook, on the class that
  knows the answer, so it no longer has to be spelled ``except: pass``.
  Everything else propagates, with the field's name in the message.
- ``TC_base.archiveFields`` yields nothing, so every existing coefficients
  class keeps working untouched.
- There is one code path, exercised by every field, so the next bug of the
  ``NameError`` kind fails on the first test run instead of hiding.
- ``center`` and ``rank`` are exactly the YMF/XDMF ``Center`` and
  ``AttributeType``, validated against closed sets, so a malformed field
  is not representable.
"""

from .Profiling import logEvent

#: Legal ``center`` values -- what mesh entity a field is attached to.
#: Mirrors ``ymf.archive.CENTERINGS``, restated here so this module does
#: not import ymf just to validate two strings.
CENTERINGS = frozenset({"Node", "Cell", "Face", "Edge", "Grid"})

#: Legal ``rank`` values -- becomes XDMF's ``AttributeType``.
RANKS = frozenset({"Scalar", "Vector", "Tensor"})


class ArchiveFieldError(ValueError):
    """A declared archive field is malformed, or writing one failed.

    Deliberately not caught anywhere in the archive path. The whole point
    of this module is that a broken field declaration is a loud failure
    naming the field, rather than a silently skipped write.
    """


class ArchiveField(object):
    """One field to write into the archive each time the archive is written.

    Parameters
    ----------
    name : str
        The field's name in the archive. May contain ``{model}``, which is
        expanded to the owning model's name -- needed when several models
        share one archive (``useOneArchive``) and would otherwise collide.
    value : numpy.ndarray
        The DOF array to write. Read at write time, so a declaration may
        compute it (``self.b.dof + lm.u[0].dof``) and stay current.
    center : str
        Which mesh entity the values live on; one of :data:`CENTERINGS`.
    rank : str
        ``Scalar``, ``Vector`` or ``Tensor``; one of :data:`RANKS`.
    femSpace : object, optional
        The finite element space the DOFs belong to. Defaults to the space
        of the model's component ``component``, which is what every field
        migrated from the old code wanted.
    component : int
        Which solution component's FEM space to default to.
    on_mesh_nodes : bool
        Write the field onto the base mesh's grid, one value per mesh
        vertex, instead of dispatching through a finite element space.
        Use it for fields defined on the mesh itself rather than on a
        solution space -- a solid-phase indicator sized
        ``mesh.nodeArray.shape[0]`` belongs on the mesh's nodes, not on a
        quadratic space's degrees of freedom.
    units, std_name : str, optional
        Carried into the archive as metadata for the YMF front-end and
        CSDMS interop. **Not validated here** -- per the campaign's trust
        boundary, semantic checking belongs to the front-end, off the
        compute path.
    """

    __slots__ = (
        "name",
        "value",
        "center",
        "rank",
        "femSpace",
        "component",
        "units",
        "std_name",
        "model_name",
        "on_mesh_nodes",
    )

    def __init__(
        self,
        name,
        value,
        center="Node",
        rank="Scalar",
        femSpace=None,
        component=0,
        units=None,
        std_name=None,
        on_mesh_nodes=False,
    ):
        if not name:
            raise ArchiveFieldError("an ArchiveField needs a non-empty name")
        if center not in CENTERINGS:
            raise ArchiveFieldError(
                "ArchiveField(%r): center=%r is not one of %s"
                % (name, center, sorted(CENTERINGS))
            )
        if rank not in RANKS:
            raise ArchiveFieldError(
                "ArchiveField(%r): rank=%r is not one of %s"
                % (name, rank, sorted(RANKS))
            )
        if value is None:
            raise ArchiveFieldError(
                "ArchiveField(%r): value is None. A field whose data may be "
                "absent should be guarded by an 'if' in the archiveFields "
                "hook rather than declared with no data." % (name,)
            )
        self.name = name
        self.value = value
        self.center = center
        self.rank = rank
        self.femSpace = femSpace
        self.component = component
        self.units = units
        self.std_name = std_name
        self.on_mesh_nodes = on_mesh_nodes
        self.model_name = None

    @property
    def archive_name(self):
        """The name to write, with ``{model}`` expanded."""
        if "{model}" not in self.name:
            return self.name
        if self.model_name is None:
            raise ArchiveFieldError(
                "ArchiveField(%r) uses {model} but no model name was supplied; "
                "it must be collected via archive_fields_for()" % (self.name,)
            )
        return self.name.format(model=self.model_name)

    def resolve_fem_space(self, lm):
        """Return the FEM space these DOFs belong to."""
        if self.femSpace is not None:
            return self.femSpace
        try:
            return lm.u[self.component].femSpace
        except (AttributeError, KeyError, IndexError) as exc:
            raise ArchiveFieldError(
                "ArchiveField(%r): no femSpace given and the model has no "
                "component %r to take one from" % (self.name, self.component)
            ) from exc

    def __repr__(self):
        return "ArchiveField(%r, center=%r, rank=%r)" % (
            self.name,
            self.center,
            self.rank,
        )


def archive_fields_for(lm, model_name=None):
    """Yield the :class:`ArchiveField` objects a level model declares.

    Parameters
    ----------
    lm : OneLevelTransport
        The level model whose coefficients are asked for declarations.
    model_name : str, optional
        The owning multilevel model's name, used to expand ``{model}`` in
        field names. Several models can share one archive
        (``useOneArchive``), so a field like ``quantDOFs`` needs the model
        name to stay unique.

    Errors are **not** swallowed: a hook that raises produces an
    :class:`ArchiveFieldError` naming it, because a field that was meant to
    be archived and silently wasn't is the failure this design exists to
    prevent.
    """
    hook = getattr(lm.coefficients, "archiveFields", None)
    if hook is None:
        # A coefficients class that predates TC_base gaining the hook.
        return
    try:
        declared = list(hook(lm))
    except Exception as exc:
        raise ArchiveFieldError(
            "%s.archiveFields() failed for model %r: %s"
            % (type(lm.coefficients).__name__, model_name or "?", exc)
        ) from exc

    seen = {}
    for field in declared:
        if not isinstance(field, ArchiveField):
            raise ArchiveFieldError(
                "%s.archiveFields() yielded %r, which is not an ArchiveField"
                % (type(lm.coefficients).__name__, field)
            )
        field.model_name = model_name
        name = field.archive_name
        if name in seen:
            raise ArchiveFieldError(
                "%s.archiveFields() declared %r twice; archive field names "
                "must be unique within a model, or one silently shadows the "
                "other" % (type(lm.coefficients).__name__, name)
            )
        seen[name] = field
        logEvent("Archiving declared field %s" % (name,), level=4)
        yield field
