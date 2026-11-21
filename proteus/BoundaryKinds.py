"""
How a boundary condition is imposed, per face: ``strong``, ``weak`` or ``none`` (C12).

Until now the choice was made per numerical-flux class (``NF_base.useWeakDirichletConditions``,
``useStrongDirichletConstraints``) and per model (``forceStrongConditions``): every Dirichlet face of a
problem was imposed the same way. A face can now say how it is imposed:

* ``strong`` -- the degrees of freedom on the face are set to the data (their rows are replaced by
  ``u = g``, or eliminated by a flux class that eliminates them). The numerical flux adds **no** weak term
  on the face: a strong face never also gets the Nitsche terms (that would impose the condition twice).
* ``weak`` -- the data enters the boundary flux (Nitsche, upwind inflow value); the degrees of freedom stay
  unknowns.
* ``none`` -- a declared open face: no value, no flux datum. The face carries what the flux class does for a
  face with no condition: the upwinded advective outflow, a natural (zero) diffusive flux. This is what
  ``fluxBoundaryConditions[ci] == 'outFlow'`` meant for every undeclared face of a component; ``none``
  says it for one face.

A boundary-condition setter ``getDirichletConditions(x, flag)`` returns ``None`` (no condition), or a function
``g(x, t)``. To give the face a kind, return the function through :func:`strong`, :func:`weak` or
:data:`OPEN`, and register the setter with :func:`declare`::

    from proteus import BoundaryKinds as bk

    def u_dirichlet(x, flag):
        if flag == left:
            return bk.strong(lambda x, t: 1.0)
        if flag == bottom:
            return bk.weak(lambda x, t: 0.0)
        if flag == outlet:
            return bk.OPEN

    dirichletConditions = {0: bk.declare(u_dirichlet)}

**Defaults, which keep every existing result.** A setter that is not declared behaves exactly as before: the
kind of each of its faces is the kind of the flux class (``weak`` under a weak class, ``strong`` under
``StrongDirichlet``). Inside a declared setter a bare ``g`` (no wrapper) gets the same default, so a Dirichlet
callable under a weak class stays weak. Only the faces that name a kind change.

Nothing here is read for an undeclared setter, and a model that does not implement per-face kinds refuses a declared
setter (``supportsBoundaryKinds``) instead of ignoring it.
"""

STRONG = 'strong'
WEAK = 'weak'
NONE = 'none'
KINDS = (STRONG, WEAK, NONE)


class _Imposed(object):
    """A boundary function ``g(x, t)`` together with how the face imposes it."""
    __slots__ = ('g', 'kind')

    def __init__(self, g, kind):
        if kind not in (STRONG, WEAK):
            raise ValueError("a boundary function is imposed 'strong' or 'weak', not %r" % (kind,))
        if not callable(g):
            raise TypeError("a boundary function must be callable: g(x, t); got %r" % (g,))
        self.g = g
        self.kind = kind

    def __call__(self, x, t):
        return self.g(x, t)

    def __repr__(self):
        return "BoundaryKinds.%s(%r)" % (self.kind, self.g)


class _Open(object):
    """The declared open face (kind ``none``). It has no value: calling it is an error, so a consumer that
    forgot to read the kind fails loudly instead of imposing 0."""
    kind = NONE
    __slots__ = ()

    def __call__(self, x, t):
        raise TypeError("a face declared 'none' (BoundaryKinds.OPEN) has no boundary value; a consumer of "
                        "boundary conditions must read BoundaryKinds.kind_of before calling the function")

    def __repr__(self):
        return "BoundaryKinds.OPEN"


OPEN = _Open()


def strong(g):
    """``g`` imposed strongly on the faces that return it."""
    return _Imposed(g, STRONG)


def weak(g):
    """``g`` imposed weakly (through the numerical flux) on the faces that return it."""
    return _Imposed(g, WEAK)


def kind_of(g, default):
    """The kind of what a setter returned: ``strong``/``weak``/``none``, or ``default`` for a bare function.
    ``g`` is not ``None`` (a setter that returned ``None`` has no condition on the face)."""
    return getattr(g, 'kind', default)


class KindedSetter(object):
    """A boundary-condition setter ``(x, flag) -> g | None`` whose faces may carry a kind (:func:`declare`)."""
    declaresBoundaryKinds = True

    def __init__(self, setter):
        if isinstance(setter, KindedSetter):
            setter = setter.setter
        self.setter = setter

    def __call__(self, *args):
        return self.setter(*args)

    def __repr__(self):
        return "BoundaryKinds.declare(%r)" % (self.setter,)


def declare(setter):
    """Mark ``setter`` as one whose faces may name their kind (see the module docstring). Idempotent."""
    return KindedSetter(setter)


def declaresKinds(setter):
    """Whether ``setter`` was passed through :func:`declare`."""
    return bool(getattr(setter, 'declaresBoundaryKinds', False))


def split(gReturn):
    """The ``(g, gFlag)`` of a setter's return, as the code that reads setters has always done: a bare ``g`` is
    ``(g, 1)``; a ``(g, gFlag)`` tuple keeps its flag (``gFlag`` 2 marks the nonlinear/Darcy kinds)."""
    try:
        return gReturn[0], gReturn[1]
    except TypeError:
        return gReturn, 1
