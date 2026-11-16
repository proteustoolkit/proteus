# Publishing py-proteus to PyPI

This is the PyPI part of a release; [RELEASE.md](RELEASE.md) is still the procedure (versions, release notes,
Zenodo, conda-forge, Spack). `.github/workflows/release.yml` builds the **sdist** (no wheels: PETSc, petsc4py and
an MPI-enabled HDF5 must match the user's MPI, so a wheel would pin an ABI nobody can match) from the tagged commit,
checks it (`tools/check_sdist.sh`: size, content, metadata, `twine check --strict`), and publishes it with PyPI
Trusted Publishing (OIDC: no API token, no secret). The same shape as ymf's `RELEASING.md`.

The distribution is **`py-proteus`**; the import name stays `proteus`. `pip install py-proteus` is a source build that
needs a compiler, an MPI, and PETSc, petsc4py and HDF5 already built to match (conda-forge, Spack and Nix remain the
supported routes). Its sdist installs and runs in `sdist.yml`, which this workflow does not repeat.

| tag (bare number) | goes to |
|---|---|
| `2.0.0rc1`, `2.0.0a1`, `2.0.0b1` | TestPyPI (environment `testpypi`) |
| `2.0.0` | PyPI (environment `pypi`) |

The tag must equal the version in `setup.py` (PEP 440, as RELEASE.md's three files carry it); the workflow refuses a
mismatch. `2.0.0.dev` is not publishable (the workflow refuses a dev version).

## Zenodo and the tag

Zenodo fires on a GitHub *release* being published, never on a tag push, and this workflow never creates or edits a
release. So:

* a **release candidate** (`2.0.0rc1`): push the tag only (`git tag -a 2.0.0rc1 -m ... && git push origin 2.0.0rc1`).
  No release, so no Zenodo version, and TestPyPI gets the sdist;
* a **final version**: RELEASE.md step 6 as written (`gh release create X.Y.Z ...`). That creates the tag, which starts
  this workflow once, and publishes the release, which fires Zenodo once. Nothing here can cause the 409 of a
  repeated release.

## What the maintainer does by hand (once)

Names, checked 2026-11-16: `py-proteus` and `py_proteus` are 404 on pypi.org and test.pypi.org. Tryton's client library
is the PyPI project `proteus` and installs a top-level module `proteus`; it and py-proteus cannot share an environment
(PyPI cannot declare a conflict, and the README should say so). PyPI can still refuse a name too close to an existing
one: register soon.

1. **TestPyPI**: log in at test.pypi.org, *Your account → Publishing → Add a new pending publisher*: project name
   `py-proteus`, owner `proteustoolkit` (or `cekees` to rehearse from the fork), repository `proteus`, workflow
   `release.yml`, environment `testpypi`.
2. **PyPI**: the same at pypi.org with environment `pypi`. A pending publisher creates the project on first upload, so
   no upload is needed to claim the name.
3. **GitHub environments** on the repository that runs the workflow: Settings → Environments → create `testpypi` and
   `pypi` (add yourself as a required reviewer on `pypi`). No secrets.
4. **ymf first.** `py-proteus` requires `ymf`; until `ymf` is on PyPI, `pip install py-proteus` cannot resolve it. Do
   ymf's `RELEASING.md` first. An rc on TestPyPI installs with the extra index for dependencies:

   ```bash
   pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ \
       --no-build-isolation "py-proteus==2.0.0rc1"   # in an environment that already has PETSc, petsc4py, h5py, mpi4py
   ```

   (`ymf` must then also come from TestPyPI or PyPI; install it first.)
5. **Release candidate**: set the three version files to `2.0.0rc1` in a PR on main (RELEASE.md step 2), push the tag.
   A TestPyPI file name can never be reused: a mistake is fixed by `rc2`.

To rehearse without publishing: `tools/check_sdist.sh` in an environment with the build requirements and `build` and
`twine` installed.

## How ymf and the conda-forge pins interact (nothing here changes them)

* **ymf.** The environment files install ymf from `git+https://github.com/proteustoolkit/ymf.git@main` because it was
  not on PyPI; `install_requires` already says plain `ymf`, with no floor. After ymf's first tag is on PyPI, the
  follow-up (a separate PR, C3's workflows and the environment files) replaces `@main` with the tag or `ymf==X.Y.Z`,
  and the py-proteus requirement can gain the floor (`ymf>=0.3`) once a release exists to name.
* **conda-forge** builds from the GitHub release and does not use this workflow; the feedstock's `ymf` requirement is
  separate and must exist on conda-forge before a feedstock can depend on it. Nothing PyPI does touches it.
* **Spack** (`py-proteus`) already uses the same distribution name.
* **The `sdist.yml` and `pytest-ways.yml` globs** matched the old file name (`dist/proteus-*.tar.gz`); the rename makes
  setuptools write `py_proteus-*.tar.gz`, so both now read `dist/*.tar.gz` (changed in this PR with cekees's go).
* **License form.** `[project]` uses `license = { text = "MIT" }`, not the SPDX string: the locked CI environments carry
  setuptools 73 and build with `--no-isolation`. Switch to `license = "MIT"` once they carry setuptools >= 77.
