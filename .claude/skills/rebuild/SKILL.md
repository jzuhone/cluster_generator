---
name: rebuild
description: Recompile the Cython extensions after editing cluster_generator/opt/*.pyx. Use whenever a .pyx file changed, imports fail with stale compiled code, or a fresh build of the extensions is needed.
---

# Rebuild Cython extensions

The package has compiled Cython extensions in `cluster_generator/opt/` (`cython_utils.pyx`, `structures.pyx`). A stale `.so` is silently used if you don't force a rebuild after editing a `.pyx`.

Steps:

1. From the repo root, clean compiled artifacts:
   ```
   ./clean.sh
   ```
   This removes `*.so`, `*.pyc`, `build/`, `dist/`, and `egg-info`.

2. Reinstall editable so the extensions recompile:
   ```
   pip install -e .
   ```

3. If the build fails, check that a C compiler and `python3-dev`/`libm` are available, and that `numpy>=2.0,<3.0` and `Cython` are installed.

Report the outcome — whether the rebuild succeeded and any compiler warnings/errors.
