"""
Loading a module by file path, in one place instead of fourteen.

Motivation
----------

Every arm has a file called `measurement.py` and one called `train.py`, and
the two simulators each have `dynamics.py`, `sensors.py`, `faults.py` and
`trajectories.py`. A bare `import sensors` therefore resolves to whichever
directory happens to sit earlier on `sys.path`, and the answer changes with
which file started the process.

That is not hypothetical. It happened, silently -- one arm was handed
another's module, every number it produced was wrong in a way that looked
plausible, and it was not caught for some time. Loading by explicit path under a
unique name removes the ambiguity entirely.

The alternative is to make the repository a package and use relative imports.
That is the better answer for a library. It is a worse answer here, because
every file in this project is also a script with a `__main__` block that runs
its own checks, and relative imports do not work in a module run directly.
Keeping each file independently runnable is worth more than import purity: a
check nobody can run is a check nobody runs.
"""

import importlib.util


def load_module(path, name):
    """Import the file at `path` under `name`, ignoring sys.path entirely.

    `name` has to be unique within the process. Two calls with the same name
    return two independent modules, which is usually not what anyone wants --
    hence the convention of naming by purpose and caller, as in
    "bhr_train_for_health" rather than "train".
    """
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
