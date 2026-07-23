---
title: Conda environment
---

# Working in the Conda environment

Everything runs inside the **`rlbootcamp`** environment. This page is the
muscle-memory you need — it is short on purpose.

## Activate / deactivate

```bash
conda activate rlbootcamp     # start working
conda deactivate              # leave the environment
```

Your prompt shows `(rlbootcamp)` when it is active.

!!! tip "If you remember one thing"
    "It says the module isn't installed, but I installed it" is almost always a
    deactivated environment. Check the prompt first.

## Run a single command without activating

Handy in scripts, IDEs, cron jobs, or when you simply forget:

```bash
conda run -n rlbootcamp python scripts/smoke_test.py
conda run -n rlbootcamp tensorboard --logdir runs/
```

!!! note "Confirm which Python you're using"
    ```bash
    conda run -n rlbootcamp python -c "import sys; print(sys.executable)"
    ```
    The path must contain `envs/rlbootcamp`. If it does not, you are running a
    different interpreter and that explains your import errors.

## Keep the environment up to date

When `environment.yml` changes, refresh rather than recreate:

```bash
git pull
conda env update -f environment.yml --prune
```

`--prune` removes packages that were dropped from the file.

## Add a package for an experiment

```bash
conda activate rlbootcamp
pip install <package>
```

Mixing `pip` and `conda` inside one environment is fine as long as you install
conda packages first and pip packages second — which is what `environment.yml`
already does. Avoid `conda install` *after* heavy pip installs; the solver can
downgrade things underneath pip's feet.

## Use it in Jupyter

JupyterLab and `ipykernel` ship with the environment. Register it as a kernel
once, so notebooks can find it:

```bash
conda activate rlbootcamp
python -m ipykernel install --user --name rlbootcamp --display-name "Python (rlbootcamp)"
jupyter lab
```

Then pick the **Python (rlbootcamp)** kernel in the notebook's kernel menu.

!!! warning "The notebook kernel is a separate choice"
    Activating the environment in your terminal does **not** change which kernel
    an already-open notebook uses. If imports fail inside a notebook but work in
    the terminal, you are on the wrong kernel. Check the top-right of the
    notebook.

## Use it in VS Code / PyCharm

=== "VS Code"

    ++ctrl+shift+p++ (++cmd+shift+p++ on macOS) → **Python: Select Interpreter**
    → choose the one under `envs/rlbootcamp`.

    The interpreter is remembered per workspace, so do this once per project
    folder. The integrated terminal picks it up automatically.

=== "PyCharm"

    **Settings → Project → Python Interpreter → Add Interpreter → Add Local
    Interpreter → Conda Environment → Use existing environment** → select
    `rlbootcamp`.

    If PyCharm cannot find `conda`, point it at the executable directly
    (`~/miniforge3/bin/conda`, or `...\miniforge3\Scripts\conda.exe` on Windows).

## Start over (last resort)

Recreating is cheap and fixes a surprising number of problems:

```bash
conda deactivate
conda env remove -n rlbootcamp
conda env create -f environment.yml
```

---

Next: [Troubleshooting →](troubleshooting.md)
