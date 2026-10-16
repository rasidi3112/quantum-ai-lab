# Contributing

Thanks for helping improve Quantum AI Lab! Bug reports, new algorithms, better demos and documentation fixes are all welcome.

## Development setup

```bash
git clone https://github.com/rasidi3112/quantum-ai-lab.git
cd quantum-ai-lab
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Before opening a pull request

```bash
ruff check .          # lint, import order and NumPy-style docstrings
mypy                  # static type checking
pytest                # full test suite (~20 s)
qailab tour           # smoke-test every module
```

## Guidelines

- **Document every public API.** Use NumPy-style docstrings (`Parameters`, `Returns`, `Raises`) and cite the paper an algorithm comes from in the module docstring. `ruff` fails on missing docstrings.
- **Type everything.** Annotate all function signatures; `mypy` runs in CI.
- **Use `seed`, not global state.** Accept an optional `seed` and draw randomness from `np.random.default_rng(seed)`, never from `np.random.*` module functions.
- **Test the physics, not just the code path.** A new algorithm should come with a test that checks a known result: an exact energy, a textbook probability, a conserved quantity, or a gradient checked against finite differences.
- **Stay dependency-light.** The core runs on NumPy and SciPy only. Optional integrations (Qiskit, PennyLane, PyTorch) belong behind optional extras.
- **Keep examples fast.** Each `examples/` script should finish in well under a minute on a laptop.
- **One topic per pull request.** Small, focused PRs are reviewed fastest.

## Good first issues

- Add a noise model (depolarizing / amplitude damping) to the crypto or optimization modules.
- Add Simon's or Bernstein–Vazirani to `quantum_algorithms`.
- Add a quantum-kernel benchmark against a classical RBF kernel.
- Port a demo to a Jupyter notebook.
