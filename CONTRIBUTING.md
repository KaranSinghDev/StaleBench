# Contributing

Thanks for your interest in StaleBench. Contributions are welcome.

## Bugs and ideas
Open an issue describing the problem or proposal. For bugs, include your Python version,
how you ran StaleBench, and the output you saw.

## Pull requests
1. Fork the repository and create a branch.
2. Make your change. Keep it focused and match the existing style.
3. Run the tests: `python -m pytest -q`. Add a test if you add behaviour.
4. Open a pull request describing what changed and why.

## Development setup
```bash
pip install -e .[bm25,dense,dev]
python -m pytest -q
```

By contributing, you agree that your contributions are licensed under the Apache-2.0
license that covers this project.
