# Environment and dependency audit

## Manuscript reference environment

The repository declares Python `>=3.10,<3.12` and pins direct dependencies in
`requirements.txt`. The manuscript timing reference is Windows 11, Python 3.11.9,
PyTorch 2.3.1 CPU, a single Intel Core i7-13700H core, batch size one, and single-thread
library settings.

## CI target

`.github/workflows/ci.yml` targets Python 3.10 and 3.11, installs the pinned runtime
requirements, generates synthetic inputs, and runs the repository checks.

## Workspace validation note

The current assistant workspace has a different Python/library stack, so successful smoke
runs here demonstrate code-path executability rather than reproduction of the manuscript's
reported millisecond timing values. Numerical timing comparison should use the reference
environment above.
