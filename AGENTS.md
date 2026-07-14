# AITutor workspace notes

## Python environment

- Always use the Conda environment `aitutor` for Python commands in this repository.
- Interactive shell: `conda activate aitutor`
- Non-interactive/one-shot shell: `conda run -n aitutor <command>`
- Run tests with `python -m pytest` from that environment; do not use the base Conda Python.
- On this Windows workspace, prefer activating the environment in PowerShell over `conda run` because the latter can fail while printing Vietnamese pytest output with CP1258.
- If the host defines a non-boolean `DEBUG` value, set `$env:DEBUG='false'` for the pytest process.
