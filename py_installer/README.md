# PyInstaller

This directory contains the files needed to build a standalone application for Windows  
and Linux using [PyInstaller](https://pyinstaller.org).

## Build Instructions

1. Create a new virtual environment with dependencies from `requirements-<type>.txt`.
2. Run `pyinstaller py_installer/dinora-<type>.py`.
3. The executable will be located in the `dist/dinora` directory.

There is also a GitHub Actions workflow file located at `.github/workflows/build.yml`.
