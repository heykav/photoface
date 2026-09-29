# Security policy

## Reporting a vulnerability

Please report security problems privately, not in a public issue:

- preferably through GitHub's private vulnerability reporting
  ("Report a vulnerability" under this repository's **Security** tab), or
- by email to heykavofficial@gmail.com.

Include what you found, how to reproduce it, and the version or commit.

This is a personal project maintained on a best-effort basis. There is no
guaranteed response time and no bug bounty, but reports are read and valid
issues will be fixed and credited unless you prefer otherwise.

## Scope

In scope: the code in this repository, including the model download and
verification in `model_files.py`, database handling, and the packaged
builds produced by `.github/workflows/build.yml`.

Out of scope: vulnerabilities in dependencies (PySide6/Qt, OpenCV, NumPy,
Pillow) or in the upstream ONNX models themselves; please report those to
their projects. Recognition accuracy issues are bugs, not security issues.

## Supported versions

Only the latest commit on `main` is supported.
