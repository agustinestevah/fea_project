#!/usr/bin/env python
"""
install_packages.py  –  Install all fea_project dependencies.

Works on Windows, Linux and macOS.  Designed so that you can run it
with whatever Python interpreter you plan to use for the project.

Usage
-----
    python scripts/install_packages.py            # uses current python
    python scripts/install_packages.py --dry-run   # print what *would* be installed
    C:/path/to/python.exe scripts/install_packages.py  # use a specific python

The script:
  1. Upgrades pip / setuptools / wheel.
  2. Installs PyTorch (CPU-only or CUDA URL depending on --cuda flag).
  3. Installs every remaining package from requirements.txt.
  4. Runs a quick smoke-test importing every critical module.
  5. Optionally registers a Jupyter kernel.
"""

import argparse
import importlib
import subprocess
import sys
from pathlib import Path

# ---------- configuration ----------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REQUIREMENTS = PROJECT_ROOT / "requirements.txt"

# Modules to smoke-test after install (import-name → pip-name)
SMOKE_TEST = {
    "pandas":                "pandas",
    "numpy":                 "numpy",
    "torch":                 "torch",
    "sentence_transformers": "sentence-transformers",
    "transformers":          "transformers",
    "sklearn":               "scikit-learn",
    "datasets":              "datasets",
    "plotly":                "plotly",
    "openai":                "openai",
    "pydantic":              "pydantic",
    "tqdm":                  "tqdm",
    "openpyxl":              "openpyxl",
    "scipy":                 "scipy",
    "sentencepiece":         "sentencepiece",
    "papermill":             "papermill",
    "scrapbook":             "scrapbook",
    "optuna":                "optuna",
    "accelerate":            "accelerate",
    "IPython":               "ipykernel",
}


def pip(*args: str, dry_run: bool = False) -> int:
    """Run pip with given args."""
    cmd = [sys.executable, "-m", "pip", *args]
    if dry_run:
        print(f"  [dry-run] {' '.join(cmd)}")
        return 0
    print(f"  > {' '.join(cmd)}")
    return subprocess.call(cmd)


def main() -> None:
    parser = argparse.ArgumentParser(description="Install fea_project dependencies.")
    parser.add_argument(
        "--cuda",
        type=str,
        default=None,
        metavar="VERSION",
        help="Install CUDA-enabled PyTorch (e.g. '12.6'). Omit for CPU-only.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing.",
    )
    parser.add_argument(
        "--register-kernel",
        action="store_true",
        help="After install, register a Jupyter kernel named 'fea-project'.",
    )
    args = parser.parse_args()

    print(f"Python  : {sys.executable}")
    print(f"Version : {sys.version}")
    print(f"Project : {PROJECT_ROOT}")
    print()

    # ---- Step 1: upgrade pip ----
    print("=" * 60)
    print("STEP 1  Upgrading pip / setuptools / wheel")
    print("=" * 60)
    pip("install", "--upgrade", "pip", "setuptools", "wheel", dry_run=args.dry_run)

    # ---- Step 2: install torch (first, to avoid version conflicts) ----
    print()
    print("=" * 60)
    print("STEP 2  Installing PyTorch")
    print("=" * 60)
    if args.cuda:
        cuda_tag = "cu" + args.cuda.replace(".", "")
        pip(
            "install", "--upgrade",
            "torch",
            "--index-url", f"https://download.pytorch.org/whl/{cuda_tag}",
            dry_run=args.dry_run,
        )
    else:
        pip("install", "--upgrade", "torch", dry_run=args.dry_run)

    # ---- Step 3: install everything else from requirements.txt ----
    print()
    print("=" * 60)
    print("STEP 3  Installing from requirements.txt")
    print("=" * 60)
    if not REQUIREMENTS.exists():
        print(f"ERROR: {REQUIREMENTS} not found!")
        sys.exit(1)
    pip("install", "--upgrade", "-r", str(REQUIREMENTS), dry_run=args.dry_run)

    # ---- Step 4: smoke-test imports ----
    if not args.dry_run:
        print()
        print("=" * 60)
        print("STEP 4  Smoke-test: importing all critical modules")
        print("=" * 60)
        ok, fail = [], []
        for mod, pkg in SMOKE_TEST.items():
            try:
                importlib.import_module(mod)
                ok.append(mod)
            except Exception as exc:
                fail.append((mod, pkg, str(exc)))

        for m in ok:
            print(f"  ✓ {m}")
        for m, pkg, err in fail:
            print(f"  ✗ {m}  (pip: {pkg})  – {err}")

        if fail:
            print(f"\n⚠  {len(fail)} module(s) failed to import. "
                  "Try installing them manually:")
            for _, pkg, _ in fail:
                print(f"    pip install {pkg}")
        else:
            print(f"\n✓ All {len(ok)} modules imported successfully.")

    # ---- Step 5 (optional): register Jupyter kernel ----
    if args.register_kernel:
        print()
        print("=" * 60)
        print("STEP 5  Registering Jupyter kernel 'fea-project'")
        print("=" * 60)
        pip(
            "-m", "ipykernel", "install",
            "--user",
            "--name", "fea-project",
            "--display-name", "Python (fea_project)",
            dry_run=args.dry_run,
        )
        # Correct: call python -m ipykernel directly
        if not args.dry_run:
            subprocess.call([
                sys.executable, "-m", "ipykernel", "install",
                "--user",
                "--name", "fea-project",
                "--display-name", "Python (fea_project)",
            ])

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
