Param(
    [string]$PythonExe = "C:\\Users\\aesteva\\Downloads\\python-3.13.12-embed-amd64\\python.exe"
)

Write-Host "Using Python executable: $PythonExe"

if (-Not (Test-Path $PythonExe)) {
    Write-Host "ERROR: Python executable not found at $PythonExe" -ForegroundColor Red
    Write-Host "Edit this script and set the correct path to your embeddable python.exe, or pass -PythonExe <path>"
    exit 1
}

# Ensure site is enabled in the embeddable distribution (user must have uncommented import site in the _pth file)
Write-Host "Running ensurepip and upgrading pip/setuptools/wheel..."
& $PythonExe -m ensurepip --upgrade
& $PythonExe -m pip install --upgrade pip setuptools wheel

Write-Host "Installing packages from requirements.txt (this may take a while)..."
& $PythonExe -m pip install --upgrade -r "$(Join-Path (Split-Path -Parent $PSScriptRoot) 'requirements.txt')"

Write-Host "Installation finished. To register a kernel for Jupyter/VS Code, run:" -ForegroundColor Green
Write-Host "  & $PythonExe -m ipykernel install --user --name fea-embed --display-name \"Python (embeddable 3.13)\""

Write-Host "If torch installation fails due to platform-specific wheels, visit https://pytorch.org to get the recommended install command."
Write-Host "Done."