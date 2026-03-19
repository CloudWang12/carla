$ErrorActionPreference = "Stop"

if (-not (Test-Path ".venv")) {
    py -3.12 -m venv .venv
}

& ".\.venv\Scripts\python.exe" -m pip install --upgrade pip
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt

Write-Host "Environment is ready."
Write-Host "Train model: .\.venv\Scripts\python.exe main.py train"
Write-Host "Run CARLA controller: .\.venv\Scripts\python.exe main.py run"

