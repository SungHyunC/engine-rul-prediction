param(
    [string]$PythonPath = '',
    [switch]$SkipDocuments
)
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
if (-not $PythonPath) {
    $taskCandidates = @((Join-Path $taskRoot '.venv\Scripts\python.exe'), (Join-Path $env:USERPROFILE 'anaconda3\python.exe'))
    $PythonPath = $taskCandidates | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } | Select-Object -First 1
}
if (-not $PythonPath) { throw 'Pass -PythonPath for the environment with project and PyTorch dependencies.' }
$taskConfig = Get-Content -Raw -LiteralPath (Join-Path $taskRoot 'configs\default.json') | ConvertFrom-Json
$taskDataset = if ([IO.Path]::IsPathRooted($taskConfig.dataset)) { $taskConfig.dataset } else { Join-Path $taskRoot $taskConfig.dataset }
New-Item -ItemType Directory -Path (Join-Path $taskRoot 'output') -Force | Out-Null
if (-not (Test-Path -LiteralPath $taskDataset -PathType Leaf)) {
    & (Join-Path $taskRoot 'run.ps1') -Stage download -PythonPath $PythonPath
} else {
    $taskSourceManifest = Join-Path (Split-Path -Parent $taskDataset) 'source_manifest.json'
    if (-not (Test-Path -LiteralPath $taskSourceManifest -PathType Leaf)) { throw 'Acquisition manifest missing; run the download stage to verify the existing source.' }
    $taskExpectedSource = Get-Content -Raw -LiteralPath $taskSourceManifest | ConvertFrom-Json
    $taskActualHash = (Get-FileHash -LiteralPath $taskDataset -Algorithm SHA256).Hash
    if ($taskActualHash -ne $taskExpectedSource.sha256) { throw 'Source integrity check failed; the existing dataset was preserved.' }
}
& (Join-Path $taskRoot 'run.ps1') -Stage test -PythonPath $PythonPath 2>&1 | Tee-Object -FilePath (Join-Path $taskRoot 'output\test_results.txt')
& (Join-Path $taskRoot 'run.ps1') -Stage full -PythonPath $PythonPath
if (-not $SkipDocuments) {
    & (Join-Path $PSScriptRoot 'rebuild_delivery.ps1') -PythonPath $PythonPath
} else {
    foreach ($taskCheck in @('verify_outputs.py','verify_extended.py','analyze_errors.py')) {
        & $PythonPath (Join-Path $PSScriptRoot $taskCheck)
        if ($LASTEXITCODE -ne 0) { throw "Post-experiment check failed: $taskCheck" }
    }
}
