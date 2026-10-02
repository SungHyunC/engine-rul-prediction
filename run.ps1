param(
    [ValidateSet('download','inspect','prepare','train','benchmark','all','sensitivity','cnn','extended','finalize','predict','full','test')]
    [string]$Stage = 'inspect',
    [string]$PythonPath = '',
    [int]$Repeats = 3,
    [string]$Dataset = '',
    [ValidateSet('dev','test')]
    [string]$Split = 'test'
)
$ErrorActionPreference = 'Stop'
if (-not $PythonPath) {
    $taskPythonCandidates = @(
        (Join-Path $PSScriptRoot '.venv\Scripts\python.exe'),
        (Join-Path $env:USERPROFILE 'anaconda3\python.exe')
    )
    $PythonPath = $taskPythonCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
}
if (-not $PythonPath) { throw 'Pass -PythonPath with a Python executable containing the project dependencies.' }
$taskOriginalPythonPath = $env:PYTHONPATH
$taskOriginalLocation = Get-Location
try {
    Set-Location -LiteralPath $PSScriptRoot
    $env:PYTHONPATH = Join-Path $PSScriptRoot 'src'
    if ($Stage -eq 'download') { & $PythonPath tools/download_data.py }
    elseif ($Stage -eq 'test') { & $PythonPath -m pytest -q }
    else {
        $taskArguments = @('-m', 'ncmapss_rul', $Stage, '--repeats', $Repeats)
        if ($Dataset) { $taskArguments += @('--dataset', $Dataset) }
        if ($Stage -in @('predict','full')) { $taskArguments += @('--split', $Split) }
        & $PythonPath @taskArguments
    }
    if ($LASTEXITCODE -ne 0) { throw "Stage $Stage failed with exit code $LASTEXITCODE" }
} finally {
    $env:PYTHONPATH = $taskOriginalPythonPath
    Set-Location -LiteralPath $taskOriginalLocation
}
