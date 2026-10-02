param(
    [string]$PythonPath = (Join-Path $env:USERPROFILE 'anaconda3\python.exe'),
    [string]$DocumentPythonPath = (Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'),
    [string]$NodePath = (Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe')
)
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$taskOriginalLocation = Get-Location
$taskPreviousSlidesOutput = $env:SLIDES_OUTPUT_DIR
try {
    Set-Location -LiteralPath $taskRoot
    foreach ($taskExecutable in @($PythonPath, $DocumentPythonPath, $NodePath)) {
        if (-not (Test-Path -LiteralPath $taskExecutable -PathType Leaf)) { throw "Runtime not found: $taskExecutable" }
    }
    & $PythonPath tools/verify_outputs.py
    if ($LASTEXITCODE -ne 0) { throw 'Baseline output verification failed' }
    & $PythonPath tools/verify_extended.py
    if ($LASTEXITCODE -ne 0) { throw 'Extended output verification failed' }
    & $PythonPath tools/analyze_errors.py
    if ($LASTEXITCODE -ne 0) { throw 'Error analysis failed' }
    & $DocumentPythonPath tools/build_proposals.py
    if ($LASTEXITCODE -ne 0) { throw 'Proposal build failed' }
    & $DocumentPythonPath tools/build_reports.py
    if ($LASTEXITCODE -ne 0) { throw 'Report build failed' }
    # Finalize each deck in a fresh private revision before replacing delivery.
    $taskSlidesRevision = Join-Path $PSScriptRoot ('.slides-build\delivery-' + [guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $taskSlidesRevision -Force | Out-Null
    $env:SLIDES_OUTPUT_DIR = $taskSlidesRevision
    & $NodePath tools/build_slides.mjs
    if ($LASTEXITCODE -ne 0) { throw 'Presentation build failed' }
    foreach ($taskLanguage in @('ko','en')) {
        $taskSlideFile = 'presentation_' + $taskLanguage + '.pptx'
        Copy-Item -LiteralPath (Join-Path $taskSlidesRevision $taskSlideFile) -Destination (Join-Path $taskRoot ('output\' + $taskSlideFile)) -Force
    }
    & $PythonPath tools/package_submission.py
    if ($LASTEXITCODE -ne 0) { throw 'Submission packaging failed' }
} finally {
    $env:SLIDES_OUTPUT_DIR = $taskPreviousSlidesOutput
    Set-Location -LiteralPath $taskOriginalLocation
}
