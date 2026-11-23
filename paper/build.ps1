param([string]$BuildDirectory)
$ErrorActionPreference = 'Stop'
$source = $PSScriptRoot
if (-not $BuildDirectory) {
    $BuildDirectory = Join-Path ([IO.Path]::GetTempPath()) ('permission-paper-' + [IO.Path]::GetRandomFileName())
}
$work = (New-Item -ItemType Directory -Path $BuildDirectory -Force).FullName
if ($work -eq $source) { throw 'Use a separate LaTeX build directory.' }
foreach ($name in 'main.tex','results.tex','natural.tex','native.tex','discussion.tex','appendix.tex','references.bib') {
    Copy-Item -LiteralPath (Join-Path $source $name) -Destination $work
}
$graphics = (New-Item -ItemType Directory -Path (Join-Path $work 'generated') -Force).FullName
Get-ChildItem -LiteralPath (Join-Path $source 'generated') -File | Copy-Item -Destination $graphics
Push-Location -LiteralPath $work
try {
    foreach ($step in 'latex','bibtex','latex','latex') {
        if ($step -eq 'bibtex') {
            $log = & bibtex authorization-review 2>&1
        } else {
            $log = & pdflatex -no-shell-escape -halt-on-error -interaction=nonstopmode -jobname=authorization-review main.tex 2>&1
        }
        if ($LASTEXITCODE -ne 0) {
            $log | Select-Object -Last 35 | Write-Output
            throw "Failed $step build in $work"
        }
    }
    $finalLog = Get-Content -LiteralPath 'authorization-review.log' -Raw
    if ($finalLog -match 'undefined references|Citation .+ undefined|Reference .+ undefined') {
        throw 'Unresolved LaTeX reference or citation.'
    }
    Copy-Item -LiteralPath 'authorization-review.pdf' -Destination (Join-Path $source 'authorization-review.pdf') -Force
    Write-Output "PDF built. Diagnostic files in $work"
} finally {
    Pop-Location
}
