param(
    [Parameter(Mandatory)][ValidateSet('granite','agentdecision')][string]$Model,
    [Parameter(Mandatory)][string]$Server,
    [Parameter(Mandatory)][string]$Weights,
    [Parameter(Mandatory)][string]$Template,
    [string]$Adapter,
    [Parameter(Mandatory)][ValidateSet('CUDA','Vulkan')][string]$Backend,
    [Parameter(Mandatory)][string]$Device,
    [ValidateRange(1024,65535)][int]$Port=18374,
    [switch]$Controls
)

$ErrorActionPreference='Stop'
$binary=(Resolve-Path -LiteralPath $Server).Path
$weightsFile=(Resolve-Path -LiteralPath $Weights).Path
$templateFile=(Resolve-Path -LiteralPath $Template).Path
$benchRoot=(Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$info=[Diagnostics.ProcessStartInfo]::new()
$info.FileName=$binary
$info.WorkingDirectory=Split-Path -Parent $binary
$info.UseShellExecute=$false
$info.CreateNoWindow=$true
$info.RedirectStandardOutput=$true
$info.RedirectStandardError=$true
$options=@('-m',$weightsFile,'--host','127.0.0.1','--port',[string]$Port,'--device',$Device,'-ngl','99','-c','8192','-np','2')
if ($Model -eq 'agentdecision') {
    if (-not $Adapter) { throw 'AgentDecision requires its decision adapter.' }
    $options+=@('--lora',(Resolve-Path -LiteralPath $Adapter).Path)
}
foreach ($opt in $options) { $info.ArgumentList.Add($opt) }
$process=[Diagnostics.Process]::new()
$process.StartInfo=$info
if (Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue) { throw "Port $Port is already in use." }
$null=$process.Start()
$output=$process.StandardOutput.ReadToEndAsync()
$errors=$process.StandardError.ReadToEndAsync()
$runName=if($Controls){"$Model-controls"}else{$Model}
$logRoot=Join-Path $benchRoot "data/local_review/$runName"
$null=New-Item -ItemType Directory -Force -Path $logRoot
try {
    $deadline=(Get-Date).AddMinutes(4)
    do {
        if ($process.HasExited) { throw "Local server exited with status $($process.ExitCode)." }
        try {
            $health=Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -NoProxy -TimeoutSec 3
            if ($health.status -eq 'ok') { break }
        } catch { }
        Start-Sleep -Seconds 1
    } while ((Get-Date) -lt $deadline)
    if ($health.status -ne 'ok') { throw 'Local model loading timed out.' }
    Push-Location -LiteralPath $benchRoot
    try {
        if ($Controls) {
            & uv run --no-project --python 3.14 --with jinja2 python local_review_controls.py $Model --template $templateFile --port $Port
        } else {
            & uv run --no-project --python 3.14 --with jinja2 python local_review.py $Model --template $templateFile --backend $Backend --port $Port
        }
        if ($LASTEXITCODE -ne 0) { throw 'Local evaluation failed.' }
    } finally { Pop-Location }
} finally {
    if (-not $process.HasExited) { $process.Kill(); $process.WaitForExit() }
    [IO.File]::WriteAllText((Join-Path $logRoot 'server.stdout.log'),$output.GetAwaiter().GetResult())
    [IO.File]::WriteAllText((Join-Path $logRoot 'server.stderr.log'),$errors.GetAwaiter().GetResult())
    $process.Dispose()
}
