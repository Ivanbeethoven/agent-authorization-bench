param(
    [Parameter(Mandatory=$true)][string[]]$Datasets,
    [int]$Limit = 50,
    [int]$Workers = 5
)
$ErrorActionPreference = 'Stop'
$previous = $env:TYPESAFE_API_KEY
try {
    if (-not $previous) {
        $path = Join-Path $env:LOCALAPPDATA 'AgentPermissionBench/typesafe.credential'
        $secret = Get-Content -LiteralPath $path -Raw | ConvertTo-SecureString
        $env:TYPESAFE_API_KEY = ([pscredential]::new('key', $secret)).GetNetworkCredential().Password
    }
    $python = if (Get-Command uv -ErrorAction SilentlyContinue) { uv python find } else { (Get-Command python).Source }
    & $python (Join-Path $PSScriptRoot '../pilot.py') run @Datasets --limit $Limit --workers $Workers
    $code = $LASTEXITCODE
} finally {
    $env:TYPESAFE_API_KEY = $previous
}
exit $code
