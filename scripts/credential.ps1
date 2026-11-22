param([ValidateSet('typesafe','deepseek')][string]$Provider='typesafe')
$ErrorActionPreference = 'Stop'
$folder = Join-Path $env:LOCALAPPDATA 'AgentPermissionBench'
[void][System.IO.Directory]::CreateDirectory($folder)
$path = Join-Path $folder "$Provider.credential"
$secret = Read-Host "$Provider API key" -AsSecureString
if ($secret.Length -eq 0) { throw 'A key is required.' }
$protected = ConvertFrom-SecureString -SecureString $secret
[System.IO.File]::WriteAllText($path, $protected)
$secret.Dispose()
Write-Output "$Provider credential stored for the current Windows user."
