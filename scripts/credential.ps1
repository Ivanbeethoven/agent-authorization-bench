$ErrorActionPreference = 'Stop'
$folder = Join-Path $env:LOCALAPPDATA 'AgentPermissionBench'
[void][System.IO.Directory]::CreateDirectory($folder)
$path = Join-Path $folder 'typesafe.credential'
$secret = Read-Host 'TypeSafe API key' -AsSecureString
if ($secret.Length -eq 0) { throw 'A key is required.' }
$protected = ConvertFrom-SecureString -SecureString $secret
[System.IO.File]::WriteAllText($path, $protected)
$secret.Dispose()
Write-Output 'TypeSafe credential stored for the current Windows user.'
