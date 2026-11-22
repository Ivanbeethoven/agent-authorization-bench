param(
    [Parameter(Mandatory=$true)][ValidateSet('deepseek','jev','aliyun')][string]$Provider,
    [Parameter(Mandatory=$true)][ValidateSet('pilot','review','risk','coverage','conditions','confirmation','parameter_calibration','parameter_test','parameter_versions','parameter_final')][string]$Study,
    [ValidateRange(1,16)][int]$Workers=8
)
$ErrorActionPreference='Stop'
if ($Provider -eq 'aliyun') {
    if ($Study -in @('confirmation','parameter_calibration','parameter_test','parameter_versions','parameter_final')) { throw 'This experiment uses Jev and DeepSeek.' }
    & (Join-Path $PSScriptRoot 'bailian.ps1') -Study $Study -Workers $Workers
    return
}
$varName=if($Provider -eq 'deepseek'){'DEEPSEEK_API_KEY'}else{'TYPESAFE_API_KEY'}
$credentialName=if($Provider -eq 'deepseek'){'deepseek.credential'}else{'typesafe.credential'}
$prior=[Environment]::GetEnvironmentVariable($varName,'Process')
try {
    if(-not $prior){
        $s=Get-Content -LiteralPath (Join-Path (Join-Path $env:LOCALAPPDATA 'AgentPermissionBench') $credentialName) -Raw | ConvertTo-SecureString
        [Environment]::SetEnvironmentVariable($varName,([pscredential]::new('key',$s)).GetNetworkCredential().Password,'Process')
    }
    $python=if(Get-Command uv -ErrorAction SilentlyContinue){uv python find}else{(Get-Command python).Source}
    & $python (Join-Path $PSScriptRoot '../review_study.py') $Study --provider $Provider --workers $Workers
    if($LASTEXITCODE -ne 0){throw 'Study did not complete.'}
} finally {
    [Environment]::SetEnvironmentVariable($varName,$prior,'Process')
}
