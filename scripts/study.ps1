param(
    [Parameter(Mandatory=$true)][ValidateSet('deepseek','jev','aliyun')][string]$Provider,
    [Parameter(Mandatory=$true)][ValidateSet('pilot','review','risk','coverage','conditions','confirmation','parameter_calibration','parameter_test','parameter_versions','parameter_final','reply_validation','reply_precedent','shared_review','commitment_review','retained_validation','jev_triage','parameter_router','parameter_router_extra','material_facts','material_fact_replies','full_permission','full_permission_review','numeric_permission','numeric_permission_review','natural_direct','natural_review','natural_extract','natural_matched','replica_direct','replica_review','replica_extract','replica_matched','branch_direct','branch_review','branch_original','branch_extract','branch_attest','scope_confirm_direct','scope_confirm_review','scope_confirm_original','scope_confirm_extract','scope_confirm_attest','public_transfer_direct','public_transfer_extract','native_direct','native_screen','native_bind','native_guard_bind')][string]$Study,
    [ValidateRange(1,16)][int]$Workers=8
)
$ErrorActionPreference='Stop'
if ($Study -in @('full_permission','full_permission_review','numeric_permission','numeric_permission_review') -and ($Provider -eq 'aliyun' -or ($Study -like '*_review' -and $Provider -ne 'deepseek'))) { throw 'Unsupported full-study provider.' }
if ($Study -eq 'material_fact_replies' -and $Provider -ne 'deepseek') { throw 'The reply experiment uses only DeepSeek.' }
if ($Study -eq 'material_facts' -and $Provider -notin @('deepseek','jev')) { throw 'The primary experiment supports DeepSeek and the Jev comparator.' }
if ($Provider -eq 'aliyun') {
    if ($Study -in @('confirmation','parameter_calibration','parameter_test','parameter_versions','parameter_final','reply_validation','reply_precedent','shared_review','commitment_review')) { throw 'This experiment uses Jev and DeepSeek.' }
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
