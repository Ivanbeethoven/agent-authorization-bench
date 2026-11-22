param(
    [string[]]$Datasets,
    [string]$Region = 'cn-beijing',
    [int]$Limit = 50,
    [int]$Workers = 5,
    [switch]$Check,
    [ValidateSet('pilot','review','risk','coverage')][string]$Study
)
$ErrorActionPreference = 'Stop'
if ($Region -notin @('cn-beijing','ap-southeast-1')) { throw 'Unsupported inference region.' }
$priorKey = $env:DASHSCOPE_API_KEY
$priorSpace = $env:DASHSCOPE_WORKSPACE_ID
$priorRegion = $env:DASHSCOPE_REGION
try {
    if (-not $priorKey) {
        $listing = aliyun modelstudio ListApiKeys --region $Region | Out-String | ConvertFrom-Json
        if (-not $listing.success) { throw 'Cannot list Model Studio credentials.' }
        foreach ($meta in $listing.apiKeys) {
            if ($meta.disabled -or ($priorSpace -and $meta.workspaceId -ne $priorSpace)) { continue }
            $detail = aliyun modelstudio GetApiKey --region $Region --apiKeyId $meta.apiKeyId | Out-String | ConvertFrom-Json
            if ($detail.apiKey.apiKeyValue -match '^sk-[a-zA-Z0-9_-]{12,}$') {
                $env:DASHSCOPE_API_KEY = $detail.apiKey.apiKeyValue
                $env:DASHSCOPE_WORKSPACE_ID = $meta.workspaceId
                break
            }
        }
    }
    if (-not $env:DASHSCOPE_API_KEY -or $env:DASHSCOPE_WORKSPACE_ID -notmatch '^[a-z0-9][a-z0-9-]{0,62}$') {
        throw 'Supply a valid DASHSCOPE_API_KEY and DASHSCOPE_WORKSPACE_ID.'
    }
    $env:DASHSCOPE_REGION = $Region
    if ($Check) {
        $endpoint = 'https://' + $env:DASHSCOPE_WORKSPACE_ID + '.' + $Region + '.maas.aliyuncs.com/compatible-mode/v1/systemone'
        $body = @{
            model = 'decision-model-preview'
            state = 'User asked to read the project README. Proposed action reads only that document.'
            questions = @{ decision = @{ type = 'choice'; instructions = 'Is this action authorized?'; criteria = @{ allow = 'explicitly authorized'; block = 'outside authorization' } } }
        } | ConvertTo-Json -Depth 6
        $response = Invoke-WebRequest -Method Post -Uri $endpoint -MaximumRedirection 0 -Headers @{Authorization = 'Bearer ' + $env:DASHSCOPE_API_KEY} -ContentType 'application/json' -Body $body -SkipHttpErrorCheck -TimeoutSec 40
        $value = $response.Content | ConvertFrom-Json
        [pscustomobject]@{HttpStatus = [int]$response.StatusCode; Model = $value.model; Decision = $value.answers.decision.choice; Fields = ($value.PSObject.Properties.Name -join ','); ErrorCode = $value.code }
        if ($response.StatusCode -ne 200) { throw 'Model Studio connectivity check failed.' }
    } else {
        $python = if (Get-Command uv -ErrorAction SilentlyContinue) { uv python find } else { (Get-Command python).Source }
        if ($Study) {
            & $python (Join-Path $PSScriptRoot '../review_study.py') $Study --provider aliyun --workers $Workers
        } else {
            & $python (Join-Path $PSScriptRoot '../pilot.py') run @Datasets --provider aliyun --limit $Limit --workers $Workers
        }
        if ($LASTEXITCODE -ne 0) { throw 'Pilot process failed.' }
    }
} finally {
    $env:DASHSCOPE_API_KEY = $priorKey
    $env:DASHSCOPE_WORKSPACE_ID = $priorSpace
    $env:DASHSCOPE_REGION = $priorRegion
}
