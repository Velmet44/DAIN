# Publish the DAIN coordinator metadata (S23).
#
#   powershell -File Scripts\publish-meta.ps1 -CoordUrl https://box.tail-scale.ts.net -ApiKey dain-client -ModelId llama-3-2-1b-int4s -Push
#
# Writes Client/public/meta.json (served at /DAIN/meta.json next Pages deploy)
# and, with -Push, commits + pushes it so clients and nodes pick the new
# coordinator URL on their next open/reconnect. NEVER put the node join token
# in this file.
param(
    [Parameter(Mandatory = $true)][string]$CoordUrl,
    [string]$ApiKey = "",
    [string]$ModelId = "",
    [switch]$Push
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$path = Join-Path $root "Client\public\meta.json"

$doc = [ordered]@{
    coord_url = $CoordUrl
    api_key   = $ApiKey
    model_id  = $ModelId
}
$json = $doc | ConvertTo-Json
Set-Content -Path $path -Value $json -Encoding UTF8
Write-Host "meta.json written: $path"
Write-Host $json

if ($Push) {
    git -C $root add Client/public/meta.json
    git -C $root commit -m "meta: coordinator at $CoordUrl"
    git -C $root push
    Write-Host "Pushed - the Pages site redeploys (~1-2 min), then clients/nodes pick it up."
} else {
    Write-Host "Review the file, then commit+push (or rerun with -Push)."
}
