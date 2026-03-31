param(
    [string]$Config = "configs/data.yaml",
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot

Push-Location $ProjectRoot
try {
    & $Python -m src.data.build_cohort --config $Config
    & $Python -m src.data.stage_filtered_tables --config $Config
    & $Python -m src.data.build_vocab --config $Config
    & $Python -m src.data.build_ddi_matrix --config $Config
    & $Python -m src.data.build_trajectories --config $Config
}
finally {
    Pop-Location
}
