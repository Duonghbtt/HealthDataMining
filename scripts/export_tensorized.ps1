param(
    [string]$Config = "configs/data.yaml",
    [string]$OutputRoot = "",
    [switch]$Overwrite
)

$args = @("-m", "src.data.export_tensorized_trajectories", "--config", $Config)
if ($OutputRoot) {
    $args += @("--output-root", $OutputRoot)
}
if ($Overwrite) {
    $args += "--overwrite"
}

python @args
