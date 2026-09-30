param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("single_full_window", "visual_deepest", "visual_deepest_repeat8")]
    [string]$Variant,
    [string]$Subjects = "sub-01,sub-02,sub-03,sub-04,sub-05,sub-06,sub-07,sub-08,sub-09,sub-10",
    [string]$DataRoot = "data\things_eeg",
    [string]$OutputDir = "results\cevr_ablation",
    [int]$Seed = 20260815,
    [switch]$Cpu
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$ConfigMap = @{
    single_full_window = "configs\single_full_window_control.json"
    visual_deepest = "configs\visual_deepest.json"
    visual_deepest_repeat8 = "configs\visual_deepest_repeat8.json"
}
$Python = "python"
$TrainScript = Join-Path $Root "scripts\train_intra_subject.py"
$ConfigPath = Join-Path $Root $ConfigMap[$Variant]
$EegDir = Join-Path $Root (Join-Path $DataRoot "preprocessed_eeg_npy")
$VisualRoot = Join-Path $Root (Join-Path $DataRoot "image_feature\ViT-B32-multilevel-8")
$OutputRoot = Join-Path $Root (Join-Path $OutputDir $Variant)

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
$SubjectValues = @($Subjects -split "[,\s]+" | Where-Object { $_ })
foreach ($Subject in $SubjectValues) {
    $Args = @(
        $TrainScript,
        "--config", $ConfigPath,
        "--subject", $Subject,
        "--eeg-data-dir", $EegDir,
        "--visual-feature-root", $VisualRoot,
        "--output-dir", $OutputRoot,
        "--seed", "$Seed"
    )
    if ($Cpu) {
        $Args += "--cpu"
    }
    Write-Host "START $Subject $Variant"
    & $Python @Args
    if ($LASTEXITCODE -ne 0) {
        throw "Training failed for $Subject $Variant."
    }
    Write-Host "DONE $Subject $Variant"
}
