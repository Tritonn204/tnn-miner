# Script to swap HIPRTC DLL versions for testing
# Usage: .\scripts\swap_hiprtc_version.ps1 [6.1|6.4|restore]

param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("6.1", "6.4", "restore")]
    [string]$Version
)

$ProjectRoot = Split-Path $PSScriptRoot -Parent
$LibHiprtcDir = Join-Path $ProjectRoot "lib\hiprtc"
$BackupDir = Join-Path $ProjectRoot "lib\hiprtc_backup"

function Backup-Current {
    Write-Host "Backing up current DLLs..." -ForegroundColor Cyan

    if (-not (Test-Path $BackupDir)) {
        New-Item -ItemType Directory -Path $BackupDir | Out-Null
    }

    if (Test-Path "$LibHiprtcDir\*.dll") {
        Copy-Item "$LibHiprtcDir\*.dll" $BackupDir -Force
        Write-Host "  Backed up to: $BackupDir" -ForegroundColor Green
    }
}

function Copy-ROCmDlls {
    param([string]$RocmVersion)

    $RocmBinDir = "C:\Program Files\AMD\ROCm\$RocmVersion\bin"

    if (-not (Test-Path $RocmBinDir)) {
        Write-Host "ERROR: ROCm $RocmVersion not found at: $RocmBinDir" -ForegroundColor Red
        exit 1
    }

    # Find the hiprtc DLLs
    $HiprtcDlls = Get-ChildItem "$RocmBinDir\hiprtc*.dll"

    if ($HiprtcDlls.Count -eq 0) {
        Write-Host "ERROR: No HIPRTC DLLs found in $RocmBinDir" -ForegroundColor Red
        exit 1
    }

    Write-Host "`nCopying ROCm $RocmVersion HIPRTC DLLs:" -ForegroundColor Cyan
    foreach ($dll in $HiprtcDlls) {
        Write-Host "  $($dll.Name) ($([math]::Round($dll.Length/1MB, 2)) MB)" -ForegroundColor Gray
        Copy-Item $dll.FullName $LibHiprtcDir -Force
    }

    Write-Host "`nDLLs copied to: $LibHiprtcDir" -ForegroundColor Green
    Write-Host "Rebuild your project to use these DLLs." -ForegroundColor Yellow
}

function Restore-Backup {
    if (-not (Test-Path $BackupDir)) {
        Write-Host "ERROR: No backup found at: $BackupDir" -ForegroundColor Red
        exit 1
    }

    Write-Host "Restoring from backup..." -ForegroundColor Cyan
    Copy-Item "$BackupDir\*.dll" $LibHiprtcDir -Force
    Write-Host "Restored original DLLs" -ForegroundColor Green
}

# Main logic
Write-Host "============================================" -ForegroundColor Magenta
Write-Host "  HIPRTC DLL Version Swapper" -ForegroundColor Magenta
Write-Host "============================================" -ForegroundColor Magenta

# Show current state
Write-Host "`nCurrent DLLs in lib/hiprtc:" -ForegroundColor White
if (Test-Path "$LibHiprtcDir\*.dll") {
    Get-ChildItem "$LibHiprtcDir\*.dll" | ForEach-Object {
        Write-Host "  $($_.Name) - $([math]::Round($_.Length/1MB, 2)) MB" -ForegroundColor Gray
    }
} else {
    Write-Host "  (empty)" -ForegroundColor Gray
}

Write-Host ""

# Backup current DLLs before making changes
if ($Version -ne "restore") {
    Backup-Current
}

# Execute requested action
switch ($Version) {
    "6.1" {
        Write-Host "`nSwitching to ROCm 6.1 HIPRTC..." -ForegroundColor Yellow
        Copy-ROCmDlls "6.1"
    }
    "6.4" {
        Write-Host "`nSwitching to ROCm 6.4 HIPRTC..." -ForegroundColor Yellow
        Copy-ROCmDlls "6.4"
    }
    "restore" {
        Write-Host "`nRestoring original DLLs..." -ForegroundColor Yellow
        Restore-Backup
    }
}

Write-Host "`n============================================" -ForegroundColor Magenta
Write-Host "Done!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Magenta
