param(
  [ValidateSet("create-wallet", "wallet", "address", "node", "gateway", "all")]
  [string]$Stage,

  [string]$PearlRoot = "F:\git\pearl",
  [string]$RpcUser = "rpcuser",
  [string]$RpcPassword = "rpcpass",
  [string]$MiningAddress = "",
  [int]$WalletStartupSeconds = 4
)

$ErrorActionPreference = "Stop"

function Require-Path {
  param([string]$PathToCheck, [string]$Label)
  if (-not (Test-Path -LiteralPath $PathToCheck)) {
    throw "$Label not found: $PathToCheck"
  }
}

function Require-MiningAddress {
  if ([string]::IsNullOrWhiteSpace($MiningAddress)) {
    throw "Stage '$Stage' requires -MiningAddress."
  }
}

function Start-StageWindow {
  param(
    [string]$TargetStage,
    [string]$StageMiningAddress = ""
  )

  $scriptPath = $MyInvocation.MyCommand.Path
  if ([string]::IsNullOrWhiteSpace($scriptPath)) {
    throw "Unable to determine script path for stage launch."
  }

  $argList = @(
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-File", $scriptPath,
    "-Stage", $TargetStage,
    "-PearlRoot", $PearlRoot,
    "-RpcUser", $RpcUser,
    "-RpcPassword", $RpcPassword
  )

  if (-not [string]::IsNullOrWhiteSpace($StageMiningAddress)) {
    $argList += @("-MiningAddress", $StageMiningAddress)
  }

  Start-Process -FilePath "pwsh" -ArgumentList $argList | Out-Null
}

function Get-MiningAddressFromWallet {
  Require-Path $prlctl "prlctl.exe"

  $output = & $prlctl -u $RpcUser -P $RpcPassword -s https://localhost:18554 getnewaddress 2>&1
  if ($LASTEXITCODE -ne 0) {
    throw "Failed to fetch mining address from wallet: $output"
  }

  $joined = ($output | ForEach-Object { "$_" }) -join "`n"
  $lines = $joined -split "`r?`n"
  $addr = ($lines | Where-Object { $_ -match '^(bc1p|tb1p|bcrt1p|sb1p)' } | Select-Object -Last 1)
  if ([string]::IsNullOrWhiteSpace($addr)) {
    throw "Could not find a Taproot address in wallet output: $joined"
  }

  return $addr.Trim()
}

$binDir = Join-Path $PearlRoot "bin"
$oyster = Join-Path $binDir "oyster.exe"
$prlctl = Join-Path $binDir "prlctl.exe"
$pearld = Join-Path $binDir "pearld.exe"

Require-Path $PearlRoot "Pearl root"
Require-Path $binDir "Pearl bin dir"

switch ($Stage) {
  "create-wallet" {
    Require-Path $oyster "oyster.exe"
    Write-Host "[pearl-simnet] creating wallet in simnet context"
    & $oyster -u $RpcUser -P $RpcPassword --create
    break
  }

  "wallet" {
    Require-Path $oyster "oyster.exe"
    Write-Host "[pearl-simnet] starting oyster wallet on simnet"
    & $oyster -u $RpcUser -P $RpcPassword --simnet
    break
  }

  "address" {
    Require-Path $prlctl "prlctl.exe"
    Write-Host "[pearl-simnet] requesting a new Taproot mining address from the simnet wallet"
    & $prlctl -u $RpcUser -P $RpcPassword -s https://localhost:18554 getnewaddress
    break
  }

  "node" {
    Require-MiningAddress
    Require-Path $pearld "pearld.exe"
    Write-Host "[pearl-simnet] starting pearld on simnet"
    & $pearld `
      --simnet `
      --notls `
      --rpcuser=$RpcUser `
      --rpcpass=$RpcPassword `
      --rpclisten=127.0.0.1:18556 `
      --miningaddr=$MiningAddress `
      --txindex `
      --debuglevel=debug
    break
  }

  "gateway" {
    Require-MiningAddress
    $env:PEARLD_RPC_URL = "http://127.0.0.1:18556"
    $env:PEARLD_RPC_USER = $RpcUser
    $env:PEARLD_RPC_PASSWORD = $RpcPassword
    $env:PEARLD_MINING_ADDRESS = $MiningAddress
    $env:MINER_RPC_TRANSPORT = "tcp"
    $env:MINER_RPC_HOST = "127.0.0.1"
    $env:MINER_RPC_PORT = "8337"
    Write-Host "[pearl-simnet] starting pearl-gateway on tcp://127.0.0.1:8337"
    pearl-gateway start -v
    break
  }

  "all" {
    Write-Host "[pearl-simnet] launching wallet in a new window"
    Start-StageWindow -TargetStage "wallet"

    Write-Host "[pearl-simnet] waiting $WalletStartupSeconds seconds for wallet startup"
    Start-Sleep -Seconds $WalletStartupSeconds

    if ([string]::IsNullOrWhiteSpace($MiningAddress)) {
      Write-Host "[pearl-simnet] fetching a Taproot mining address from the running wallet"
      $MiningAddress = Get-MiningAddressFromWallet
    }

    Write-Host "[pearl-simnet] mining address: $MiningAddress"
    Write-Host "[pearl-simnet] launching pearld in a new window"
    Start-StageWindow -TargetStage "node" -StageMiningAddress $MiningAddress

    Write-Host "[pearl-simnet] launching pearl-gateway in a new window"
    Start-StageWindow -TargetStage "gateway" -StageMiningAddress $MiningAddress

    Write-Host "[pearl-simnet] simnet stack launched"
    Write-Host "[pearl-simnet] wallet:  https://localhost:18554"
    Write-Host "[pearl-simnet] pearld:  http://127.0.0.1:18556"
    Write-Host "[pearl-simnet] gateway: tcp://127.0.0.1:8337"
    break
  }
}
