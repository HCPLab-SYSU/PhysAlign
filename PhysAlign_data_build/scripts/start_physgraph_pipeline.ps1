param(
    [string]$Config = "",
    [string]$Workspace = "",
    [int]$Port = 8770,
    [switch]$OpenBrowser
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot

$pythonExe = ""
$pythonPrefix = @()
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (Test-Path -LiteralPath $venvPython -PathType Leaf) {
    $pythonExe = $venvPython
}
else {
    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCommand) {
        $pythonExe = $pythonCommand.Source
    }
    else {
        $launcher = Get-Command py -ErrorAction SilentlyContinue
        if ($launcher) {
            $pythonExe = $launcher.Source
            $pythonPrefix = @("-3")
        }
    }
}

if (-not $pythonExe) {
    throw "Python 3.11+ was not found. Install Python and run: python -m venv .venv"
}

$targetArgs = @()
if ($Config) {
    if (-not (Test-Path -LiteralPath $Config -PathType Leaf)) {
        throw "Config file does not exist: $Config"
    }
    $targetArgs = @("--config", (Resolve-Path -LiteralPath $Config).Path)
}
elseif ($Workspace) {
    if (-not (Test-Path -LiteralPath $Workspace -PathType Container)) {
        throw "Workspace does not exist: $Workspace"
    }
    $targetArgs = @("--workspace", (Resolve-Path -LiteralPath $Workspace).Path)
}
else {
    throw "Specify a target with -Config <file> or -Workspace <directory>."
}

& $pythonExe @pythonPrefix -X utf8 -m physgraph_pipeline doctor @targetArgs
if ($LASTEXITCODE -ne 0) {
    throw "Preflight checks failed. Resolve the doctor errors shown above."
}

$serveArgs = @("-X", "utf8", "-m", "physgraph_pipeline", "serve") + $targetArgs + @("--port", "$Port")
if ($OpenBrowser) {
    $serveArgs += "--open-browser"
}
& $pythonExe @pythonPrefix @serveArgs
exit $LASTEXITCODE
