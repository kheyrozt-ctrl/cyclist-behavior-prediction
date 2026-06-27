param(
    [int]$WebcamIndex = 0,
    [int]$Frames = 450,
    [int]$Port = 8890,
    [int]$MediaPipeComplexity = 1
)

$ErrorActionPreference = "Stop"
if ($Frames -lt 1) {
    throw "Frames must be at least 1."
}
if ($Port -lt 1 -or $Port -gt 65535) {
    throw "Port must be between 1 and 65535."
}
if ($MediaPipeComplexity -notin 0, 1, 2) {
    throw "MediaPipeComplexity must be 0, 1, or 2."
}
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$ReleaseDir = Join-Path $RepoRoot "release"
$Docker = Get-Command docker -ErrorAction SilentlyContinue
if (-not $Docker) {
    $candidate = "C:\Program Files\Docker\Docker\resources\bin\docker.exe"
    if (-not (Test-Path -LiteralPath $candidate)) {
        throw "Docker CLI was not found. Start Docker Desktop and retry."
    }
    $Docker = Get-Item -LiteralPath $candidate
}
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Missing $Python. Create the project virtual environment first."
}

New-Item -ItemType Directory -Force $ReleaseDir | Out-Null
$image = "cyclist-holoscan:3.11"
& $Docker.FullName image inspect $image *> $null
if ($LASTEXITCODE -ne 0) {
    & $Docker.FullName build `
        -t $image `
        -f (Join-Path $PSScriptRoot "Dockerfile") `
        $PSScriptRoot
    if ($LASTEXITCODE -ne 0) {
        throw "Holoscan image build failed."
    }
}

$bridgeOut = Join-Path $ReleaseDir "camera-bridge.log"
$bridgeErr = Join-Path $ReleaseDir "camera-bridge-error.log"
$bridge = Start-Process `
    -FilePath $Python `
    -ArgumentList @(
        (Join-Path $PSScriptRoot "camera_bridge.py"),
        "--host", "127.0.0.1",
        "--port", $Port,
        "--webcam-index", $WebcamIndex,
        "--width", "640",
        "--height", "480",
        "--fps", "30"
    ) `
    -WorkingDirectory $RepoRoot `
    -WindowStyle Hidden `
    -RedirectStandardOutput $bridgeOut `
    -RedirectStandardError $bridgeErr `
    -PassThru

try {
    $ready = $false
    for ($attempt = 0; $attempt -lt 20; $attempt++) {
        Start-Sleep -Milliseconds 500
        if ($bridge.HasExited) {
            break
        }
        try {
            $response = Invoke-WebRequest `
                -UseBasicParsing `
                "http://127.0.0.1:$Port/health" `
                -TimeoutSec 2
            if ($response.StatusCode -eq 200) {
                $ready = $true
                break
            }
        }
        catch {
        }
    }
    if (-not $ready) {
        $details = Get-Content $bridgeErr -Raw -ErrorAction SilentlyContinue
        throw "Camera bridge failed to start. $details"
    }

    $duration = [string]::Format(
        [Globalization.CultureInfo]::InvariantCulture,
        "{0:R}",
        $Frames / 30.0
    )
    $jsonl = "release/holoscan-live-validation.jsonl"
    $console = "release/holoscan-live-validation-console.log"
    $inner = @(
        "python3 deployment/holoscan/app.py",
        "--model bus",
        "--runtime onnx",
        "--camera stream",
        "--camera-url http://host.docker.internal:$Port/stream.mjpg",
        "--pose mediapipe",
        "--mediapipe-complexity $MediaPipeComplexity",
        "--duration $duration",
        "--fps 30",
        "--headless",
        "--output-jsonl $jsonl",
        "> $console 2>&1"
    ) -join " "

    & $Docker.FullName run --rm --gpus all --ipc=host `
        --ulimit memlock=-1 --ulimit stack=67108864 `
        -v "${RepoRoot}:/workspace" `
        -w /workspace `
        $image bash -lc $inner
    if ($LASTEXITCODE -ne 0) {
        throw "Holoscan live validation failed. See $console."
    }

    & $Python (Join-Path $RepoRoot "tools\summarize_holoscan_run.py") `
        (Join-Path $RepoRoot $jsonl) `
        --output (Join-Path $ReleaseDir "holoscan-live-validation-summary.json")
}
finally {
    if ($bridge -and -not $bridge.HasExited) {
        Stop-Process -Id $bridge.Id
        Wait-Process -Id $bridge.Id -Timeout 5 -ErrorAction SilentlyContinue
    }
}
