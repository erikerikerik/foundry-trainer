# Launch Claude Code against the local Strata server (dev-only helper).
# Settings apply to this PowerShell process only; other Claude sessions are unaffected.
# Usage (from the repo root):  .\tools\dev\claude-strata.ps1

$StrataUrl = "http://127.0.0.1:8080"

# 1. Strata must be up, and we need its model name.
try {
    $health = Invoke-RestMethod -Uri "$StrataUrl/health" -TimeoutSec 5
} catch {
    Write-Error "Strata is not answering at $StrataUrl. Start Strata first."
    exit 1
}
if ($health.service -ne "strata") {
    Write-Error "Something answered at $StrataUrl but it is not Strata (service='$($health.service)')."
    exit 1
}
$model = $health.model
if (-not $health.loaded) {
    Write-Warning "Strata is up but its model is not loaded; the first request may be slow or fail."
}

# 2. Claude Code needs the claude CLI.
if (-not (Get-Command claude -ErrorAction SilentlyContinue)) {
    Write-Error "The 'claude' command was not found. Install Claude Code first."
    exit 1
}

# 3. Point this process at Strata. Strata has no API key; Claude Code only needs a non-empty value.
$env:ANTHROPIC_BASE_URL = $StrataUrl
$env:ANTHROPIC_AUTH_TOKEN = "local-strata-no-key"
$env:ANTHROPIC_MODEL = $model
# Background/small-model calls must also use Strata's only model, not a Haiku name it doesn't have.
$env:ANTHROPIC_SMALL_FAST_MODEL = $model
$env:ANTHROPIC_DEFAULT_HAIKU_MODEL = $model
$env:ANTHROPIC_DEFAULT_SONNET_MODEL = $model
$env:ANTHROPIC_DEFAULT_OPUS_MODEL = $model
$env:CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC = "1"
# Claude Code does not know Strata's model and assumes a 200k window. Compact before Strata's real limit
# (leave ~30k for the system prompt and output). Minimum accepted value is 100000.
$window = [math]::Max(100000, [int]$health.max_context - 30000)
$env:CLAUDE_CODE_AUTO_COMPACT_WINDOW = "$window"

Write-Host "Claude Code -> Strata ($model) at $StrataUrl"
# Ask before acting. Auto mode needs a separate safety-check model that Strata does not provide.
claude --permission-mode default @args
