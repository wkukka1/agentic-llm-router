# Installs the routers for the bake-off (python -m evaluation.gateway.router_compare) into
# ~/.aar-router-bench/, each in its own folder and environment. Nothing global, nothing in
# ~/.claude. Versions are pinned to the ones the published comparison ran on (2026-10-20).
#
#   powershell -File scripts/router_bench/install.ps1
#
# Needs node >= 22, git, uv, and Ollama (for the LiteLLM auto-router's embeddings).

$ErrorActionPreference = "Stop"
$bench = Join-Path $HOME ".aar-router-bench"
New-Item -ItemType Directory -Force $bench | Out-Null

# claude-code-router: 1.0.73 is the last release with the fixed scenario rules
# (default / background / think / longContext). v3 routes by user-written scripts instead.
$ccr = Join-Path $bench "ccr-v1"
New-Item -ItemType Directory -Force $ccr | Out-Null
Push-Location $ccr
npm init -y | Out-Null
npm install "@musistudio/claude-code-router@1.0.73" --save-exact --no-fund --no-audit
Pop-Location

# claude-code-proxy (1rgs): a pinned commit, its own uv environment
$ccp = Join-Path $bench "ccp"
if (-not (Test-Path $ccp)) { git clone --quiet https://github.com/1rgs/claude-code-proxy.git $ccp }
git -C $ccp checkout --quiet 5e45ba683ded931c1832cfca6468a791c6855e45
Push-Location $ccp
uv sync
Pop-Location

# LiteLLM proxy with its declared semantic-router extra (<1.0; 1.x breaks its auto-router)
$litellm = Join-Path $bench "litellm"
New-Item -ItemType Directory -Force $litellm | Out-Null
uv venv --python 3.12 (Join-Path $litellm ".venv") -q
uv pip install --python (Join-Path $litellm ".venv\Scripts\python.exe") "litellm[proxy,semantic-router]==1.106.1"

# RouteLLM: old dependency pins, so a separate Python 3.11 environment
$routellm = Join-Path $bench "routellm"
New-Item -ItemType Directory -Force $routellm | Out-Null
uv venv --python 3.11 (Join-Path $routellm ".venv") -q
uv pip install --python (Join-Path $routellm ".venv\Scripts\python.exe") "routellm[serve,eval]==0.2.0"

ollama pull nomic-embed-text
Write-Host "installed into $bench"
