# 文档智能助手 - 环境安装脚本（Windows）
# 右键 → "使用 PowerShell 运行"

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  文档智能助手 - 环境安装" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 1. 检查 uv
Write-Host "[1/3] 检查 uv 包管理器..." -ForegroundColor Yellow
$uv = Get-Command uv -ErrorAction SilentlyContinue
if (-not $uv) {
    Write-Host "  未找到 uv，正在安装..." -ForegroundColor Yellow
    irm https://astral.sh/uv/install.ps1 | iex
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}
Write-Host "  uv 就绪 ✓" -ForegroundColor Green

# 2. 安装 Python 依赖
Write-Host ""
Write-Host "[2/3] 安装 Python 依赖..." -ForegroundColor Yellow
uv sync
Write-Host "  依赖安装完成 ✓" -ForegroundColor Green

# 3. 验证关键依赖
Write-Host ""
Write-Host "[3/3] 验证关键依赖..." -ForegroundColor Yellow
$checkResult = uv run python -c "import onnxruntime; print(onnxruntime.__version__)" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "  ✗ onnxruntime 加载失败！" -ForegroundColor Red
    Write-Host ""
    Write-Host "  原因：缺少 Visual C++ 运行库（MSVCP140.dll / VCRUNTIME140.dll）" -ForegroundColor Red
    Write-Host ""
    Write-Host "  请下载安装 VC++ Redistributable：" -ForegroundColor Yellow
    Write-Host "  https://aka.ms/vs/17/release/vc_redist.x64.exe" -ForegroundColor White
    Write-Host ""
    Write-Host "  安装后重新运行此脚本即可。" -ForegroundColor Yellow
    Write-Host ""
    Read-Host "按 Enter 退出"
    exit 1
}
Write-Host "  onnxruntime 就绪 ✓ (v$checkResult)" -ForegroundColor Green

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  安装完成！" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "下一步：" -ForegroundColor White
Write-Host "  1. 复制 .env.example 为 .env 并填入 API Key" -ForegroundColor White
Write-Host "  2. uv run streamlit run app_streamlit.py" -ForegroundColor White
Write-Host ""
Read-Host "按 Enter 退出"
