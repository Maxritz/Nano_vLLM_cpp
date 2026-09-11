$ErrorActionPreference = "Continue"
$env:PATH = "G:\ROCM10RT-gfx1201\bin;" + $env:PATH

for ($i = 1; $i -le 30; $i++) {
    Write-Host "=== autofix iteration $i ==="
    cmake --build build --target nanovllm_golden nanovllm_test 2>&1 | Select-String -Pattern "error|Linking HIP exec"
    if ($LASTEXITCODE -ne 0) { Write-Host "BUILD FAILED"; exit 1 }
    & .\build\nanovllm_test.exe
    & .\build\nanovllm_golden.exe
    if ($LASTEXITCODE -eq 0) { Write-Host "ALL GREEN"; exit 0 }
    if (Test-Path .\scripts\autofix_done.flag) { Remove-Item .\scripts\autofix_done.flag; continue }
    Write-Host "HALT: fix the failure above, then rerun this script"
    exit 1
}