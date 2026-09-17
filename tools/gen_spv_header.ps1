# Regenerates include/nanovllm/spv.h from compiled .spv files.
# Usage: gen_spv_header.ps1 <spv_dir> <out_header>
param([string]$SpvDir, [string]$Out)

$files = Get-ChildItem -Path $SpvDir -Filter *.spv | Sort-Object Name
$nl = [Environment]::NewLine
$b = New-Object System.Text.StringBuilder
$b.Append("#pragma once") | Out-Null
$b.Append($nl) | Out-Null
$b.Append("#include <cstddef>") | Out-Null
$b.Append($nl) | Out-Null
$b.Append("/* Auto-generated: embedded SPIR-V for the Vulkan backend. */") | Out-Null
$b.Append($nl) | Out-Null

foreach ($f in $files) {
    $name = $f.BaseName
    $data = [System.IO.File]::ReadAllBytes($f.FullName)
    $n = [int]($data.Length / 4)
    $b.Append("static const unsigned int ") | Out-Null
    $b.Append($name) | Out-Null
    $b.Append("_spv[] = {") | Out-Null
    $b.Append($nl) | Out-Null
    for ($i = 0; $i -lt $n; $i++) {
        $w = [BitConverter]::ToUInt32($data, $i * 4)
        if (($i % 8) -eq 0) { $b.Append("  ") | Out-Null }
        $b.Append(("0x{0:X8}," -f $w)) | Out-Null
        if (($i % 8) -eq 7 -or $i -eq ($n - 1)) { $b.Append($nl) | Out-Null } else { $b.Append(" ") | Out-Null }
    }
    $b.Append("};") | Out-Null
    $b.Append($nl) | Out-Null
}

$b.Append("static inline int spv_count(const char* n) {") | Out-Null
$b.Append($nl) | Out-Null
foreach ($f in $files) {
    $name = $f.BaseName
    $data = [System.IO.File]::ReadAllBytes($f.FullName)
    $cnt = $data.Length / 4
    $b.Append("  if (strcmp(n, """) | Out-Null
    $b.Append($name) | Out-Null
    $b.Append(""")==0) return ") | Out-Null
    $b.Append($cnt.ToString()) | Out-Null
    $b.Append(";") | Out-Null
    $b.Append($nl) | Out-Null
}
$b.Append("  return 0;") | Out-Null
$b.Append($nl) | Out-Null
$b.Append("}") | Out-Null
$b.Append($nl) | Out-Null
$b.Append("static const unsigned int* spv_for(const char* n) {") | Out-Null
$b.Append($nl) | Out-Null
$b.Append("  if (!n) return 0;") | Out-Null
$b.Append($nl) | Out-Null
foreach ($f in $files) {
    $name = $f.BaseName
    $b.Append("  if (strcmp(n, """) | Out-Null
    $b.Append($name) | Out-Null
    $b.Append(""")==0) return ") | Out-Null
    $b.Append($name) | Out-Null
    $b.Append("_spv;") | Out-Null
    $b.Append($nl) | Out-Null
}
$b.Append("  return 0;") | Out-Null
$b.Append($nl) | Out-Null
$b.Append("}") | Out-Null
$b.Append($nl) | Out-Null

[System.IO.File]::WriteAllText($Out, $b.ToString())
Write-Host "wrote $Out ($($files.Count) shaders)"