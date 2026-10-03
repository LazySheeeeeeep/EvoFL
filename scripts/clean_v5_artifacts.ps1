param([switch]$Execute)
$ErrorActionPreference = 'Stop'
$workspace = [System.IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot)).TrimEnd('\')
$allowed = @((Join-Path $workspace 'runs'), (Join-Path $workspace 'skill_pools'))
$files = [System.Collections.Generic.List[string]]::new()
$directories = [System.Collections.Generic.List[string]]::new()
$links = [System.Collections.Generic.List[string]]::new()
$stack = [System.Collections.Generic.Stack[string]]::new()
foreach ($root in $allowed) { if (Test-Path -LiteralPath $root) { $stack.Push($root) } }
while ($stack.Count -gt 0) {
    $path = [System.IO.Path]::GetFullPath($stack.Pop())
    if (-not $path.StartsWith($workspace + '\', [System.StringComparison]::OrdinalIgnoreCase)) { throw "Outside workspace: $path" }
    $item = Get-Item -LiteralPath $path -Force
    if ($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) { throw "Refusing to follow reparse point: $path" }
    foreach ($child in Get-ChildItem -LiteralPath $path -Force) {
        $resolved = [System.IO.Path]::GetFullPath($child.FullName)
        if (-not ($allowed | Where-Object { $resolved.StartsWith($_ + '\', [System.StringComparison]::OrdinalIgnoreCase) })) { throw "Outside cleanup roots: $resolved" }
        if ($child.Attributes -band [System.IO.FileAttributes]::ReparsePoint) { $links.Add($resolved); continue }
        if ($child.PSIsContainer) { $directories.Add($resolved); $stack.Push($resolved) }
        elseif ($child.Name -ne '.gitkeep') { $files.Add($resolved) }
    }
}
Write-Output "Verified roots: $($allowed -join ', ')"
Write-Output "Files to delete: $($files.Count); links to unlink: $($links.Count); inspected directories: $($directories.Count); execute: $Execute"
if ($Execute) {
    foreach ($path in $links) {
        # Literal non-recursive removal unlinks the entry, never its target.
        Remove-Item -LiteralPath $path -Force
    }
    foreach ($path in $files) {
        $attributes = [System.IO.File]::GetAttributes($path)
        if ($attributes -band [System.IO.FileAttributes]::ReparsePoint) { throw "Target changed to reparse point: $path" }
        if ($attributes -band [System.IO.FileAttributes]::ReadOnly) { [System.IO.File]::SetAttributes($path, $attributes -band (-bnot [System.IO.FileAttributes]::ReadOnly)) }
        [System.IO.File]::Delete($path)
    }
    # Remove only empty directories, without recursive shell deletion.
    foreach ($path in ($directories | Sort-Object Length -Descending)) {
        if (-not [System.IO.Directory]::EnumerateFileSystemEntries($path).GetEnumerator().MoveNext()) { [System.IO.Directory]::Delete($path, $false) }
    }
    Write-Output 'Cleanup completed; workspace source/config/prompt/test/Git and Docker untouched.'
}
