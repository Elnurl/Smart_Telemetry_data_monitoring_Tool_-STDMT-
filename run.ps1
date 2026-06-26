Set-Location $PSScriptRoot
$py310 = "$env:LOCALAPPDATA\Programs\Python\Python310\python.exe"
if (Test-Path $py310) {
    & $py310 main.py @args
    exit $LASTEXITCODE
}
Write-Error "STDMS requires Python 3.10. Not found: $py310"
Write-Host "Try: py -3.10 main.py"
exit 1
