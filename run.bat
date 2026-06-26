@echo off
setlocal
cd /d "%~dp0"
set PY310=%LocalAppData%\Programs\Python\Python310\python.exe
if exist "%PY310%" (
    "%PY310%" main.py %*
    exit /b %ERRORLEVEL%
)
echo STDMS requires Python 3.10. Not found: %PY310%
echo Install Python 3.10 or run: py -3.10 main.py
exit /b 1
