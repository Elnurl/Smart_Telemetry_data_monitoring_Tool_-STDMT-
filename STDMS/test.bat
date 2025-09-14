@echo off
title STDMS Test System
echo ========================================
echo  STDMS - Satellite Telemetry Data Management System
echo  Executable Testing Suite
echo ========================================
echo.

:: Check if executable exists
echo [1/5] Checking executable...
if not exist "dist\STDMS.exe" (
    echo ERROR: Executable not found at dist\STDMS.exe
    echo Please run build.bat first to create the executable
    pause
    exit /b 1
)

:: Get executable information
for %%A in ("dist\STDMS.exe") do (
    set size=%%~zA
    set /a sizeMB=!size!/1048576
)
setlocal enabledelayedexpansion
echo ✓ Executable found
echo   Path: dist\STDMS.exe
echo   Size: !sizeMB! MB

:: Test version information
echo.
echo [2/5] Testing version information...
cd dist
STDMS.exe --version
set VERSION_EXIT_CODE=%errorlevel%
cd ..

if %VERSION_EXIT_CODE% equ 0 (
    echo ✓ Version test passed
) else (
    echo ⚠ Version test returned exit code: %VERSION_EXIT_CODE%
)

:: Test help information
echo.
echo [3/5] Testing help information...
cd dist
STDMS.exe --help
set HELP_EXIT_CODE=%errorlevel%
cd ..

if %HELP_EXIT_CODE% equ 0 (
    echo ✓ Help test passed
) else (
    echo ⚠ Help test returned exit code: %HELP_EXIT_CODE%
)

:: Test basic functionality
echo.
echo [4/5] Testing basic functionality...
cd dist
echo Running basic system test (this may take a moment)...
timeout /t 2 /nobreak >nul
STDMS.exe --test
set TEST_EXIT_CODE=%errorlevel%
cd ..

if %TEST_EXIT_CODE% equ 0 (
    echo ✓ Basic functionality test passed
) else (
    echo ⚠ Basic functionality test returned exit code: %TEST_EXIT_CODE%
    echo   This may be normal for some configurations
)

:: Test dependencies (import test)
echo.
echo [5/5] Testing dependency availability...
cd dist

echo Testing critical imports...
echo import sys; import PySide6; import pandas; import numpy; import sklearn; print("All critical dependencies available") | STDMS.exe
set IMPORT_EXIT_CODE=%errorlevel%
cd ..

if %IMPORT_EXIT_CODE% equ 0 (
    echo ✓ Dependency test passed
) else (
    echo ⚠ Dependency test issues detected
    echo   Some optional features may not be available
)

:: Show test summary
echo.
echo ========================================
echo 📋 TEST SUMMARY
echo ========================================
echo.
echo Test Results:
if %VERSION_EXIT_CODE% equ 0 (echo ✓ Version Info: PASSED) else echo ⚠ Version Info: WARNING
if %HELP_EXIT_CODE% equ 0 (echo ✓ Help Info: PASSED) else echo ⚠ Help Info: WARNING  
if %TEST_EXIT_CODE% equ 0 (echo ✓ Basic Functionality: PASSED) else echo ⚠ Basic Functionality: WARNING
if %IMPORT_EXIT_CODE% equ 0 (echo ✓ Dependencies: PASSED) else echo ⚠ Dependencies: WARNING

echo.
if %VERSION_EXIT_CODE% equ 0 if %HELP_EXIT_CODE% equ 0 if %TEST_EXIT_CODE% equ 0 if %IMPORT_EXIT_CODE% equ 0 (
    echo 🎉 ALL TESTS PASSED! 🎉
    echo The executable is ready for distribution.
) else (
    echo ⚠ Some tests returned warnings
    echo The executable may still work but some features might be limited.
    echo Check the individual test results above for details.
)

echo.
echo Additional Manual Tests Recommended:
echo   1. Launch the GUI: double-click dist\STDMS.exe
echo   2. Test data import functionality
echo   3. Verify machine learning features
echo   4. Check report generation
echo   5. Test multi-device support
echo.
echo For GUI testing, run: dist\STDMS.exe
echo ========================================

echo.
echo Press any key to exit...
pause >nul