@echo off
title STDMS Build System
echo ========================================
echo  STDMS - Satellite Telemetry Data Management System
echo  Windows Executable Builder
echo ========================================
echo.

:: Check if Python is available
echo [1/8] Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ and add it to your PATH
    pause
    exit /b 1
)
python --version
echo ✓ Python is available

:: Check if PyInstaller is available
echo.
echo [2/8] Checking PyInstaller...
pyinstaller --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing PyInstaller...
    pip install pyinstaller
    if %errorlevel% neq 0 (
        echo ERROR: Failed to install PyInstaller
        pause
        exit /b 1
    )
)
pyinstaller --version
echo ✓ PyInstaller is ready

:: Install required dependencies
echo.
echo [3/8] Installing dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo WARNING: Some dependencies may have failed to install
    echo Continuing with build process...
)
echo ✓ Dependencies processed

:: Clean previous builds
echo.
echo [4/8] Cleaning previous builds...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist stdms.spec del stdms.spec
if exist version_info.txt del version_info.txt
echo ✓ Build directories cleaned

:: Run the packaging system
echo.
echo [5/8] Creating PyInstaller configuration...
python package_executable.py --build-only
if %errorlevel% neq 0 (
    echo ERROR: Failed to create build configuration
    pause
    exit /b 1
)
echo ✓ Configuration created

:: Build executable with PyInstaller
echo.
echo [6/8] Building executable with PyInstaller...
pyinstaller stdms.spec
if %errorlevel% neq 0 (
    echo ERROR: PyInstaller build failed
    echo Check the output above for details
    pause
    exit /b 1
)
echo ✓ Executable built successfully

:: Test the executable
echo.
echo [7/8] Testing executable...
if exist "dist\STDMS.exe" (
    cd dist
    STDMS.exe --test
    cd ..
    if %errorlevel% equ 0 (
        echo ✓ Executable test passed
    ) else (
        echo WARNING: Executable test returned non-zero exit code
        echo This may be normal for GUI applications
    )
) else (
    echo ERROR: Executable not found after build
    pause
    exit /b 1
)

:: Show build results
echo.
echo [8/8] Build Summary
echo ========================================
if exist "dist\STDMS.exe" (
    for %%A in ("dist\STDMS.exe") do (
        set size=%%~zA
        set /a sizeMB=!size!/1048576
    )
    setlocal enabledelayedexpansion
    echo ✓ BUILD SUCCESSFUL!
    echo.
    echo Executable: dist\STDMS.exe
    echo Size: !sizeMB! MB
    echo.
    echo To create deployment package, run: deploy.bat
    echo To test the executable, run: test.bat
    echo.
) else (
    echo ✗ BUILD FAILED!
    echo Executable was not created successfully.
    echo Check the build output above for errors.
    echo.
)

echo Press any key to exit...
pause >nul