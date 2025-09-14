@echo off
title STDMS Deployment System
echo ========================================
echo  STDMS - Satellite Telemetry Data Management System  
echo  Windows Deployment Package Creator
echo ========================================
echo.

:: Check if executable exists
echo [1/6] Checking build status...
if not exist "dist\STDMS.exe" (
    echo ERROR: Executable not found at dist\STDMS.exe
    echo Please run build.bat first to create the executable
    pause
    exit /b 1
)
echo ✓ Executable found

:: Get executable information
for %%A in ("dist\STDMS.exe") do (
    set size=%%~zA
    set /a sizeMB=!size!/1048576
)
setlocal enabledelayedexpansion
echo   File: dist\STDMS.exe
echo   Size: !sizeMB! MB

:: Create deployment directory
echo.
echo [2/6] Creating deployment directory...
set DEPLOY_DIR=deploy_STDMS_v1.0.0
if exist %DEPLOY_DIR% rmdir /s /q %DEPLOY_DIR%
mkdir %DEPLOY_DIR%
echo ✓ Deployment directory created: %DEPLOY_DIR%

:: Copy main files
echo.
echo [3/6] Copying application files...
copy "dist\STDMS.exe" "%DEPLOY_DIR%\" >nul
copy "README.md" "%DEPLOY_DIR%\README.txt" >nul
if exist "LICENSE" copy "LICENSE" "%DEPLOY_DIR%\LICENSE.txt" >nul

:: Create directory structure for portable mode
echo ✓ Application files copied

echo.
echo [4/6] Setting up portable environment...
mkdir "%DEPLOY_DIR%\data"
mkdir "%DEPLOY_DIR%\config"
mkdir "%DEPLOY_DIR%\logs"
mkdir "%DEPLOY_DIR%\reports"
mkdir "%DEPLOY_DIR%\models"
echo ✓ Directory structure created

:: Copy configuration files if they exist
if exist "telemetry_monitor\*.json" (
    copy "telemetry_monitor\*.json" "%DEPLOY_DIR%\config\" >nul 2>&1
    echo ✓ Configuration files copied
)

if exist "telemetry_monitor\models\*" (
    xcopy "telemetry_monitor\models\*" "%DEPLOY_DIR%\models\" /E /I >nul 2>&1
    echo ✓ ML models copied
)

:: Create portable launcher
echo.
echo [5/6] Creating portable launcher...
echo @echo off > "%DEPLOY_DIR%\STDMS_Portable.bat"
echo title STDMS - Satellite Telemetry Data Management System >> "%DEPLOY_DIR%\STDMS_Portable.bat"
echo. >> "%DEPLOY_DIR%\STDMS_Portable.bat"
echo :: Set portable mode environment >> "%DEPLOY_DIR%\STDMS_Portable.bat"
echo cd /d "%%~dp0" >> "%DEPLOY_DIR%\STDMS_Portable.bat"
echo set STDMS_PORTABLE=1 >> "%DEPLOY_DIR%\STDMS_Portable.bat"
echo set STDMS_DATA_DIR=%%cd%%\data >> "%DEPLOY_DIR%\STDMS_Portable.bat"
echo set STDMS_CONFIG_DIR=%%cd%%\config >> "%DEPLOY_DIR%\STDMS_Portable.bat"
echo set STDMS_LOGS_DIR=%%cd%%\logs >> "%DEPLOY_DIR%\STDMS_Portable.bat"
echo set STDMS_REPORTS_DIR=%%cd%%\reports >> "%DEPLOY_DIR%\STDMS_Portable.bat"
echo. >> "%DEPLOY_DIR%\STDMS_Portable.bat"
echo :: Create directories if they don't exist >> "%DEPLOY_DIR%\STDMS_Portable.bat"
echo if not exist data mkdir data >> "%DEPLOY_DIR%\STDMS_Portable.bat"
echo if not exist config mkdir config >> "%DEPLOY_DIR%\STDMS_Portable.bat"
echo if not exist logs mkdir logs >> "%DEPLOY_DIR%\STDMS_Portable.bat"
echo if not exist reports mkdir reports >> "%DEPLOY_DIR%\STDMS_Portable.bat"
echo if not exist models mkdir models >> "%DEPLOY_DIR%\STDMS_Portable.bat"
echo. >> "%DEPLOY_DIR%\STDMS_Portable.bat"
echo echo Starting STDMS - Satellite Telemetry Data Management System... >> "%DEPLOY_DIR%\STDMS_Portable.bat"
echo echo. >> "%DEPLOY_DIR%\STDMS_Portable.bat"
echo STDMS.exe >> "%DEPLOY_DIR%\STDMS_Portable.bat"
echo. >> "%DEPLOY_DIR%\STDMS_Portable.bat"
echo :: Handle exit codes >> "%DEPLOY_DIR%\STDMS_Portable.bat"
echo if %%errorlevel%% neq 0 ( >> "%DEPLOY_DIR%\STDMS_Portable.bat"
echo     echo. >> "%DEPLOY_DIR%\STDMS_Portable.bat"
echo     echo Application exited with error code: %%errorlevel%% >> "%DEPLOY_DIR%\STDMS_Portable.bat"
echo     echo Check logs\stdms.log for detailed error information. >> "%DEPLOY_DIR%\STDMS_Portable.bat"
echo     echo. >> "%DEPLOY_DIR%\STDMS_Portable.bat"
echo     pause >> "%DEPLOY_DIR%\STDMS_Portable.bat"
echo ^) >> "%DEPLOY_DIR%\STDMS_Portable.bat"

:: Create system installer launcher
echo @echo off > "%DEPLOY_DIR%\STDMS_Install.bat"
echo title STDMS - System Installation >> "%DEPLOY_DIR%\STDMS_Install.bat"
echo echo Installing STDMS to system... >> "%DEPLOY_DIR%\STDMS_Install.bat"
echo echo This will copy STDMS to Program Files and create shortcuts. >> "%DEPLOY_DIR%\STDMS_Install.bat"
echo echo. >> "%DEPLOY_DIR%\STDMS_Install.bat"
echo pause >> "%DEPLOY_DIR%\STDMS_Install.bat"
echo. >> "%DEPLOY_DIR%\STDMS_Install.bat"
echo :: Request administrator privileges >> "%DEPLOY_DIR%\STDMS_Install.bat"
echo echo Requesting administrator privileges... >> "%DEPLOY_DIR%\STDMS_Install.bat"
echo powershell -Command "Start-Process cmd -ArgumentList '/c xcopy \"%%~dp0*\" \"%%ProgramFiles%%\STDMS\\" /E /I /Y ^&^& pause' -Verb RunAs" >> "%DEPLOY_DIR%\STDMS_Install.bat"

echo ✓ Launcher scripts created

:: Create ZIP archive
echo.
echo [6/6] Creating ZIP archive...
powershell -command "Compress-Archive -Path '%DEPLOY_DIR%\*' -DestinationPath 'STDMS_v1.0.0_Portable.zip' -Force"
if %errorlevel% equ 0 (
    echo ✓ ZIP archive created: STDMS_v1.0.0_Portable.zip
) else (
    echo WARNING: Failed to create ZIP archive
    echo Manual zip creation may be required
)

:: Calculate deployment size
for /f %%A in ('powershell -command "(Get-ChildItem '%DEPLOY_DIR%' -Recurse | Measure-Object -Property Length -Sum).Sum"') do set totalSize=%%A
set /a totalSizeMB=!totalSize!/1048576

:: Show deployment summary
echo.
echo ========================================
echo 🎉 DEPLOYMENT COMPLETED SUCCESSFULLY! 🎉
echo ========================================
echo.
echo Deployment Package: %DEPLOY_DIR%\
echo Package Size: !totalSizeMB! MB
echo.
echo Portable ZIP: STDMS_v1.0.0_Portable.zip
echo.
echo Launch Options:
echo   1. Run %DEPLOY_DIR%\STDMS_Portable.bat (Portable Mode)
echo   2. Run %DEPLOY_DIR%\STDMS.exe (Direct Launch)
echo   3. Run %DEPLOY_DIR%\STDMS_Install.bat (System Install)
echo.
echo Distribution Files:
echo   - STDMS.exe (Main Application)
echo   - README.txt (Documentation)
echo   - Data directories (data, config, logs, reports)
echo   - Portable launcher script
echo   - System installer script
echo.
echo The deployment is ready for distribution!
echo ========================================

echo.
echo Press any key to exit...
pause >nul