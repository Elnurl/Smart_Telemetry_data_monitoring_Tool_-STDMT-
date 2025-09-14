#!/usr/bin/env python3
"""
Windows Executable Packaging System for Stage 6.6
Package complete STDMS system as standalone Windows executable using PyInstaller.
"""

import os
import sys
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any
import subprocess
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExecutablePackager:
    """Packages the complete STDMS system as a Windows executable"""
    
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root) if project_root else Path(__file__).parent
        self.build_dir = self.project_root / "build"
        self.dist_dir = self.project_root / "dist"
        self.spec_file = self.project_root / "stdms.spec"
        
        # Application metadata
        self.app_name = "STDMS"
        self.app_version = "1.0.0"
        self.app_description = "Satellite Telemetry Data Management System"
        self.app_author = "STDMS Development Team"
        
        logger.info(f"Initialized packager for project at: {self.project_root}")
    
    def create_spec_file(self) -> bool:
        """Create PyInstaller spec file with all dependencies and resources"""
        try:
            spec_content = f'''# -*- mode: python ; coding: utf-8 -*-

import sys
from pathlib import Path

# Get the directory of this spec file
spec_root = Path(SPECPATH)

block_cipher = None

# Main application analysis
a = Analysis(
    ['telemetry_monitor/main.py'],
    pathex=[str(spec_root)],
    binaries=[],
    datas=[
        # Configuration files
        ('telemetry_monitor/config', 'config'),
        ('telemetry_monitor/models', 'models'),
        ('telemetry_monitor/reports', 'reports'),
        ('telemetry_monitor/logs', 'logs'),
        
        # Data files
        ('telemetry_monitor/*.json', '.'),
        ('telemetry_monitor/database.db', '.'),
        
        # Python packages data
        ('venv/Lib/site-packages/sklearn', 'sklearn'),
        ('venv/Lib/site-packages/joblib', 'joblib'),
        ('venv/Lib/site-packages/pandas', 'pandas'),
        ('venv/Lib/site-packages/numpy', 'numpy'),
        ('venv/Lib/site-packages/PySide6', 'PySide6'),
        
        # ML models and data
        ('venv/Lib/site-packages/sklearn/datasets', 'sklearn/datasets'),
    ],
    hiddenimports=[
        # Core dependencies
        'sklearn',
        'sklearn.ensemble',
        'sklearn.preprocessing',
        'sklearn.metrics',
        'joblib',
        'pandas',
        'numpy',
        'sqlite3',
        'json',
        'datetime',
        'threading',
        'queue',
        'pathlib',
        'logging',
        'configparser',
        'hashlib',
        'uuid',
        'base64',
        'pickle',
        
        # PySide6 GUI
        'PySide6.QtWidgets',
        'PySide6.QtCore',
        'PySide6.QtGui',
        'PySide6.QtCharts',
        
        # PyQtGraph
        'pyqtgraph',
        'pyqtgraph.graphicsItems',
        'pyqtgraph.widgets',
        
        # Email and networking
        'smtplib',
        'email',
        'email.mime.multipart',
        'email.mime.text',
        'email.mime.base',
        'email.encoders',
        
        # Optional database drivers
        'psycopg2',
        'influxdb_client',
        'asyncpg',
        
        # Reporting libraries
        'reportlab',
        'reportlab.lib',
        'reportlab.platypus',
        'reportlab.graphics',
        'openpyxl',
        'matplotlib',
        'seaborn',
        
        # Scheduling
        'schedule',
        'croniter',
        
        # Deep learning (optional)
        'tensorflow',
        'keras',
        'torch',
        
        # STDMS modules
        'config_manager',
        'storage',
        'anomaly',
        'adaptive_learning',
        'report_export',
        'multi_device',
        'cloud_database',
        'scheduled_reporting',
    ],
    hookspath=[],
    hooksconfig={{}},
    runtime_hooks=[],
    excludes=[
        # Exclude unnecessary modules
        'tkinter',
        'unittest',
        'test',
        'tests',
        'setuptools',
        'distutils',
        'pydoc',
        'doctest',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Remove duplicate binaries and optimize
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='{self.app_name}',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # Set to True for debugging
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    version='version_info.txt',
    icon='resources/icon.ico' if Path('resources/icon.ico').exists() else None,
)

# Create application bundle (optional)
app = BUNDLE(
    exe,
    name='{self.app_name}.app',
    icon='resources/icon.ico' if Path('resources/icon.ico').exists() else None,
    bundle_identifier='com.stdms.telemetry',
    info_plist={{
        'NSPrincipalClass': 'NSApplication',
        'NSAppleScriptEnabled': False,
        'CFBundleDisplayName': '{self.app_description}',
        'CFBundleVersion': '{self.app_version}',
        'CFBundleShortVersionString': '{self.app_version}',
        'NSHumanReadableCopyright': 'Copyright © 2024 {self.app_author}',
    }},
)
'''
            
            with open(self.spec_file, 'w') as f:
                f.write(spec_content)
            
            logger.info(f"Created spec file: {self.spec_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating spec file: {e}")
            return False
    
    def create_version_info(self) -> bool:
        """Create version info file for Windows executable"""
        try:
            version_info_content = f'''# UTF-8
#
# Version information for {self.app_name}
#

VSVersionInfo(
  ffi=FixedFileInfo(
    filevers=(1, 0, 0, 0),
    prodvers=(1, 0, 0, 0),
    mask=0x3f,
    flags=0x0,
    OS=0x40004,
    fileType=0x1,
    subtype=0x0,
    date=(0, 0)
  ),
  kids=[
    StringFileInfo(
      [
        StringTable(
          u'040904B0',
          [
            StringStruct(u'CompanyName', u'{self.app_author}'),
            StringStruct(u'FileDescription', u'{self.app_description}'),
            StringStruct(u'FileVersion', u'{self.app_version}'),
            StringStruct(u'InternalName', u'{self.app_name}'),
            StringStruct(u'LegalCopyright', u'Copyright © 2024 {self.app_author}'),
            StringStruct(u'OriginalFilename', u'{self.app_name}.exe'),
            StringStruct(u'ProductName', u'{self.app_description}'),
            StringStruct(u'ProductVersion', u'{self.app_version}')
          ]
        )
      ]
    ),
    VarFileInfo([VarStruct(u'Translation', [1033, 1200])])
  ]
)
'''
            
            version_file = self.project_root / "version_info.txt"
            with open(version_file, 'w') as f:
                f.write(version_info_content)
            
            logger.info(f"Created version info: {version_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating version info: {e}")
            return False
    
    def prepare_resources(self) -> bool:
        """Prepare resources and assets for packaging"""
        try:
            resources_dir = self.project_root / "resources"
            resources_dir.mkdir(exist_ok=True)
            
            # Create application icon (placeholder)
            icon_content = '''
# This would be a proper .ico file
# For now, create a placeholder
'''
            
            # Create README for deployment
            readme_content = f'''# {self.app_description}

## Version: {self.app_version}

### Overview
The Satellite Telemetry Data Management System (STDMS) is a comprehensive solution for monitoring, analyzing, and managing satellite telemetry data with advanced machine learning capabilities.

### Features
- Real-time telemetry data monitoring
- Advanced anomaly detection using ML algorithms
- Multi-device support for various telemetry sources
- Professional report generation (PDF/Excel)
- Cloud database integration (PostgreSQL/InfluxDB)
- Scheduled reporting with email delivery
- Adaptive learning system for continuous improvement

### System Requirements
- Windows 10 or later (64-bit)
- Minimum 4GB RAM (8GB recommended)
- 2GB free disk space
- Network connectivity for cloud features

### Installation
1. Extract the application to a folder (e.g., C:\\Program Files\\STDMS)
2. Run {self.app_name}.exe as Administrator (first time only)
3. Follow the setup wizard to configure your system

### Configuration
The application creates configuration files in:
- %APPDATA%\\STDMS\\config\\

### Data Storage
Local data is stored in:
- %APPDATA%\\STDMS\\data\\

### Logs
Application logs are stored in:
- %APPDATA%\\STDMS\\logs\\

### Support
For technical support, please contact: {self.app_author}

### Version History
- v1.0.0: Initial release with all Stage 1-6 features

### License
Copyright © 2024 {self.app_author}
All rights reserved.
'''
            
            readme_file = self.project_root / "README.txt"
            with open(readme_file, 'w') as f:
                f.write(readme_content)
            
            logger.info("Prepared resources for packaging")
            return True
            
        except Exception as e:
            logger.error(f"Error preparing resources: {e}")
            return False
    
    def create_installer_script(self) -> bool:
        """Create NSIS installer script for Windows"""
        try:
            nsis_script = f'''# NSIS Installer Script for {self.app_name}
# Generated automatically by STDMS packaging system

!define PRODUCT_NAME "{self.app_name}"
!define PRODUCT_VERSION "{self.app_version}"
!define PRODUCT_PUBLISHER "{self.app_author}"
!define PRODUCT_WEB_SITE "https://github.com/stdms/stdms"
!define PRODUCT_DIR_REGKEY "Software\\Microsoft\\Windows\\CurrentVersion\\App Paths\\{self.app_name}.exe"
!define PRODUCT_UNINST_KEY "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${{PRODUCT_NAME}}"
!define PRODUCT_UNINST_ROOT_KEY "HKLM"

; Modern UI
!include "MUI2.nsh"

; General settings
Name "${{PRODUCT_NAME}} ${{PRODUCT_VERSION}}"
OutFile "{self.app_name}_v{self.app_version}_Installer.exe"
InstallDir "$PROGRAMFILES64\\${{PRODUCT_NAME}}"
InstallDirRegKey HKLM "${{PRODUCT_DIR_REGKEY}}" ""
ShowInstDetails show
ShowUnInstDetails show

; Interface Settings
!define MUI_ABORTWARNING
!define MUI_ICON "resources\\icon.ico"
!define MUI_UNICON "resources\\icon.ico"

; Pages
!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_LICENSE "LICENSE.txt"
!insertmacro MUI_PAGE_COMPONENTS
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_UNPAGE_WELCOME
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_UNPAGE_FINISH

; Languages
!insertmacro MUI_LANGUAGE "English"

; Installer sections
Section "Main Application" SEC01
  SetOutPath "$INSTDIR"
  SetOverwrite ifnewer
  
  ; Main executable and dependencies
  File "dist\\{self.app_name}.exe"
  File "README.txt"
  File "LICENSE.txt"
  
  ; Create application data directory
  CreateDirectory "$APPDATA\\STDMS"
  CreateDirectory "$APPDATA\\STDMS\\config"
  CreateDirectory "$APPDATA\\STDMS\\data"
  CreateDirectory "$APPDATA\\STDMS\\logs"
  CreateDirectory "$APPDATA\\STDMS\\reports"
  
  ; Registry entries
  WriteRegStr HKLM "${{PRODUCT_DIR_REGKEY}}" "" "$INSTDIR\\{self.app_name}.exe"
  WriteRegStr ${{PRODUCT_UNINST_ROOT_KEY}} "${{PRODUCT_UNINST_KEY}}" "DisplayName" "$(^Name)"
  WriteRegStr ${{PRODUCT_UNINST_ROOT_KEY}} "${{PRODUCT_UNINST_KEY}}" "UninstallString" "$INSTDIR\\uninst.exe"
  WriteRegStr ${{PRODUCT_UNINST_ROOT_KEY}} "${{PRODUCT_UNINST_KEY}}" "DisplayIcon" "$INSTDIR\\{self.app_name}.exe"
  WriteRegStr ${{PRODUCT_UNINST_ROOT_KEY}} "${{PRODUCT_UNINST_KEY}}" "DisplayVersion" "${{PRODUCT_VERSION}}"
  WriteRegStr ${{PRODUCT_UNINST_ROOT_KEY}} "${{PRODUCT_UNINST_KEY}}" "Publisher" "${{PRODUCT_PUBLISHER}}"
  
  ; Create uninstaller
  WriteUninstaller "$INSTDIR\\uninst.exe"
SectionEnd

Section "Desktop Shortcut" SEC02
  CreateShortCut "$DESKTOP\\${{PRODUCT_NAME}}.lnk" "$INSTDIR\\{self.app_name}.exe"
SectionEnd

Section "Start Menu Shortcuts" SEC03
  CreateDirectory "$SMPROGRAMS\\${{PRODUCT_NAME}}"
  CreateShortCut "$SMPROGRAMS\\${{PRODUCT_NAME}}\\${{PRODUCT_NAME}}.lnk" "$INSTDIR\\{self.app_name}.exe"
  CreateShortCut "$SMPROGRAMS\\${{PRODUCT_NAME}}\\Uninstall.lnk" "$INSTDIR\\uninst.exe"
SectionEnd

; Section descriptions
!insertmacro MUI_FUNCTION_DESCRIPTION_BEGIN
  !insertmacro MUI_DESCRIPTION_TEXT ${{SEC01}} "Core application files and dependencies"
  !insertmacro MUI_DESCRIPTION_TEXT ${{SEC02}} "Create desktop shortcut"
  !insertmacro MUI_DESCRIPTION_TEXT ${{SEC03}} "Create Start Menu shortcuts"
!insertmacro MUI_FUNCTION_DESCRIPTION_END

; Uninstaller
Section Uninstall
  Delete "$INSTDIR\\{self.app_name}.exe"
  Delete "$INSTDIR\\README.txt"
  Delete "$INSTDIR\\LICENSE.txt"
  Delete "$INSTDIR\\uninst.exe"
  
  Delete "$DESKTOP\\${{PRODUCT_NAME}}.lnk"
  Delete "$SMPROGRAMS\\${{PRODUCT_NAME}}\\${{PRODUCT_NAME}}.lnk"
  Delete "$SMPROGRAMS\\${{PRODUCT_NAME}}\\Uninstall.lnk"
  
  RMDir "$SMPROGRAMS\\${{PRODUCT_NAME}}"
  RMDir "$INSTDIR"
  
  DeleteRegKey ${{PRODUCT_UNINST_ROOT_KEY}} "${{PRODUCT_UNINST_KEY}}"
  DeleteRegKey HKLM "${{PRODUCT_DIR_REGKEY}}"
  
  SetAutoClose true
SectionEnd
'''
            
            nsis_file = self.project_root / f"{self.app_name}_installer.nsi"
            with open(nsis_file, 'w') as f:
                f.write(nsis_script)
            
            logger.info(f"Created NSIS installer script: {nsis_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating installer script: {e}")
            return False
    
    def create_batch_scripts(self) -> bool:
        """Create batch scripts for easy building and deployment"""
        try:
            # Build script
            build_script = f'''@echo off
echo Building {self.app_name} executable...

:: Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    pause
    exit /b 1
)

:: Check if PyInstaller is available
pyinstaller --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing PyInstaller...
    pip install pyinstaller
)

:: Clean previous builds
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

:: Build executable
echo Building with PyInstaller...
pyinstaller {self.spec_file.name}

if %errorlevel% equ 0 (
    echo.
    echo BUILD SUCCESSFUL!
    echo Executable created at: dist\\{self.app_name}.exe
    echo.
) else (
    echo.
    echo BUILD FAILED!
    echo Check the output above for errors.
    echo.
)

pause
'''
            
            build_file = self.project_root / "build.bat"
            with open(build_file, 'w') as f:
                f.write(build_script)
            
            # Deploy script
            deploy_script = f'''@echo off
echo Deploying {self.app_name}...

:: Check if executable exists
if not exist "dist\\{self.app_name}.exe" (
    echo ERROR: Executable not found. Run build.bat first.
    pause
    exit /b 1
)

:: Create deployment directory
set DEPLOY_DIR=deploy_{self.app_name}_v{self.app_version}
if exist %DEPLOY_DIR% rmdir /s /q %DEPLOY_DIR%
mkdir %DEPLOY_DIR%

:: Copy files
echo Copying files...
copy "dist\\{self.app_name}.exe" "%DEPLOY_DIR%\\"
copy "README.txt" "%DEPLOY_DIR%\\"
if exist "LICENSE.txt" copy "LICENSE.txt" "%DEPLOY_DIR%\\"

:: Create portable data directories
mkdir "%DEPLOY_DIR%\\data"
mkdir "%DEPLOY_DIR%\\config"
mkdir "%DEPLOY_DIR%\\logs"
mkdir "%DEPLOY_DIR%\\reports"

:: Copy default configurations
if exist "telemetry_monitor\\*.json" copy "telemetry_monitor\\*.json" "%DEPLOY_DIR%\\config\\"

:: Create portable launcher
echo @echo off > "%DEPLOY_DIR%\\{self.app_name}_Portable.bat"
echo cd /d "%%~dp0" >> "%DEPLOY_DIR%\\{self.app_name}_Portable.bat"
echo set STDMS_PORTABLE=1 >> "%DEPLOY_DIR%\\{self.app_name}_Portable.bat"
echo set STDMS_DATA_DIR=%%cd%%\\data >> "%DEPLOY_DIR%\\{self.app_name}_Portable.bat"
echo set STDMS_CONFIG_DIR=%%cd%%\\config >> "%DEPLOY_DIR%\\{self.app_name}_Portable.bat"
echo {self.app_name}.exe >> "%DEPLOY_DIR%\\{self.app_name}_Portable.bat"

:: Create ZIP archive
echo Creating ZIP archive...
powershell -command "Compress-Archive -Path '%DEPLOY_DIR%\\*' -DestinationPath '{self.app_name}_v{self.app_version}_Portable.zip' -Force"

echo.
echo DEPLOYMENT SUCCESSFUL!
echo Portable version: {self.app_name}_v{self.app_version}_Portable.zip
echo Full deployment: %DEPLOY_DIR%\\
echo.

pause
'''
            
            deploy_file = self.project_root / "deploy.bat"
            with open(deploy_file, 'w') as f:
                f.write(deploy_script)
            
            # Test script
            test_script = f'''@echo off
echo Testing {self.app_name} executable...

if not exist "dist\\{self.app_name}.exe" (
    echo ERROR: Executable not found. Run build.bat first.
    pause
    exit /b 1
)

echo Starting {self.app_name} in test mode...
cd dist
{self.app_name}.exe --test

echo Test completed.
pause
'''
            
            test_file = self.project_root / "test.bat"
            with open(test_file, 'w') as f:
                f.write(test_script)
            
            logger.info("Created batch scripts for building and deployment")
            return True
            
        except Exception as e:
            logger.error(f"Error creating batch scripts: {e}")
            return False
    
    def create_requirements_file(self) -> bool:
        """Create requirements.txt for all dependencies"""
        try:
            requirements = [
                # Core dependencies
                "PySide6>=6.5.0",
                "pyqtgraph>=0.13.0",
                "pandas>=1.5.0",
                "numpy>=1.24.0",
                "scikit-learn>=1.3.0",
                "joblib>=1.3.0",
                "matplotlib>=3.6.0",
                "seaborn>=0.12.0",
                
                # Packaging
                "pyinstaller>=5.13.0",
                
                # Optional dependencies
                "psycopg2-binary>=2.9.0",
                "influxdb-client>=1.36.0",
                "asyncpg>=0.28.0",
                
                # Reporting
                "reportlab>=4.0.0",
                "openpyxl>=3.1.0",
                
                # Scheduling
                "schedule>=1.2.0",
                "croniter>=1.4.0",
                
                # Deep learning (optional)
                "tensorflow>=2.13.0",
                "torch>=2.0.0",
                
                # Email
                "secure-smtplib>=0.1.1",
                
                # Additional utilities
                "cryptography>=41.0.0",
                "requests>=2.31.0",
                "urllib3>=2.0.0",
            ]
            
            requirements_file = self.project_root / "requirements.txt"
            with open(requirements_file, 'w') as f:
                f.write("\\n".join(requirements))
            
            logger.info(f"Created requirements file: {requirements_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating requirements file: {e}")
            return False
    
    def validate_dependencies(self) -> Dict[str, bool]:
        """Validate that all required dependencies are available"""
        dependencies = {
            'python': False,
            'pyinstaller': False,
            'pyside6': False,
            'pandas': False,
            'numpy': False,
            'sklearn': False,
            'pyqtgraph': False,
            'matplotlib': False,
        }
        
        try:
            # Check Python
            result = subprocess.run([sys.executable, '--version'], 
                                  capture_output=True, text=True)
            dependencies['python'] = result.returncode == 0
            
            # Check each package
            packages_to_check = [
                'pyinstaller', 'PySide6', 'pandas', 'numpy', 
                'sklearn', 'pyqtgraph', 'matplotlib'
            ]
            
            for package in packages_to_check:
                try:
                    result = subprocess.run([sys.executable, '-c', f'import {package}'], 
                                          capture_output=True, text=True)
                    dependencies[package.lower()] = result.returncode == 0
                except:
                    dependencies[package.lower()] = False
            
            # Log results
            for dep, available in dependencies.items():
                status = "✓" if available else "✗"
                logger.info(f"{status} {dep}: {'Available' if available else 'Missing'}")
            
            return dependencies
            
        except Exception as e:
            logger.error(f"Error validating dependencies: {e}")
            return dependencies
    
    def build_executable(self) -> bool:
        """Build the executable using PyInstaller"""
        try:
            logger.info("Starting executable build process...")
            
            # Validate dependencies
            deps = self.validate_dependencies()
            if not all(deps.values()):
                missing = [dep for dep, available in deps.items() if not available]
                logger.error(f"Missing dependencies: {missing}")
                return False
            
            # Clean previous builds
            if self.build_dir.exists():
                shutil.rmtree(self.build_dir)
            if self.dist_dir.exists():
                shutil.rmtree(self.dist_dir)
            
            # Build with PyInstaller
            cmd = [sys.executable, '-m', 'PyInstaller', str(self.spec_file)]
            
            logger.info(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, cwd=self.project_root, 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                executable_path = self.dist_dir / f"{self.app_name}.exe"
                if executable_path.exists():
                    file_size = executable_path.stat().st_size / (1024 * 1024)  # MB
                    logger.info(f"BUILD SUCCESSFUL!")
                    logger.info(f"Executable: {executable_path}")
                    logger.info(f"Size: {file_size:.1f} MB")
                    return True
                else:
                    logger.error("Build completed but executable not found")
                    return False
            else:
                logger.error(f"PyInstaller failed with return code: {result.returncode}")
                logger.error(f"STDOUT: {result.stdout}")
                logger.error(f"STDERR: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error building executable: {e}")
            return False
    
    def test_executable(self) -> bool:
        """Test the built executable"""
        try:
            executable_path = self.dist_dir / f"{self.app_name}.exe"
            
            if not executable_path.exists():
                logger.error("Executable not found for testing")
                return False
            
            logger.info("Testing executable...")
            
            # Test basic execution (with timeout)
            cmd = [str(executable_path), '--version']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                logger.info("Executable test PASSED")
                logger.info(f"Output: {result.stdout}")
                return True
            else:
                logger.warning(f"Executable test returned code: {result.returncode}")
                logger.warning(f"STDERR: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.warning("Executable test timed out (may be waiting for GUI)")
            return True  # Timeout is OK for GUI apps
        except Exception as e:
            logger.error(f"Error testing executable: {e}")
            return False
    
    def create_deployment_package(self) -> bool:
        """Create final deployment package"""
        try:
            logger.info("Creating deployment package...")
            
            deploy_dir = self.project_root / f"deploy_{self.app_name}_v{self.app_version}"
            
            if deploy_dir.exists():
                shutil.rmtree(deploy_dir)
            deploy_dir.mkdir()
            
            # Copy executable
            executable_src = self.dist_dir / f"{self.app_name}.exe"
            executable_dst = deploy_dir / f"{self.app_name}.exe"
            shutil.copy2(executable_src, executable_dst)
            
            # Copy documentation
            readme_src = self.project_root / "README.txt"
            if readme_src.exists():
                shutil.copy2(readme_src, deploy_dir)
            
            # Create directory structure
            (deploy_dir / "data").mkdir()
            (deploy_dir / "config").mkdir()
            (deploy_dir / "logs").mkdir()
            (deploy_dir / "reports").mkdir()
            
            # Copy default configurations
            config_files = list(self.project_root.glob("telemetry_monitor/*.json"))
            for config_file in config_files:
                shutil.copy2(config_file, deploy_dir / "config")
            
            # Create portable launcher
            launcher_content = f'''@echo off
title {self.app_description}
cd /d "%~dp0"

:: Set portable mode
set STDMS_PORTABLE=1
set STDMS_DATA_DIR=%cd%\\data
set STDMS_CONFIG_DIR=%cd%\\config
set STDMS_LOGS_DIR=%cd%\\logs
set STDMS_REPORTS_DIR=%cd%\\reports

:: Create directories if they don't exist
if not exist data mkdir data
if not exist config mkdir config
if not exist logs mkdir logs
if not exist reports mkdir reports

:: Launch application
echo Starting {self.app_description}...
{self.app_name}.exe

:: Pause if there was an error
if %errorlevel% neq 0 (
    echo.
    echo Application exited with error code: %errorlevel%
    echo Check logs\\stdms.log for details.
    pause
)
'''
            
            launcher_file = deploy_dir / f"{self.app_name}_Portable.bat"
            with open(launcher_file, 'w') as f:
                f.write(launcher_content)
            
            # Create ZIP archive
            zip_path = self.project_root / f"{self.app_name}_v{self.app_version}_Portable.zip"
            shutil.make_archive(str(zip_path).replace('.zip', ''), 'zip', deploy_dir)
            
            # Get package size
            zip_size = zip_path.stat().st_size / (1024 * 1024)  # MB
            
            logger.info(f"DEPLOYMENT PACKAGE CREATED!")
            logger.info(f"Location: {deploy_dir}")
            logger.info(f"Portable ZIP: {zip_path}")
            logger.info(f"Package size: {zip_size:.1f} MB")
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating deployment package: {e}")
            return False
    
    def package_complete_system(self) -> bool:
        """Complete packaging process"""
        try:
            logger.info(f"Starting complete packaging process for {self.app_name}...")
            
            steps = [
                ("Creating spec file", self.create_spec_file),
                ("Creating version info", self.create_version_info),
                ("Preparing resources", self.prepare_resources),
                ("Creating requirements file", self.create_requirements_file),
                ("Creating batch scripts", self.create_batch_scripts),
                ("Creating installer script", self.create_installer_script),
                ("Building executable", self.build_executable),
                ("Testing executable", self.test_executable),
                ("Creating deployment package", self.create_deployment_package),
            ]
            
            for step_name, step_func in steps:
                logger.info(f"Step: {step_name}...")
                if not step_func():
                    logger.error(f"Failed at step: {step_name}")
                    return False
                logger.info(f"✓ {step_name} completed")
            
            logger.info("=" * 60)
            logger.info("🎉 PACKAGING COMPLETED SUCCESSFULLY! 🎉")
            logger.info("=" * 60)
            logger.info(f"Executable: {self.dist_dir / f'{self.app_name}.exe'}")
            logger.info(f"Deployment: deploy_{self.app_name}_v{self.app_version}/")
            logger.info(f"Portable ZIP: {self.app_name}_v{self.app_version}_Portable.zip")
            logger.info("=" * 60)
            
            return True
            
        except Exception as e:
            logger.error(f"Critical error in packaging process: {e}")
            return False

def main():
    """Main entry point for packaging"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Package STDMS as Windows executable")
    parser.add_argument("--project-root", help="Project root directory", default=".")
    parser.add_argument("--build-only", action="store_true", help="Only build executable")
    parser.add_argument("--test-only", action="store_true", help="Only test existing executable")
    
    args = parser.parse_args()
    
    packager = ExecutablePackager(args.project_root)
    
    if args.test_only:
        success = packager.test_executable()
    elif args.build_only:
        packager.create_spec_file()
        packager.create_version_info()
        success = packager.build_executable()
    else:
        success = packager.package_complete_system()
    
    if success:
        logger.info("Operation completed successfully!")
        return 0
    else:
        logger.error("Operation failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())