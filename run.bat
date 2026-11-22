@echo off
REM Usage: run.bat [all|left|right|motion|clockwise|directions|textures|build]
REM Requires BLENDER_EXE pointing at blender.exe (Blender 4.2+).
setlocal
cd /d "%~dp0"
if "%BLENDER_EXE%"=="" set BLENDER_EXE=blender
set TASK=%1
if "%TASK%"=="" set TASK=all

if "%TASK%"=="all" (
  python code\texture_synth.py || exit /b 1
  "%BLENDER_EXE%" -b --python code\build_scene.py || exit /b 1
  "%BLENDER_EXE%" -b scene.blend --python wrapper\eyes.py -- --input all
  exit /b %errorlevel%
)
if "%TASK%"=="build" (
  python code\texture_synth.py || exit /b 1
  "%BLENDER_EXE%" -b --python code\build_scene.py
  exit /b %errorlevel%
)
"%BLENDER_EXE%" -b scene.blend --python wrapper\eyes.py -- --input %TASK% %2 %3 %4 %5
