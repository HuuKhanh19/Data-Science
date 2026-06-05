@echo off
REM ============================================================
REM Phase 2 + Phase 3 - cap nhat hang ngay + auto-publish GitHub Pages.
REM Task Scheduler 04:00 VN. Model FROZEN, KHONG refit.
REM ============================================================
setlocal
chcp 65001 >nul

set "REPO=C:\Users\BKAI\ducluong\DrugOptimization\Data-Science"
set "ENVDIR=C:\ProgramData\miniconda3\envs\ds"
set "PY=%ENVDIR%\python.exe"
set "LOG=%REPO%\logs\phase2.log"
set "TCB_NO_OPEN=1"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PYTHONUNBUFFERED=1"
set "PATH=%ENVDIR%;%ENVDIR%\Library\bin;%ENVDIR%\Library\usr\bin;%ENVDIR%\Library\mingw-w64\bin;%ENVDIR%\Scripts;%ENVDIR%\bin;%PATH%"

if not exist "%REPO%\logs" mkdir "%REPO%\logs"
cd /d "%REPO%"

echo.>> "%LOG%"
echo ================= RUN %DATE% %TIME% ================= >> "%LOG%"
"%PY%" -u scripts\phase2_update.py >> "%LOG%" 2>&1
set "RC=%ERRORLEVEL%"
echo ----------------- phase2 EXIT %RC% ----------------- >> "%LOG%"

REM Chi publish khi cap nhat du lieu thanh cong
if "%RC%"=="0" (
  "%PY%" -u scripts\publish_phase3.py >> "%LOG%" 2>&1
  echo ----------------- publish EXIT %ERRORLEVEL% ----------------- >> "%LOG%"
)

endlocal & exit /b %RC%