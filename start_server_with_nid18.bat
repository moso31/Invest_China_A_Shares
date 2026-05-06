@echo off
cd /d "%~dp0"
setlocal

echo.
echo Paste Eastmoney nid18 for this LOHA server session.
echo Leave blank to use the saved nid18 if one exists.
echo.
set /p "LOHA_EASTMONEY_NID18=nid18: "

if "%LOHA_EASTMONEY_NID18%"=="" (
    echo No new nid18 entered. LOHA will use the saved local value if present.
) else (
    echo LOHA_EASTMONEY_NID18 is set for this server process only.
    if not exist "%~dp0LOHA\data" mkdir "%~dp0LOHA\data"
    > "%~dp0LOHA\data\eastmoney_nid18.txt" echo %LOHA_EASTMONEY_NID18%
    echo Saved nid18 to LOHA\data\eastmoney_nid18.txt.
)

call "%~dp0start_server.bat"

endlocal
