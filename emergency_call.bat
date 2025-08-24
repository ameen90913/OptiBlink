@echo off
echo.
echo ================================
echo    EMERGENCY CALL ACTIVATED
echo ================================
echo Target Number: +91 6366011723
echo.
echo Attempting to initiate call...
echo.

REM Try multiple methods
start "" "ms-phone:call?number=+916366011723"
timeout /t 2 /nobreak >nul

start "" "tel:+916366011723"
timeout /t 1 /nobreak >nul

echo.
echo ================================
echo PHONE LINK SHOULD HAVE OPENED
echo ================================
echo 1. Check your Phone Link app
echo 2. The number should be displayed
echo 3. Click the CALL button in Phone Link
echo 4. Or manually dial: +91 6366011723
echo ================================
echo.
echo Number copied to clipboard for manual dialing
echo +916366011723 | clip
echo.
pause
