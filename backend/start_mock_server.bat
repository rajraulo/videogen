@echo off
echo ================================================
echo  VideoGen Mock Server  (no GPU required)
echo  Android emulator connects via 10.0.2.2:8000
echo ================================================
echo.

cd /d "%~dp0"

:: Install lightweight dependencies if needed
echo Checking dependencies...
pip install fastapi uvicorn[standard] numpy imageio imageio-ffmpeg python-multipart --quiet

echo.
echo Starting server on http://0.0.0.0:8000
echo Press Ctrl+C to stop.
echo.

uvicorn api.mock_server:app --host 0.0.0.0 --port 8000 --reload

pause
