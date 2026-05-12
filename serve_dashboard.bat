@echo off
echo ===================================================
echo  Starting Dashboard Web Server
echo ===================================================
echo.
echo Starting local web server on port 8000...
echo Open http://localhost:8000/dashboard.html in your browser.
echo.
python -m http.server 8000
