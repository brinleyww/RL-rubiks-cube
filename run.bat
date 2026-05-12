@echo off
echo ===================================================
echo  Rubik's Cube RL Solver - Build and Run
echo ===================================================
echo.

REM Kill any existing java agents to prevent duplicate file writing
echo [INFO] Terminating any existing background Java processes...
taskkill /F /IM java.exe /T 2>nul
echo.

REM Set JAVA_HOME if not already set
if "%JAVA_HOME%"=="" (
    if exist "C:\Program Files\Java\jdk-17" (
        set "JAVA_HOME=C:\Program Files\Java\jdk-17"
        echo [INFO] Auto-detected JAVA_HOME: C:\Program Files\Java\jdk-17
    ) else (
        echo [ERROR] JAVA_HOME is not set and could not be auto-detected.
        echo         Please install JDK 11+ and set JAVA_HOME.
        pause
        exit /b 1
    )
)

REM Use the bundled Maven
set "MVN=%~dp0.maven\apache-maven-3.9.6\bin\mvn.cmd"

if not exist "%MVN%" (
    echo [ERROR] Maven not found at %MVN%
    echo         Please download Maven and extract it to .maven\ folder
    echo         Or install Maven globally and use: mvn compile exec:java
    pause
    exit /b 1
)

echo.
echo [STEP 1] Compiling...
call "%MVN%" -f "%~dp0pom.xml" compile -q
if errorlevel 1 (
    echo [ERROR] Compilation failed!
    pause
    exit /b 1
)

echo [STEP 2] Running Rubik's Cube RL Solver...
echo.
call "%MVN%" -f "%~dp0pom.xml" exec:java -q
if errorlevel 1 (
    echo [ERROR] Execution failed!
)

echo.
pause
