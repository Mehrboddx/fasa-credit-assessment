@echo off
REM Run the Fasanara Credit Assessment Streamlit App
REM Usage: run_app.bat

echo.
echo ========================================
echo Fasanara Credit Assessment App
echo ========================================
echo.

REM Check if venv exists
if exist .venv (
    echo Activating virtual environment...
    call .venv\Scripts\activate.bat
) else (
    echo Warning: Virtual environment not found
    echo Please run from project root
)

echo.
echo Starting Streamlit app...
echo Opening browser at http://localhost:8501
echo Press Ctrl+C to stop
echo.

streamlit run app/app.py

pause
