@echo off
echo ========================================================
echo Reproducing Machine Learning Assignment 2
echo Roll Number: M26CSA017 | Seed: 26017
echo ========================================================

python scripts/run_all.py
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Pipeline execution failed.
    exit /b %ERRORLEVEL%
)

echo Compiling CVPR Report to PDF...
python -c "import typst; typst.compile('report/report.typ', output='report/report.pdf')"
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Report compilation failed.
    exit /b %ERRORLEVEL%
)

echo [SUCCESS] All experiments reproduced and report/report.pdf generated!
