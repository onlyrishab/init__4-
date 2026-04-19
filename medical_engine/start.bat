@echo off
echo Starting SignSetu...
cd /d "%~dp0"

if exist ".venv\Scripts\activate.bat" (
	call .venv\Scripts\activate.bat
) else if exist "venv\Scripts\activate.bat" (
	call venv\Scripts\activate.bat
) else (
	echo No virtual environment found. Creating .venv...
	py -3.11 -m venv .venv
	call .venv\Scripts\activate.bat
	python -m pip install -r requirements.txt
)

python app.py
