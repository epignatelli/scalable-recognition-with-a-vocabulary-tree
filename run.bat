call "%ALLUSERSPROFILE%/Anaconda3/Scripts/activate.bat"

cd %~dp0

jupyter-lab --port 8888

start "" https://localhost:8888
