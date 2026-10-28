@echo off
rem Load the VS x64 developer environment, then dump the resulting variables.
rem Called from tools/dev/vcvars-env.sh (path kept space-free so Git Bash quoting works).
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
set
