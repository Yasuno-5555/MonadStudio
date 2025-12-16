@echo off
echo [Monad] Building Wasm Engine...

REM Check for emcmake
where emcmake >nul 2>nul
if %errorlevel% neq 0 (
    echo [Error] emcmake not found. Please activate Emscripten environment.
    echo Example: call C:\emsdk\emsdk_env.bat
    exit /b 1
)

if not exist build_wasm mkdir build_wasm
cd build_wasm
call emcmake cmake ..
if %errorlevel% neq 0 exit /b %errorlevel%

call emmake make
if %errorlevel% neq 0 exit /b %errorlevel%

echo [Monad] Build Complete. Generated:
echo   - build_wasm/MonadEngineWasm.js
echo   - build_wasm/MonadEngineWasm.wasm
cd ..
pause
