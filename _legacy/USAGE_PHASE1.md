# Monad Studio Phase 1: Wasm & Bridge Setup

You have successfully set up the Monad Studio v0.1 environment. 
Since the Emscripten SDK was not active in the agent environment, you need to perform the final build steps.

## 1. Build Wasm Engine
This step compiles the C++ `MonadEngine` into WebAssembly.

1.  Open a terminal where `emcmake` is available (e.g., Run `emsdk_env.bat`).
2.  Run the build script:
    ```cmd
    build_wasm.bat
    ```
    *Success:* This should create `build_wasm/MonadEngineWasm.js` and `.wasm`.

## 2. Headless Test (Optional)
Verify the engine works in the browser without React.

1.  Start a local server in the root directory:
    ```cmd
    python -m http.server 8000
    ```
2.  Open `http://localhost:8000/headless_test.html` in your browser.
    *Success:* You should see "Steady State r = 0.02..." in green text.

## 3. Deploy & Run Frontend
This connects the Wasm engine to the React UI.

1.  Deploy the built Wasm files to the frontend:
    ```cmd
    deploy_wasm.bat
    ```
2.  Install & Start Frontend:
    ```cmd
    cd frontend
    npm install
    npm run dev
    ```
3.  Open `http://localhost:5173`.
4.  Click **"1. Load Engine"** then **"2. Solve Steady State"**.

## Files Created
- `src/wasm_entry.cpp`: The C++ to Wasm binding code.
- `CMakeLists.txt`: Build configuration for Emscripten.
- `frontend/`: The React application.
- `frontend/public/monad_worker.js`: The Web Worker bridge.
