// Monad Web Worker
importScripts('/wasm/MonadEngineWasm.js');

let engine = null;

self.onmessage = function(e) {
    const { type, payload } = e.data;
    
    if (type === 'INIT') {
        const Module = {
            locateFile: (path) => '/wasm/' + path,
            onRuntimeInitialized: () => {
                engine = new Module.MonadWasm();
                console.log("MonadWasm Initialized in Worker");
                self.postMessage({ type: 'READY' });
            }
        };
        // Initialize the module (Factory function)
        MonadEngineModule(Module);
    }
    
    else if (type === 'SOLVE_SS') {
        if(!engine) {
            console.error("Engine not initialized");
            return;
        }
        try {
            console.log("Worker: Solving SS...");
            engine.load_config(JSON.stringify(payload));
            const r_ss = engine.solve_ss();
            self.postMessage({ type: 'RESULT_SS', payload: { r_ss } });
        } catch(err) {
            console.error(err);
        }
    }
    
    else if (type === 'SOLVE_IRF') {
        if(!engine) return;
        try {
            const res = engine.solve_irf();
            self.postMessage({ type: 'RESULT_IRF', payload: res });
        } catch(err) {
             console.error(err);
        }
    }
};
