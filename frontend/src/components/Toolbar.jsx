// Toolbar Component with Auto-Run toggle
import { useState, useEffect, useRef } from 'react';
import useGraphStore from '../store/graphStore';
import { runScenario } from '../api/orchestrator';

export default function Toolbar() {
    const nodes = useGraphStore((state) => state.nodes);
    const toScenario = useGraphStore((state) => state.toScenario);
    const setResults = useGraphStore((state) => state.setResults);

    const [running, setRunning] = useState(false);
    const [error, setError] = useState(null);
    const [autoRun, setAutoRun] = useState(true);

    const lastRunRef = useRef(null);
    const debounceRef = useRef(null);

    // Execute simulation
    const executeRun = async () => {
        setRunning(true);
        setError(null);

        try {
            const scenario = toScenario();
            console.log('Sending scenario:', JSON.stringify(scenario, null, 2));

            const result = await runScenario(scenario);
            console.log('Result:', result);

            setResults(result);
        } catch (e) {
            console.error('Run failed:', e);
            setError(e.message);
        } finally {
            setRunning(false);
        }
    };

    // Auto-run when nodes change (debounced)
    useEffect(() => {
        if (!autoRun) return;

        // Generate a signature of all node params
        const signature = JSON.stringify(nodes.map(n => ({ id: n.id, params: n.data.params })));

        // Skip if same as last run
        if (signature === lastRunRef.current) return;
        lastRunRef.current = signature;

        // Debounce: wait 500ms after last change
        if (debounceRef.current) {
            clearTimeout(debounceRef.current);
        }

        debounceRef.current = setTimeout(() => {
            executeRun();
        }, 500);

        return () => {
            if (debounceRef.current) {
                clearTimeout(debounceRef.current);
            }
        };
    }, [nodes, autoRun]);

    const handleRun = () => {
        executeRun();
    };

    return (
        <div className="toolbar">
            <div className="toolbar-title">Monad Studio</div>
            <div className="toolbar-actions" style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                <label style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.85rem', color: '#8b949e' }}>
                    <input
                        type="checkbox"
                        checked={autoRun}
                        onChange={(e) => setAutoRun(e.target.checked)}
                        style={{ accentColor: '#58a6ff' }}
                    />
                    Auto-Run
                </label>
                <button
                    className="run-button"
                    onClick={handleRun}
                    disabled={running}
                >
                    {running ? '⏳ Running...' : '▶️ Run'}
                </button>
                {error && <span className="error" style={{ color: '#f85149', fontSize: '0.8rem' }}>{error}</span>}
            </div>
        </div>
    );
}
