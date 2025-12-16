// Scope Panel - Results display
import useGraphStore from '../store/graphStore';

export default function ScopePanel() {
    const { results } = useGraphStore();

    if (!results) {
        return (
            <div className="scope-panel">
                <div className="scope-header">Results</div>
                <div className="scope-empty">Run simulation to see results</div>
            </div>
        );
    }

    const { execution_order, nodes } = results.results;

    return (
        <div className="scope-panel">
            <div className="scope-header">Results ✅</div>
            <div className="scope-content">
                <div className="scope-section">
                    <strong>Execution Order:</strong>
                    <span>{execution_order.join(' → ')}</span>
                </div>
                {Object.entries(nodes).map(([id, node]) => (
                    <div key={id} className="scope-node">
                        <div className="scope-node-header">{id} ({node.type})</div>
                        {node.steady_state && (
                            <div className="scope-values">
                                <div className="scope-value">
                                    <span>r</span>
                                    <strong>{(node.steady_state.r * 100).toFixed(2)}%</strong>
                                </div>
                                <div className="scope-value">
                                    <span>w</span>
                                    <strong>{node.steady_state.w.toFixed(4)}</strong>
                                </div>
                                <div className="scope-value">
                                    <span>C</span>
                                    <strong>{node.steady_state.C.toFixed(4)}</strong>
                                </div>
                                <div className="scope-value">
                                    <span>Y</span>
                                    <strong>{node.steady_state.Y.toFixed(4)}</strong>
                                </div>
                            </div>
                        )}
                    </div>
                ))}
            </div>
        </div>
    );
}
