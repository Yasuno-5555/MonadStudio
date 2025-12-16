// Scope Panel - Results display
import useGraphStore from '../store/graphStore';
import DistributionPanel from './DistributionPanel';

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

    // Defensive: check if results.results exists
    if (!results.results) {
        return (
            <div className="scope-panel">
                <div className="scope-header">Results</div>
                <div className="scope-empty" style={{ color: '#f85149' }}>
                    Error: Invalid results format
                </div>
                <pre style={{ fontSize: '0.7rem', color: '#8b949e', overflow: 'auto' }}>
                    {JSON.stringify(results, null, 2)}
                </pre>
            </div>
        );
    }

    const { execution_order, nodes } = results.results;

    // Defensive: ensure execution_order and nodes exist
    if (!execution_order || !nodes) {
        return (
            <div className="scope-panel">
                <div className="scope-header">Results</div>
                <div className="scope-empty" style={{ color: '#f85149' }}>
                    Error: Missing execution_order or nodes
                </div>
            </div>
        );
    }

    // Extract distribution data from first Household node (if available)
    let distributionData = null;
    for (const [id, node] of Object.entries(nodes)) {
        if (node.type === 'Household' && node.steady_state) {
            const ss = node.steady_state;
            if (ss.distribution_data) {
                distributionData = {
                    distribution_data: ss.distribution_data,
                    grid_a: ss.grid_a,
                    grid_z: ss.grid_z,
                    Na: ss.Na,
                    Nz: ss.Nz
                };
                break;
            }
        }
    }

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
                {/* Distribution Heatmap */}
                {distributionData && (
                    <DistributionPanel {...distributionData} />
                )}
            </div>
        </div>
    );
}

