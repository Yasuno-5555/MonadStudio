// Inspector Panel - Parameter editing for all node types & Edges
import useGraphStore from '../store/graphStore';

const inspectorStyle = {
    flex: 1,
    padding: '1rem',
    borderBottom: '1px solid #30363d',
    overflowY: 'auto'
};

const headerStyle = {
    fontWeight: 600,
    marginBottom: '1rem',
    color: '#58a6ff'
};

const emptyStyle = {
    color: '#8b949e',
    fontStyle: 'italic'
};

const paramGroupStyle = {
    marginBottom: '1rem'
};

const labelStyle = {
    display: 'block',
    fontSize: '0.8rem',
    color: '#8b949e',
    marginBottom: '0.25rem'
};

const inputContainerStyle = {
    display: 'flex',
    alignItems: 'center',
    gap: '8px'
};

const rangeStyle = {
    flex: 1,
    accentColor: '#58a6ff'
};

const valueStyle = {
    width: '50px',
    textAlign: 'right',
    fontFamily: 'monospace',
    color: '#c9d1d9'
};

const infoRowStyle = {
    display: 'flex',
    justifyContent: 'space-between',
    marginBottom: '8px',
    fontSize: '0.85rem'
};

const infoLabelStyle = {
    color: '#8b949e'
};

const infoValueStyle = {
    color: '#c9d1d9',
    fontFamily: 'monospace'
};

function ParamSlider({ label, value, min, max, step, onChange, decimals = 2 }) {
    return (
        <div style={paramGroupStyle}>
            <label style={labelStyle}>{label}</label>
            <div style={inputContainerStyle}>
                <input
                    type="range"
                    min={min}
                    max={max}
                    step={step}
                    value={value}
                    onChange={(e) => onChange(parseFloat(e.target.value))}
                    style={rangeStyle}
                />
                <span style={valueStyle}>{value.toFixed(decimals)}</span>
            </div>
        </div>
    );
}

export default function Inspector() {
    const selectedNode = useGraphStore((state) => state.selectedNode);
    const selectedEdge = useGraphStore((state) => state.selectedEdge);
    const updateNodeParams = useGraphStore((state) => state.updateNodeParams);
    const deleteNode = useGraphStore((state) => state.deleteNode);
    const deleteEdge = useGraphStore((state) => state.deleteEdge);

    if (selectedEdge) {
        const { id, source, target, data, sourceHandle, targetHandle } = selectedEdge;

        return (
            <div style={inspectorStyle}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                    <div style={headerStyle}>Link Info</div>
                    <button
                        onClick={() => deleteEdge(id)}
                        style={{
                            background: '#f85149',
                            border: 'none',
                            color: '#fff',
                            padding: '4px 8px',
                            borderRadius: '4px',
                            cursor: 'pointer',
                            fontSize: '0.75rem'
                        }}
                    >
                        Delete
                    </button>
                </div>

                <div style={{ background: '#21262d', padding: '12px', borderRadius: '6px', marginBottom: '1rem' }}>
                    <div style={infoRowStyle}>
                        <span style={infoLabelStyle}>Source Node</span>
                        <span style={infoValueStyle}>{source}</span>
                    </div>
                    <div style={infoRowStyle}>
                        <span style={infoLabelStyle}>Source Port</span>
                        <span style={{ ...infoValueStyle, color: '#58a6ff' }}>{data?.port_out || sourceHandle || 'out'}</span>
                    </div>
                    <hr style={{ borderColor: '#30363d', margin: '8px 0' }} />
                    <div style={infoRowStyle}>
                        <span style={infoLabelStyle}>Target Node</span>
                        <span style={infoValueStyle}>{target}</span>
                    </div>
                    <div style={infoRowStyle}>
                        <span style={infoLabelStyle}>Target Port</span>
                        <span style={{ ...infoValueStyle, color: '#d29922' }}>{data?.port_in || targetHandle || 'in'}</span>
                    </div>
                </div>

                <div style={emptyStyle}>
                    Semantic connection established.
                </div>
            </div>
        );
    }

    if (!selectedNode) {
        return (
            <div style={inspectorStyle}>
                <div style={headerStyle}>Inspector</div>
                <div style={emptyStyle}>Select a node or link to edit</div>
            </div>
        );
    }

    const { id, type, data } = selectedNode;
    const params = data.params || {};

    const handleChange = (key, value) => {
        updateNodeParams(id, { ...params, [key]: value });
    };

    const handleDelete = () => {
        if (confirm('Delete this node?')) {
            deleteNode(id);
        }
    };

    return (
        <div style={inspectorStyle}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                <div style={headerStyle}>{data.label || type}</div>
                <button
                    onClick={handleDelete}
                    style={{
                        background: '#f85149',
                        border: 'none',
                        color: '#fff',
                        padding: '4px 8px',
                        borderRadius: '4px',
                        cursor: 'pointer',
                        fontSize: '0.75rem'
                    }}
                >
                    Delete
                </button>
            </div>

            <div>
                {type === 'household' && (
                    <>
                        <ParamSlider label="β (Discount Factor)" value={params.beta ?? 0.986}
                            min={0.9} max={0.999} step={0.001} decimals={3}
                            onChange={(v) => handleChange('beta', v)} />
                        <ParamSlider label="σ (Risk Aversion)" value={params.sigma ?? 2.0}
                            min={0.5} max={5.0} step={0.1} decimals={1}
                            onChange={(v) => handleChange('sigma', v)} />
                        <ParamSlider label="χ₁ (Adjustment Cost)" value={params.chi1 ?? 5.0}
                            min={0} max={20} step={0.5} decimals={1}
                            onChange={(v) => handleChange('chi1', v)} />
                    </>
                )}

                {type === 'centralbank' && (
                    <>
                        <ParamSlider label="φπ (Inflation Response)" value={params.phi_pi ?? 1.5}
                            min={1.0} max={3.0} step={0.1} decimals={1}
                            onChange={(v) => handleChange('phi_pi', v)} />
                        <ParamSlider label="φy (Output Gap Response)" value={params.phi_y ?? 0.5}
                            min={0} max={1.5} step={0.1} decimals={1}
                            onChange={(v) => handleChange('phi_y', v)} />
                    </>
                )}

                {type === 'firm' && (
                    <>
                        <ParamSlider label="θ (Price Stickiness)" value={params.theta ?? 0.75}
                            min={0} max={0.99} step={0.01} decimals={2}
                            onChange={(v) => handleChange('theta', v)} />
                        <ParamSlider label="ε (Elasticity)" value={params.epsilon ?? 6.0}
                            min={2} max={20} step={0.5} decimals={1}
                            onChange={(v) => handleChange('epsilon', v)} />
                    </>
                )}

                {type === 'fiscalauthority' && (
                    <>
                        <ParamSlider label="G (Government Spending)" value={params.G ?? 0.2}
                            min={0} max={0.5} step={0.01} decimals={2}
                            onChange={(v) => handleChange('G', v)} />
                        <ParamSlider label="B (Bonds Supply)" value={params.B ?? 0.6}
                            min={0} max={2.0} step={0.1} decimals={1}
                            onChange={(v) => handleChange('B', v)} />
                    </>
                )}

                {type === 'marketclearing' && (
                    <div style={emptyStyle}>
                        Market clearing conditions are computed automatically.
                    </div>
                )}
            </div>
        </div>
    );
}
