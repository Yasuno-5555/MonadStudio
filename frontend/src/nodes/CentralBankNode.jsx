// CentralBank Node Component
import { Handle, Position } from '@xyflow/react';

const nodeStyle = {
    padding: '12px',
    borderRadius: '8px',
    minWidth: '140px',
    fontSize: '0.85rem',
    boxShadow: '0 4px 12px rgba(0,0,0,0.3)',
    background: 'linear-gradient(135deg, #1a3a5c, #2d4a6d)',
    border: '2px solid #58a6ff',
    color: '#c9d1d9'
};

const headerStyle = {
    fontWeight: 600,
    marginBottom: '8px',
    color: '#fff'
};

const paramStyle = {
    color: '#8b949e',
    fontSize: '0.8rem'
};

export default function CentralBankNode({ data, selected }) {
    const style = {
        ...nodeStyle,
        boxShadow: selected
            ? '0 0 0 2px #58a6ff, 0 4px 16px rgba(88,166,255,0.3)'
            : nodeStyle.boxShadow
    };

    return (
        <div style={style}>
            <div style={headerStyle}>🏦 Central Bank</div>
            <div>
                <div style={paramStyle}>φπ: {data.params.phi_pi}</div>
                <div style={paramStyle}>φy: {data.params.phi_y}</div>
            </div>
            <Handle type="source" position={Position.Right} id="r" />
        </div>
    );
}
