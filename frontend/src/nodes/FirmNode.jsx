// Firm Node Component
import { Handle, Position } from '@xyflow/react';

const nodeStyle = {
    padding: '12px',
    borderRadius: '8px',
    minWidth: '140px',
    fontSize: '0.85rem',
    boxShadow: '0 4px 12px rgba(0,0,0,0.3)',
    background: 'linear-gradient(135deg, #5c4a1a, #6d5a2d)',
    border: '2px solid #d29922',
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

export default function FirmNode({ data, selected }) {
    const style = {
        ...nodeStyle,
        boxShadow: selected
            ? '0 0 0 2px #d29922, 0 4px 16px rgba(210,153,34,0.3)'
            : nodeStyle.boxShadow
    };

    return (
        <div style={style}>
            <Handle type="target" position={Position.Left} id="Y" />
            <div style={headerStyle}>🏭 Firm</div>
            <div>
                <div style={paramStyle}>θ: {data.params?.theta ?? 0.75}</div>
                <div style={paramStyle}>ε: {data.params?.epsilon ?? 6.0}</div>
            </div>
            <Handle type="source" position={Position.Right} id="w" style={{ top: '30%' }} />
            <Handle type="source" position={Position.Right} id="d" style={{ top: '70%' }} />
        </div>
    );
}
