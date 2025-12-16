// Household Node Component
import { Handle, Position } from '@xyflow/react';

const nodeStyle = {
    padding: '12px',
    borderRadius: '8px',
    minWidth: '140px',
    fontSize: '0.85rem',
    boxShadow: '0 4px 12px rgba(0,0,0,0.3)',
    background: 'linear-gradient(135deg, #1a472a, #2d5a3d)',
    border: '2px solid #3fb950',
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

export default function HouseholdNode({ data, selected }) {
    const style = {
        ...nodeStyle,
        boxShadow: selected
            ? '0 0 0 2px #58a6ff, 0 4px 16px rgba(88,166,255,0.3)'
            : nodeStyle.boxShadow
    };

    return (
        <div style={style}>
            <Handle type="target" position={Position.Left} id="r_a" />
            <div style={headerStyle}>🏠 Household</div>
            <div>
                <div style={paramStyle}>β: {data.params.beta?.toFixed(3) ?? '0.986'}</div>
                <div style={paramStyle}>σ: {data.params.sigma?.toFixed(1) ?? '2.0'}</div>
                <div style={paramStyle}>χ₁: {data.params.chi1?.toFixed(1) ?? '5.0'}</div>
            </div>
            <Handle type="source" position={Position.Right} id="out" />
        </div>
    );
}
