// FiscalAuthority Node Component
import { Handle, Position } from '@xyflow/react';

const nodeStyle = {
    padding: '12px',
    borderRadius: '8px',
    minWidth: '140px',
    fontSize: '0.85rem',
    boxShadow: '0 4px 12px rgba(0,0,0,0.3)',
    background: 'linear-gradient(135deg, #1a5c4a, #2d6d5a)',
    border: '2px solid #3fb9a0',
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

export default function FiscalAuthorityNode({ data, selected }) {
    const style = {
        ...nodeStyle,
        boxShadow: selected
            ? '0 0 0 2px #3fb9a0, 0 4px 16px rgba(63,185,160,0.3)'
            : nodeStyle.boxShadow
    };

    return (
        <div style={style}>
            <Handle type="target" position={Position.Left} id="r" />
            <div style={headerStyle}>🏛️ Fiscal Authority</div>
            <div>
                <div style={paramStyle}>G: {data.params?.G ?? 0.2}</div>
                <div style={paramStyle}>B: {data.params?.B ?? 0.6}</div>
            </div>
            <Handle type="source" position={Position.Right} id="T" />
        </div>
    );
}
