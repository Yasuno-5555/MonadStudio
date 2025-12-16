// MarketClearing Node Component
import { Handle, Position } from '@xyflow/react';

const nodeStyle = {
    padding: '12px',
    borderRadius: '8px',
    minWidth: '160px',
    fontSize: '0.85rem',
    boxShadow: '0 4px 12px rgba(0,0,0,0.3)',
    background: 'linear-gradient(135deg, #5c1a5c, #6d2d6d)',
    border: '2px solid #a371f7',
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

export default function MarketClearingNode({ data, selected }) {
    const style = {
        ...nodeStyle,
        boxShadow: selected
            ? '0 0 0 2px #a371f7, 0 4px 16px rgba(163,113,247,0.3)'
            : nodeStyle.boxShadow
    };

    return (
        <div style={style}>
            <Handle type="target" position={Position.Left} id="A_hh" style={{ top: '25%' }} />
            <Handle type="target" position={Position.Left} id="C_hh" style={{ top: '50%' }} />
            <Handle type="target" position={Position.Left} id="Y_firm" style={{ top: '75%' }} />
            <div style={headerStyle}>⚖️ Market Clearing</div>
            <div>
                <div style={paramStyle}>Assets: A = K + B</div>
                <div style={paramStyle}>Goods: Y = C + G</div>
            </div>
            <Handle type="source" position={Position.Right} id="equilibrium" />
        </div>
    );
}
