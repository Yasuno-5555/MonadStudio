// Context Menu Component
import { useEffect } from 'react';

export default function ContextMenu({ x, y, onClose, onDelete, label }) {
    useEffect(() => {
        const handleClick = () => onClose();
        document.addEventListener('click', handleClick);
        return () => document.removeEventListener('click', handleClick);
    }, [onClose]);

    return (
        <div
            style={{
                position: 'fixed',
                top: y,
                left: x,
                background: '#21262d',
                border: '1px solid #30363d',
                borderRadius: '6px',
                boxShadow: '0 8px 24px rgba(0,0,0,0.4)',
                zIndex: 1000,
                minWidth: '150px',
            }}
        >
            <div style={{
                padding: '8px 12px',
                borderBottom: '1px solid #30363d',
                color: '#8b949e',
                fontSize: '0.75rem'
            }}>
                {label}
            </div>
            <button
                onClick={onDelete}
                style={{
                    display: 'block',
                    width: '100%',
                    padding: '8px 12px',
                    background: 'transparent',
                    border: 'none',
                    color: '#f85149',
                    textAlign: 'left',
                    cursor: 'pointer',
                    fontSize: '0.85rem',
                }}
                onMouseEnter={(e) => e.target.style.background = '#30363d'}
                onMouseLeave={(e) => e.target.style.background = 'transparent'}
            >
                🗑️ Delete
            </button>
        </div>
    );
}
