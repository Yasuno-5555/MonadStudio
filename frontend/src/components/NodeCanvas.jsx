// Node Canvas - Using @xyflow/react with all node types
import { useCallback, useState, useEffect } from 'react';
import {
    ReactFlow,
    Background,
    Controls,
    MiniMap,
    ReactFlowProvider,
    Panel
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import useGraphStore from '../store/graphStore';
import HouseholdNode from '../nodes/HouseholdNode';
import CentralBankNode from '../nodes/CentralBankNode';
import FirmNode from '../nodes/FirmNode';
import MarketClearingNode from '../nodes/MarketClearingNode';
import FiscalAuthorityNode from '../nodes/FiscalAuthorityNode';
import CustomEdge from '../edges/CustomEdge';
import ContextMenu from './ContextMenu';

// IMPORTANT: nodeTypes must be defined OUTSIDE the component
const nodeTypes = {
    household: HouseholdNode,
    centralbank: CentralBankNode,
    firm: FirmNode,
    marketclearing: MarketClearingNode,
    fiscalauthority: FiscalAuthorityNode,
};

const edgeTypes = {
    custom: CustomEdge,
};

const nodeTemplates = [
    { type: 'household', label: '🏠 Household', params: { beta: 0.986, sigma: 2.0, chi0: 0.0, chi1: 5.0, chi2: 0.0 } },
    { type: 'centralbank', label: '🏦 Central Bank', params: { phi_pi: 1.5, phi_y: 0.5 } },
    { type: 'firm', label: '🏭 Firm', params: { theta: 0.75, epsilon: 6.0 } },
    { type: 'fiscalauthority', label: '🏛️ Fiscal', params: { G: 0.2, B: 0.6 } },
    { type: 'marketclearing', label: '⚖️ Market', params: {} },
];

function NodePalette() {
    const addNode = useGraphStore((state) => state.addNode);

    const handleAddNode = (template) => {
        addNode({
            type: template.type,
            label: template.label,
            params: { ...template.params }
        });
    };

    return (
        <Panel position="top-left" style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
            {nodeTemplates.map((t) => (
                <button
                    key={t.type}
                    onClick={() => handleAddNode(t)}
                    style={{
                        padding: '8px 12px',
                        background: '#21262d',
                        border: '1px solid #30363d',
                        borderRadius: '6px',
                        color: '#c9d1d9',
                        cursor: 'pointer',
                        fontSize: '0.8rem',
                        textAlign: 'left'
                    }}
                >
                    {t.label}
                </button>
            ))}
        </Panel>
    );
}

function Flow() {
    const nodes = useGraphStore((state) => state.nodes);
    const edges = useGraphStore((state) => state.edges);
    const selectedNode = useGraphStore((state) => state.selectedNode);
    const onNodesChange = useGraphStore((state) => state.onNodesChange);
    const onEdgesChange = useGraphStore((state) => state.onEdgesChange);
    const onConnect = useGraphStore((state) => state.onConnect);
    const selectNode = useGraphStore((state) => state.selectNode);
    const deleteEdge = useGraphStore((state) => state.deleteEdge);
    const deleteNode = useGraphStore((state) => state.deleteNode);

    // Context menu state
    const [contextMenu, setContextMenu] = useState(null);

    // Keyboard event handler for Delete key
    useEffect(() => {
        const handleKeyDown = (event) => {
            if ((event.key === 'Delete' || event.key === 'Backspace') && selectedNode) {
                event.preventDefault();
                deleteNode(selectedNode.id);
            }
        };

        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [selectedNode, deleteNode]);

    const handleNodeClick = useCallback((event, node) => {
        selectNode(node);
        setContextMenu(null);
    }, [selectNode]);

    const handleNodeContextMenu = useCallback((event, node) => {
        event.preventDefault();
        setContextMenu({
            x: event.clientX,
            y: event.clientY,
            type: 'node',
            id: node.id,
            label: node.data.label || node.id
        });
    }, []);

    const handleEdgeClick = useCallback((event, edge) => {
        // Just select, don't delete
        setContextMenu(null);
    }, []);

    const handleEdgeContextMenu = useCallback((event, edge) => {
        event.preventDefault();
        setContextMenu({
            x: event.clientX,
            y: event.clientY,
            type: 'edge',
            id: edge.id,
            label: `Edge: ${edge.source} → ${edge.target}`
        });
    }, []);

    const handlePaneClick = useCallback(() => {
        setContextMenu(null);
        selectNode(null);
    }, [selectNode]);

    const handleDeleteFromMenu = useCallback(() => {
        if (contextMenu) {
            if (contextMenu.type === 'node') {
                deleteNode(contextMenu.id);
            } else if (contextMenu.type === 'edge') {
                deleteEdge(contextMenu.id);
            }
            setContextMenu(null);
        }
    }, [contextMenu, deleteNode, deleteEdge]);

    return (
        <>
            <ReactFlow
                nodes={nodes}
                edges={edges}
                onNodesChange={onNodesChange}
                onEdgesChange={onEdgesChange}
                onConnect={onConnect}
                onNodeClick={handleNodeClick}
                onEdgeClick={handleEdgeClick}
                onNodeContextMenu={handleNodeContextMenu}
                onEdgeContextMenu={handleEdgeContextMenu}
                onPaneClick={handlePaneClick}
                nodeTypes={nodeTypes}
                edgeTypes={edgeTypes}
                fitView
                edgesFocusable={true}
                edgesReconnectable={true}
                elementsSelectable={true}
                selectNodesOnDrag={false}
                defaultEdgeOptions={{
                    style: { strokeWidth: 3, stroke: '#58a6ff' },
                    type: 'custom',
                    animated: false,
                }}
            >
                <Background color="#333" gap={20} />
                <Controls />
                <MiniMap />
                <NodePalette />
            </ReactFlow>

            {contextMenu && (
                <ContextMenu
                    x={contextMenu.x}
                    y={contextMenu.y}
                    label={contextMenu.label}
                    onClose={() => setContextMenu(null)}
                    onDelete={handleDeleteFromMenu}
                />
            )}
        </>
    );
}

export default function NodeCanvas() {
    return (
        <div style={{ height: '100%', width: '100%' }}>
            <ReactFlowProvider>
                <Flow />
            </ReactFlowProvider>
        </div>
    );
}
