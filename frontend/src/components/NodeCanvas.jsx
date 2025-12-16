// Node Canvas - Using @xyflow/react with all node types
import { useCallback, useState } from 'react';
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
    // Store Selectors
    const nodes = useGraphStore((state) => state.nodes);
    const edges = useGraphStore((state) => state.edges);
    const onNodesChange = useGraphStore((state) => state.onNodesChange);
    const onEdgesChange = useGraphStore((state) => state.onEdgesChange);
    const onConnect = useGraphStore((state) => state.onConnect);
    const onNodesDelete = useGraphStore((state) => state.onNodesDelete);
    const onEdgesDelete = useGraphStore((state) => state.onEdgesDelete);

    // Actions
    const selectNode = useGraphStore((state) => state.selectNode);
    const selectEdge = useGraphStore((state) => state.selectEdge);
    const deleteNode = useGraphStore((state) => state.deleteNode);
    const deleteEdge = useGraphStore((state) => state.deleteEdge);

    // Context menu state
    const [contextMenu, setContextMenu] = useState(null);

    const handleNodeClick = useCallback((event, node) => {
        // Selection is handled by onNodesChange, but we can double check or trigger Inspector logic here if needed
        // React Flow handles selection state internally via changes
        setContextMenu(null);
    }, []);

    const handleEdgeClick = useCallback((event, edge) => {
        setContextMenu(null);
    }, []);

    const handleNodeContextMenu = useCallback((event, node) => {
        event.preventDefault();
        // Also select the node on right click
        selectNode(node);
        setContextMenu({
            x: event.clientX,
            y: event.clientY,
            type: 'node',
            id: node.id,
            label: node.data.label || node.id
        });
    }, [selectNode]);

    const handleEdgeContextMenu = useCallback((event, edge) => {
        event.preventDefault();
        selectEdge(edge);
        setContextMenu({
            x: event.clientX,
            y: event.clientY,
            type: 'edge',
            id: edge.id,
            label: `Connection`
        });
    }, [selectEdge]);

    const handlePaneClick = useCallback(() => {
        setContextMenu(null);
    }, []);

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
                onNodesDelete={onNodesDelete}
                onEdgesDelete={onEdgesDelete}
                onConnect={onConnect}
                onNodeClick={handleNodeClick}
                onEdgeClick={handleEdgeClick}
                onNodeContextMenu={handleNodeContextMenu}
                onEdgeContextMenu={handleEdgeContextMenu}
                onPaneClick={handlePaneClick}
                nodeTypes={nodeTypes}
                edgeTypes={edgeTypes}
                fitView
                // Native Deletion Support
                deleteKeyCode={['Backspace', 'Delete']}
                // Interaction Settings
                edgesFocusable={true}
                edgesReconnectable={true}
                elementsSelectable={true}
                selectNodesOnDrag={false}
                // Default Options
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
