// Zustand store for graph state
import { create } from 'zustand';
import { applyNodeChanges, applyEdgeChanges, addEdge } from '@xyflow/react';

const useGraphStore = create((set, get) => ({
    // Initial State
    nodes: [
        {
            id: 'hh_1',
            type: 'household',
            position: { x: 250, y: 200 },
            data: {
                label: 'Household',
                params: { beta: 0.986, sigma: 2.0, chi0: 0.0, chi1: 5.0, chi2: 0.0 }
            }
        },
        {
            id: 'cb_1',
            type: 'centralbank',
            position: { x: 50, y: 200 },
            data: {
                label: 'Central Bank',
                params: { phi_pi: 1.5, phi_y: 0.5 }
            }
        }
    ],
    edges: [
        {
            id: 'e1',
            source: 'cb_1',
            target: 'hh_1',
            sourceHandle: 'r',
            targetHandle: 'r_a',
            data: { port_out: 'r', port_in: 'r_a' }, // Semantic data
            type: 'custom',
            style: { strokeWidth: 3, stroke: '#58a6ff' }
        }
    ],

    selectedNode: null,
    selectedEdge: null,
    results: null,

    // Actions
    setNodes: (nodes) => set({ nodes }),
    setEdges: (edges) => set({ edges }),

    // React Flow Callbacks
    onNodesChange: (changes) => {
        const nodes = applyNodeChanges(changes, get().nodes);
        set({ nodes });

        // Handle Selection Sync
        const selectionChange = changes.find(c => c.type === 'select');
        if (selectionChange) {
            const node = nodes.find(n => n.id === selectionChange.id);
            set({
                selectedNode: selectionChange.selected ? node : null,
                selectedEdge: selectionChange.selected ? null : get().selectedEdge // Clear edge selection if node selected
            });
        }
    },

    onEdgesChange: (changes) => {
        const edges = applyEdgeChanges(changes, get().edges);
        set({ edges });

        // Handle Edge Selection Sync
        const selectionChange = changes.find(c => c.type === 'select');
        if (selectionChange) {
            const edge = edges.find(e => e.id === selectionChange.id);
            set({
                selectedEdge: selectionChange.selected ? edge : null,
                selectedNode: selectionChange.selected ? null : get().selectedNode // Clear node selection if edge selected
            });
        }
    },

    onConnect: (connection) => {
        const edges = get().edges;
        const newEdge = {
            ...connection,
            id: `e_${Date.now()}`,
            type: 'custom',
            data: {
                port_out: connection.sourceHandle,
                port_in: connection.targetHandle
            },
            style: { strokeWidth: 3, stroke: '#58a6ff' }
        };
        set({ edges: addEdge(newEdge, edges) });
    },

    // Proper Deletion Handlers (called by React Flow)
    onNodesDelete: (deleted) => {
        const deletedIds = new Set(deleted.map(n => n.id));
        const edges = get().edges.filter(
            e => !deletedIds.has(e.source) && !deletedIds.has(e.target)
        );
        set({
            nodes: get().nodes.filter(n => !deletedIds.has(n.id)),
            edges,
            selectedNode: null
        });
    },

    onEdgesDelete: (deleted) => {
        const deletedIds = new Set(deleted.map(e => e.id));
        set({
            edges: get().edges.filter(e => !deletedIds.has(e.id)),
            selectedEdge: null
        });
    },

    // Manual Helpers (for Context Menu etc)
    addNode: (template) => {
        const newNode = {
            id: `${template.type}_${Date.now()}`,
            type: template.type,
            position: { x: 100 + Math.random() * 200, y: 100 + Math.random() * 200 },
            data: {
                label: template.label,
                params: template.params
            }
        };
        set({ nodes: [...get().nodes, newNode] });
    },

    deleteNode: (id) => {
        // Just filter it out, onNodesDelete logic is mainly for internal RF hooks
        // But to keep consistency we manually do the same
        const edges = get().edges.filter(e => e.source !== id && e.target !== id);
        set({
            nodes: get().nodes.filter(n => n.id !== id),
            edges,
            selectedNode: null
        });
    },

    deleteEdge: (id) => {
        set({
            edges: get().edges.filter(e => e.id !== id),
            selectedEdge: null
        });
    },

    updateNodeParams: (id, params) => {
        const newNodes = get().nodes.map(n =>
            n.id === id ? { ...n, data: { ...n.data, params } } : n
        );

        // Also update selectedNode if it's the one being updated
        const selectedNode = get().selectedNode;
        const newSelectedNode = selectedNode && selectedNode.id === id
            ? { ...selectedNode, data: { ...selectedNode.data, params } }
            : selectedNode;

        set({
            nodes: newNodes,
            selectedNode: newSelectedNode
        });
    },

    selectNode: (node) => set({ selectedNode: node, selectedEdge: null }),
    selectEdge: (edge) => set({ selectedEdge: edge, selectedNode: null }),

    setResults: (results) => set({ results }),

    // Export as scenario.json format
    toScenario: () => {
        const { nodes, edges } = get();
        return {
            meta: { version: "1.0", engine: "monad_core_v1" },
            dag: {
                nodes: nodes.map(n => ({
                    id: n.id,
                    type: n.type === 'household' ? 'Household' :
                        n.type === 'firm' ? 'Firm' :
                            n.type === 'marketclearing' ? 'MarketClearing' :
                                n.type === 'fiscalauthority' ? 'FiscalAuthority' : 'CentralBank',
                    params: n.data.params,
                    pos: [n.position.x, n.position.y]
                })),
                edges: edges.map(e => ({
                    from: e.source,
                    to: e.target,
                    // Use semantic data if available, fallback to handles
                    port_out: e.data?.port_out || e.sourceHandle || 'out',
                    port_in: e.data?.port_in || e.targetHandle || 'in'
                }))
            },
            simulation: {
                shock: { target: "r_m", size: -0.01, persistence: 0.8 },
                horizon: 200
            }
        };
    }
}));

export default useGraphStore;
