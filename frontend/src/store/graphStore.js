// Zustand store for graph state
import { create } from 'zustand';

const useGraphStore = create((set, get) => ({
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
        { id: 'e1', source: 'cb_1', target: 'hh_1', sourceHandle: 'r', targetHandle: 'r_a' }
    ],

    selectedNode: null,
    selectedEdge: null,
    results: null,

    setNodes: (nodes) => set({ nodes }),
    setEdges: (edges) => set({ edges }),

    onNodesChange: (changes) => {
        // Handle position changes, selection, and removal
        let nodes = get().nodes;
        let edges = get().edges;
        let selectedNode = get().selectedNode;

        // Handle removals first
        const removals = changes.filter(c => c.type === 'remove');
        if (removals.length > 0) {
            const removeIds = new Set(removals.map(c => c.id));
            nodes = nodes.filter(n => !removeIds.has(n.id));
            edges = edges.filter(e => !removeIds.has(e.source) && !removeIds.has(e.target));
            if (selectedNode && removeIds.has(selectedNode.id)) {
                selectedNode = null;
            }
        }

        // Handle position and selection changes
        const updated = nodes.map(node => {
            const change = changes.find(c => c.id === node.id);
            if (change) {
                if (change.type === 'position' && change.position) {
                    return { ...node, position: change.position };
                }
                if (change.type === 'select') {
                    selectedNode = change.selected ? node : null;
                }
            }
            return node;
        });

        set({ nodes: updated, edges, selectedNode });
    },

    onEdgesChange: (changes) => {
        const edges = get().edges;
        const updated = edges.filter(edge => {
            const removeChange = changes.find(c => c.type === 'remove' && c.id === edge.id);
            return !removeChange;
        });
        if (updated.length !== edges.length) {
            set({ edges: updated });
        }
    },

    onConnect: (connection) => {
        const edges = get().edges;
        const newEdge = {
            id: `e${edges.length + 1}`,
            source: connection.source,
            target: connection.target,
            sourceHandle: connection.sourceHandle,
            targetHandle: connection.targetHandle
        };
        set({ edges: [...edges, newEdge] });
    },

    updateNodeParams: (id, params) => {
        set({
            nodes: get().nodes.map(n =>
                n.id === id ? { ...n, data: { ...n.data, params } } : n
            )
        });
    },

    selectNode: (node) => set({ selectedNode: node }),

    setResults: (results) => set({ results }),

    addNode: (template) => {
        const nodes = get().nodes;
        const id = `${template.type}_${Date.now()}`;
        const newNode = {
            id,
            type: template.type,
            position: { x: 100 + Math.random() * 200, y: 100 + Math.random() * 200 },
            data: {
                label: template.label,
                params: template.params
            }
        };
        set({ nodes: [...nodes, newNode] });
    },

    deleteNode: (id) => {
        const nodes = get().nodes.filter(n => n.id !== id);
        const edges = get().edges.filter(e => e.source !== id && e.target !== id);
        set({ nodes, edges, selectedNode: null });
    },

    selectEdge: (edge) => set({ selectedEdge: edge, selectedNode: null }),

    deleteEdge: (id) => {
        const edges = get().edges.filter(e => e.id !== id);
        set({ edges, selectedEdge: null });
    },

    // Export as scenario.json format
    toScenario: () => {
        const { nodes, edges } = get();
        return {
            meta: { version: "1.0", engine: "monad_core_v1" },
            dag: {
                nodes: nodes.map(n => ({
                    id: n.id,
                    type: n.type === 'household' ? 'Household' : 'CentralBank',
                    params: n.data.params,
                    pos: [n.position.x, n.position.y]
                })),
                edges: edges.map(e => ({
                    from: e.source,
                    to: e.target,
                    port_out: e.sourceHandle || 'out',
                    port_in: e.targetHandle || 'in'
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
