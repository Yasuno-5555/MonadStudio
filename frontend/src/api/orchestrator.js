// API client for Orchestrator
const ORCHESTRATOR_URL = 'http://localhost:8000';

export async function runScenario(scenario) {
    const res = await fetch(`${ORCHESTRATOR_URL}/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(scenario)
    });

    if (!res.ok) {
        const error = await res.json();
        throw new Error(error.detail || 'Orchestrator error');
    }

    return await res.json();
}

export async function healthCheck() {
    const res = await fetch(`${ORCHESTRATOR_URL}/health`);
    return await res.json();
}
