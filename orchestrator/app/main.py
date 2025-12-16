"""FastAPI entry point for Monad Orchestrator."""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .models import Scenario
from .executor import run_scenario

app = FastAPI(
    title="Monad Orchestrator",
    version="0.1.0",
    description="Python Orchestrator for Monad HANK Engine"
)

# CORS for Electron
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok", "engine": "monad_core"}

@app.post("/run")
def run(scenario: Scenario):
    """
    Execute a scenario.
    
    Expects JSON matching the .monad file format.
    """
    try:
        result = run_scenario(scenario)
        return {
            "status": "ok",
            "results": result
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Engine error: {str(e)}")
