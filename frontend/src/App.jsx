// Monad Studio - Main App
import NodeCanvas from './components/NodeCanvas';
import Inspector from './components/Inspector';
import ScopePanel from './components/ScopePanel';
import Toolbar from './components/Toolbar';
import './App.css';

function App() {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', width: '100vw' }}>
      <Toolbar />
      <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
        <div style={{ flex: 1, height: '100%' }}>
          <NodeCanvas />
        </div>
        <div style={{ width: '300px', display: 'flex', flexDirection: 'column', borderLeft: '1px solid #30363d', background: '#161b22' }}>
          <Inspector />
          <ScopePanel />
        </div>
      </div>
    </div>
  );
}

export default App;
