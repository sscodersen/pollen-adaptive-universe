import { createRoot } from 'react-dom/client'
import App from './App.tsx'
import './index.css'
import { initializePollenFromPreferences } from './services/pollenIntegration';
import { contentOrchestrator } from './services/contentOrchestrator';

// Initialize optional Pollen AI integration from saved preferences (non-blocking)
initializePollenFromPreferences();

// Pre-warm caches and keep content fresh across key sections
contentOrchestrator.startContinuousGeneration(['social','shop','entertainment','games','music','news'], 60000);

createRoot(document.getElementById("root")!).render(<App />);
