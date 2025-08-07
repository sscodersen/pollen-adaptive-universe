import { createRoot } from 'react-dom/client'
import App from './App.tsx'
import './index.css'
import { initializePollenFromPreferences } from './services/pollenIntegration';

// Initialize optional Pollen AI integration from saved preferences (non-blocking)
initializePollenFromPreferences();

createRoot(document.getElementById("root")!).render(<App />);
