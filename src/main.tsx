import { createRoot } from 'react-dom/client'
import App from './App.tsx'
import './index.css'
import { initializePollenFromPreferences } from './services/pollenIntegration';
import { contentOrchestrator } from './services/contentOrchestrator';
import { platformOptimizer } from './services/platformOptimizer';

// Global error handling for unhandled promise rejections
window.addEventListener('unhandledrejection', (event) => {
  console.warn('Unhandled promise rejection (prevented error boundary):', event.reason);
  event.preventDefault(); // Prevent the error from propagating to error boundary
});

// Initialize platform optimization
platformOptimizer.autoOptimize().then(() => {
  console.log('🚀 Platform optimizer initialized');
}).catch(console.warn);

// Initialize optional Pollen AI integration from saved preferences (non-blocking)
initializePollenFromPreferences().catch(error => {
  console.warn('Pollen AI initialization failed (non-critical):', error);
});

// Pre-warm caches and keep content fresh across key sections (with error protection)
setTimeout(() => {
  try {
    contentOrchestrator.startContinuousGeneration(['social','shop','entertainment','games','music','news'], 60000);
    console.log('🔄 Content orchestrator started');
  } catch (error) {
    console.warn('Content orchestrator failed to start (non-critical):', error);
  }
}, 1000); // Delay to allow app to initialize first

createRoot(document.getElementById("root")!).render(<App />);
