import { createRoot } from 'react-dom/client'
import { ThemeProvider } from 'next-themes'
import App from './App.tsx'
import './index.css'
import { initializePollenFromPreferences } from './services/pollenIntegration';
import { contentOrchestrator } from './services/contentOrchestrator';
import { platformOptimizer } from './services/platformOptimizer';
import { performanceMonitor } from './services/performanceMonitor';
import { systemHealthChecker } from './services/systemHealthCheck';
import { phase15Initializer } from './services/phase15Initializer';
import { performanceOptimizer } from './services/performanceOptimizations';

// Comprehensive global error handling
window.addEventListener('unhandledrejection', (event) => {
  console.warn('Unhandled promise rejection (prevented error boundary):', event.reason);
  performanceMonitor.markError();
  event.preventDefault(); // Prevent the error from propagating to error boundary
});

window.addEventListener('error', (event) => {
  console.warn('Global error caught:', event.error);
  performanceMonitor.markError();
  event.preventDefault();
});

// Register Service Worker for edge computing
const registerServiceWorker = async () => {
  if ('serviceWorker' in navigator) {
    try {
      const registration = await navigator.serviceWorker.register('/service-worker.js');
      console.log('🌐 Edge computing service worker registered:', registration.scope);
    } catch (error) {
      console.warn('Service worker registration failed:', error);
    }
  }
};

// Initialize comprehensive platform optimization
const initializePlatform = async () => {
  try {
    // Register edge computing service worker
    registerServiceWorker();
    
    // Start performance monitoring
    performanceMonitor.startMonitoring();
    
    // Start system health checks
    systemHealthChecker.startHealthChecks();
    
    // Subscribe to performance alerts
    performanceMonitor.subscribeToAlerts((alert) => {
      if (alert.severity === 'critical' || alert.severity === 'high') {
        console.warn(`🚨 Performance Alert [${alert.severity.toUpperCase()}]:`, alert.message);
      }
    });

    // Subscribe to health check results and auto-recovery
    systemHealthChecker.subscribe((health) => {
      if (health.overall === 'unhealthy') {
        console.warn('⚠️ System health degraded:', health.recommendations);
        // Attempt auto-recovery for critical issues
        systemHealthChecker.attemptAutoRecovery();
      }
    });

    // Initialize platform optimizer
    await platformOptimizer.autoOptimize();
    console.log('🚀 Platform optimizer initialized');
    
    // Initialize Phase 15 Enhancements
    await phase15Initializer.initialize();
    performanceOptimizer.initialize();
    
    // Initialize optional Pollen AI integration from saved preferences (non-blocking)
    initializePollenFromPreferences().catch(error => {
      console.warn('Pollen AI initialization failed (non-critical):', error);
    });
    
  } catch (error) {
    console.warn('Failed to initialize platform services:', error);
  }
};

// Pre-warm caches and keep content fresh across key sections (with error protection)
const startContentOrchestration = () => {
  setTimeout(async () => {
    try {
      // Import the optimized continuous AI generation service
      const { continuousAIGeneration } = await import('./services/continuousAIGeneration');
      
      // Start continuous generation with Pollen AI (every 15 minutes)
      continuousAIGeneration.start({
        enabled: true,
        intervalMinutes: 15,
        maxConcurrentTasks: 2,
        contentTypes: ['social', 'wellness', 'news']
      });
      
      console.log('🔄 Content orchestrator initialized with Pollen AI continuous generation');
    } catch (error) {
      console.warn('Content orchestrator failed to start (non-critical):', error);
    }
  }, 2000); // Shorter delay for faster startup
};

// Initialize all platform services
initializePlatform();
startContentOrchestration();

createRoot(document.getElementById("root")!).render(
  <ThemeProvider attribute="class" defaultTheme="light" enableSystem>
    <App />
  </ThemeProvider>
);
