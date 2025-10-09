import { createRoot } from 'react-dom/client'
import App from './App.tsx'
import './index.css'
import { initializePollenFromPreferences } from './services/pollenIntegration';
import { contentOrchestrator } from './services/contentOrchestrator';
import { platformOptimizer } from './services/platformOptimizer';
import { performanceMonitor } from './services/performanceMonitor';
import { systemHealthChecker } from './services/systemHealthCheck';
import { seedDataService } from './services/seedData';

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

// Initialize comprehensive platform optimization
const initializePlatform = async () => {
  try {
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
    
    // Seed demo data for better first-time user experience
    await seedDataService.seedAllData();
    console.log('✨ Demo content loaded');
    
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
  setTimeout(() => {
    try {
      // DISABLED: Continuous generation was causing infinite loops and resource consumption
      // contentOrchestrator.startContinuousGeneration([
      //   'social','shop','entertainment','games','music','news'
      // ], 300000); // Every 5 minutes
      console.log('🔄 Content orchestrator initialized (continuous generation disabled)');
    } catch (error) {
      console.warn('Content orchestrator failed to start (non-critical):', error);
    }
  }, 2000); // Shorter delay for faster startup
};

// Initialize all platform services
initializePlatform();
startContentOrchestration();

createRoot(document.getElementById("root")!).render(<App />);
