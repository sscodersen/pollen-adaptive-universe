import { contentOrchestrator } from './contentOrchestrator';
import { userPreferences } from './userPreferences';

export async function initializePollenFromPreferences(): Promise<void> {
  try {
    const prefs = await userPreferences.get();
    if (prefs.enablePollen && prefs.pollenEndpoint) {
      await contentOrchestrator.setupBackendIntegration({
        pollenEndpoint: prefs.pollenEndpoint,
        enableSSE: prefs.enableSSE,
      });
    }
  } catch (e) {
    console.warn('Pollen initialization skipped:', e);
  }
}
