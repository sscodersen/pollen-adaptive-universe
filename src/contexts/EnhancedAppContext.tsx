import React, { createContext, useContext, useReducer, useEffect, ReactNode } from 'react';
import { storageService } from '@/services/storageService';
import { anonymousAuth, type AnonymousUser } from '@/services/anonymousAuth';
import { personalizationEngine } from '@/services/personalizationEngine';
import { realDataIntegration } from '@/services/realDataIntegration';
import { clientAI } from '@/services/clientAI';

interface AppState {
  user: AnonymousUser | null;
  notifications: any[];
  activities: any[];
  searchHistory: string[];
  theme: 'light' | 'dark' | 'auto';
  isLoading: boolean;
  realContent: any[];
  recommendations: any[];
  aiModelsLoaded: boolean;
}

type AppAction = 
  | { type: 'SET_USER'; payload: AnonymousUser | null }
  | { type: 'SET_LOADING'; payload: boolean }
  | { type: 'ADD_NOTIFICATION'; payload: any }
  | { type: 'REMOVE_NOTIFICATION'; payload: string }
  | { type: 'ADD_ACTIVITY'; payload: any }
  | { type: 'SET_THEME'; payload: 'light' | 'dark' | 'auto' }
  | { type: 'ADD_SEARCH'; payload: string }
  | { type: 'CLEAR_SEARCH_HISTORY' }
  | { type: 'SET_REAL_CONTENT'; payload: any[] }
  | { type: 'SET_RECOMMENDATIONS'; payload: any[] }
  | { type: 'SET_AI_MODELS_LOADED'; payload: boolean }
  | { type: 'UPDATE_USER_PREFERENCES'; payload: any };

const initialState: AppState = {
  user: null,
  notifications: [],
  activities: [],
  searchHistory: [],
  theme: 'auto',
  isLoading: false,
  realContent: [],
  recommendations: [],
  aiModelsLoaded: false,
};

function appReducer(state: AppState, action: AppAction): AppState {
  switch (action.type) {
    case 'SET_USER':
      return { ...state, user: action.payload };
    case 'SET_LOADING':
      return { ...state, isLoading: action.payload };
    case 'ADD_NOTIFICATION':
      return {
        ...state,
        notifications: [...state.notifications, { ...action.payload, id: Date.now().toString() }],
      };
    case 'REMOVE_NOTIFICATION':
      return {
        ...state,
        notifications: state.notifications.filter(n => n.id !== action.payload),
      };
    case 'ADD_ACTIVITY':
      return {
        ...state,
        activities: [action.payload, ...state.activities.slice(0, 99)], // Keep last 100
      };
    case 'SET_THEME':
      return { ...state, theme: action.payload };
    case 'ADD_SEARCH':
      const filteredHistory = state.searchHistory.filter(s => s !== action.payload);
      return {
        ...state,
        searchHistory: [action.payload, ...filteredHistory].slice(0, 20), // Keep last 20
      };
    case 'CLEAR_SEARCH_HISTORY':
      return { ...state, searchHistory: [] };
    case 'SET_REAL_CONTENT':
      return { ...state, realContent: action.payload };
    case 'SET_RECOMMENDATIONS':
      return { ...state, recommendations: action.payload };
    case 'SET_AI_MODELS_LOADED':
      return { ...state, aiModelsLoaded: action.payload };
    case 'UPDATE_USER_PREFERENCES':
      return { 
        ...state, 
        user: state.user ? { ...state.user, preferences: { ...state.user.preferences, ...action.payload } } : null 
      };
    default:
      return state;
  }
}

interface AppContextType {
  state: AppState;
  dispatch: React.Dispatch<AppAction>;
  trackActivity: (activity: any) => void;
  addNotification: (notification: any) => void;
  removeNotification: (id: string) => void;
  setTheme: (theme: 'light' | 'dark' | 'auto') => void;
  addSearch: (query: string) => void;
  clearSearchHistory: () => void;
  trackBehavior: (behavior: any) => void;
  updateUserPreferences: (preferences: any) => Promise<void>;
  refreshContent: () => Promise<void>;
  generateRecommendations: (items: any[]) => Promise<any[]>;
  searchWithPersonalization: (query: string, items: any[]) => Promise<any[]>;
}

const AppContext = createContext<AppContextType | undefined>(undefined);

export function EnhancedAppProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(appReducer, initialState);

  // Initialize app on mount
  useEffect(() => {
    initializeApp();
  }, []);

  const initializeApp = async () => {
    dispatch({ type: 'SET_LOADING', payload: true });
    
    try {
      // Initialize anonymous auth
      const user = await anonymousAuth.initialize();
      dispatch({ type: 'SET_USER', payload: user });
      
      // Initialize personalization engine
      await personalizationEngine.initialize();
      
      // Initialize real data integration
      await realDataIntegration.initialize();
      
      // Load saved theme and search history
      const savedTheme = await storageService.getData('theme');
      const savedSearchHistory = await storageService.getData('searchHistory');
      
      if (savedTheme && typeof savedTheme === 'string') {
        dispatch({ type: 'SET_THEME', payload: savedTheme as 'light' | 'dark' | 'auto' });
      }
      
      if (savedSearchHistory && Array.isArray(savedSearchHistory)) {
        savedSearchHistory.forEach((search: string) => {
          dispatch({ type: 'ADD_SEARCH', payload: search });
        });
      }
      
      // Load real content
      await refreshContent();
      
      // Check AI capabilities
      const webGPUSupported = await clientAI.checkWebGPUSupport();
      if (webGPUSupported) {
        console.log('WebGPU supported - AI models will run faster');
      }
      
    } catch (error) {
      console.error('Failed to initialize app:', error);
    } finally {
      dispatch({ type: 'SET_LOADING', payload: false });
    }
  };

  // Auto-save important state changes
  useEffect(() => {
    if (state.theme !== 'auto') {
      storageService.setData('theme', state.theme);
    }
  }, [state.theme]);

  useEffect(() => {
    if (state.searchHistory.length > 0) {
      storageService.setData('searchHistory', state.searchHistory);
    }
  }, [state.searchHistory]);

  const trackActivity = (activity: any) => {
    const activityWithTimestamp = {
      ...activity,
      timestamp: new Date().toISOString(),
      id: Date.now().toString(),
    };
    dispatch({ type: 'ADD_ACTIVITY', payload: activityWithTimestamp });
  };

  const addNotification = (notification: any) => {
    dispatch({ type: 'ADD_NOTIFICATION', payload: notification });
  };

  const removeNotification = (id: string) => {
    dispatch({ type: 'REMOVE_NOTIFICATION', payload: id });
  };

  const setTheme = (theme: 'light' | 'dark' | 'auto') => {
    dispatch({ type: 'SET_THEME', payload: theme });
  };

  const addSearch = (query: string) => {
    dispatch({ type: 'ADD_SEARCH', payload: query });
  };

  const clearSearchHistory = () => {
    dispatch({ type: 'CLEAR_SEARCH_HISTORY' });
    storageService.setData('searchHistory', []);
  };

  const trackBehavior = async (behavior: any) => {
    await personalizationEngine.trackBehavior(behavior);
  };

  const updateUserPreferences = async (preferences: any) => {
    await anonymousAuth.updatePreferences(preferences);
    dispatch({ type: 'UPDATE_USER_PREFERENCES', payload: preferences });
  };

  const refreshContent = async () => {
    try {
      const content = await realDataIntegration.fetchAggregatedContent();
      dispatch({ type: 'SET_REAL_CONTENT', payload: content });
      
      // Generate personalized recommendations
      const recommendations = await personalizationEngine.generateRecommendations(content, 20);
      dispatch({ type: 'SET_RECOMMENDATIONS', payload: recommendations });
    } catch (error) {
      console.error('Failed to refresh content:', error);
    }
  };

  const generateRecommendations = async (items: any[]) => {
    return await personalizationEngine.generateRecommendations(items);
  };

  const searchWithPersonalization = async (query: string, items: any[]) => {
    return await personalizationEngine.personalizedSearch(query, items);
  };

  const value: AppContextType = {
    state,
    dispatch,
    trackActivity,
    addNotification,
    removeNotification,
    setTheme,
    addSearch,
    clearSearchHistory,
    trackBehavior,
    updateUserPreferences,
    refreshContent,
    generateRecommendations,
    searchWithPersonalization,
  };

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
}

export function useEnhancedApp() {
  const context = useContext(AppContext);
  if (context === undefined) {
    throw new Error('useEnhancedApp must be used within an EnhancedAppProvider');
  }
  return context;
}