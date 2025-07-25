import React, { createContext, useContext, useReducer, useEffect, ReactNode } from 'react';
import { storageService } from '@/services/storageService';

// Enhanced interfaces
interface UserPreferences {
  theme: 'dark' | 'light' | 'auto';
  notifications: boolean;
  contentFilters: string[];
  autoRefresh: boolean;
  refreshInterval: number;
  language: string;
}

interface AppMetrics {
  totalContentGenerated: number;
  totalInteractions: number;
  lastActiveTab: string;
  sessionDuration: number;
  errorCount: number;
}

interface GlobalError {
  id: string;
  message: string;
  timestamp: Date;
  component?: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  resolved: boolean;
}

interface AppNotification {
  id: string;
  title: string;
  message: string;
  type: 'info' | 'success' | 'warning' | 'error';
  timestamp: Date;
  read: boolean;
  action?: () => void;
}

interface NetworkStatus {
  isOnline: boolean;
  connectionType: string;
  speed: 'slow' | 'medium' | 'fast';
  lastSync: Date | null;
}

interface ContentCache {
  [key: string]: {
    data: any;
    timestamp: Date;
    expiresAt: Date;
  };
}

interface Conversation {
  id: string;
  messages: Array<{
    role: 'user' | 'assistant';
    content: string;
    timestamp: Date;
  }>;
  title: string;
}

interface EnhancedAppState {
  // Existing state
  conversations: Conversation[];
  currentConversation: string | null;
  isTyping: boolean;
  uploadedFiles: Array<{
    id: string;
    name: string;
    type: string;
    size: number;
    content?: string;
  }>;
  generatedContent: Array<{
    id: string;
    type: 'blog' | 'email' | 'marketing' | 'image' | 'video';
    title: string;
    content: string;
    timestamp: Date;
  }>;

  // Enhanced state
  userPreferences: UserPreferences;
  appMetrics: AppMetrics;
  globalErrors: GlobalError[];
  notifications: AppNotification[];
  networkStatus: NetworkStatus;
  contentCache: ContentCache;
  isInitialized: boolean;
  currentTab: string;
  searchHistory: string[];
  favorites: string[];
  recentlyViewed: string[];
}

type EnhancedAppAction = 
  | { type: 'INITIALIZE_APP'; payload: Partial<EnhancedAppState> }
  | { type: 'ADD_CONVERSATION'; payload: Conversation }
  | { type: 'SET_CURRENT_CONVERSATION'; payload: string }
  | { type: 'ADD_MESSAGE'; payload: { conversationId: string; message: any } }
  | { type: 'SET_TYPING'; payload: boolean }
  | { type: 'ADD_FILE'; payload: any }
  | { type: 'REMOVE_FILE'; payload: string }
  | { type: 'ADD_GENERATED_CONTENT'; payload: any }
  | { type: 'UPDATE_USER_PREFERENCES'; payload: Partial<UserPreferences> }
  | { type: 'UPDATE_METRICS'; payload: Partial<AppMetrics> }
  | { type: 'ADD_ERROR'; payload: Omit<GlobalError, 'id'> }
  | { type: 'RESOLVE_ERROR'; payload: string }
  | { type: 'CLEAR_ERRORS' }
  | { type: 'ADD_NOTIFICATION'; payload: Omit<AppNotification, 'id'> }
  | { type: 'MARK_NOTIFICATION_READ'; payload: string }
  | { type: 'CLEAR_NOTIFICATIONS' }
  | { type: 'UPDATE_NETWORK_STATUS'; payload: Partial<NetworkStatus> }
  | { type: 'SET_CACHE'; payload: { key: string; data: any; ttl?: number } }
  | { type: 'CLEAR_CACHE'; payload?: string }
  | { type: 'SET_CURRENT_TAB'; payload: string }
  | { type: 'ADD_TO_SEARCH_HISTORY'; payload: string }
  | { type: 'ADD_TO_FAVORITES'; payload: string }
  | { type: 'REMOVE_FROM_FAVORITES'; payload: string }
  | { type: 'ADD_TO_RECENTLY_VIEWED'; payload: string };

const defaultUserPreferences: UserPreferences = {
  theme: 'dark',
  notifications: true,
  contentFilters: [],
  autoRefresh: true,
  refreshInterval: 30000,
  language: 'en'
};

const defaultAppMetrics: AppMetrics = {
  totalContentGenerated: 0,
  totalInteractions: 0,
  lastActiveTab: 'feed',
  sessionDuration: 0,
  errorCount: 0
};

const defaultNetworkStatus: NetworkStatus = {
  isOnline: navigator.onLine,
  connectionType: 'unknown',
  speed: 'medium',
  lastSync: null
};

const initialState: EnhancedAppState = {
  conversations: [],
  currentConversation: null,
  isTyping: false,
  uploadedFiles: [],
  generatedContent: [],
  userPreferences: defaultUserPreferences,
  appMetrics: defaultAppMetrics,
  globalErrors: [],
  notifications: [],
  networkStatus: defaultNetworkStatus,
  contentCache: {},
  isInitialized: false,
  currentTab: 'feed',
  searchHistory: [],
  favorites: [],
  recentlyViewed: []
};

const EnhancedAppContext = createContext<{
  state: EnhancedAppState;
  dispatch: React.Dispatch<EnhancedAppAction>;
} | null>(null);

function enhancedAppReducer(state: EnhancedAppState, action: EnhancedAppAction): EnhancedAppState {
  switch (action.type) {
    case 'INITIALIZE_APP':
      return { ...state, ...action.payload, isInitialized: true };

    case 'ADD_CONVERSATION':
      return {
        ...state,
        conversations: [...state.conversations, action.payload],
        currentConversation: action.payload.id
      };

    case 'SET_CURRENT_CONVERSATION':
      return { ...state, currentConversation: action.payload };

    case 'ADD_MESSAGE':
      return {
        ...state,
        conversations: state.conversations.map(conv =>
          conv.id === action.payload.conversationId
            ? { ...conv, messages: [...conv.messages, action.payload.message] }
            : conv
        ),
        appMetrics: {
          ...state.appMetrics,
          totalInteractions: state.appMetrics.totalInteractions + 1
        }
      };

    case 'SET_TYPING':
      return { ...state, isTyping: action.payload };

    case 'ADD_FILE':
      return { ...state, uploadedFiles: [...state.uploadedFiles, action.payload] };

    case 'REMOVE_FILE':
      return { ...state, uploadedFiles: state.uploadedFiles.filter(f => f.id !== action.payload) };

    case 'ADD_GENERATED_CONTENT':
      return { 
        ...state, 
        generatedContent: [...state.generatedContent, action.payload],
        appMetrics: {
          ...state.appMetrics,
          totalContentGenerated: state.appMetrics.totalContentGenerated + 1
        }
      };

    case 'UPDATE_USER_PREFERENCES':
      return {
        ...state,
        userPreferences: { ...state.userPreferences, ...action.payload }
      };

    case 'UPDATE_METRICS':
      return {
        ...state,
        appMetrics: { ...state.appMetrics, ...action.payload }
      };

    case 'ADD_ERROR':
      const newError: GlobalError = {
        ...action.payload,
        id: `error-${Date.now()}-${Math.random()}`,
        timestamp: new Date(),
        resolved: false
      };
      return {
        ...state,
        globalErrors: [...state.globalErrors, newError],
        appMetrics: {
          ...state.appMetrics,
          errorCount: state.appMetrics.errorCount + 1
        }
      };

    case 'RESOLVE_ERROR':
      return {
        ...state,
        globalErrors: state.globalErrors.map(error =>
          error.id === action.payload ? { ...error, resolved: true } : error
        )
      };

    case 'CLEAR_ERRORS':
      return { ...state, globalErrors: [] };

    case 'ADD_NOTIFICATION':
      const newNotification: AppNotification = {
        ...action.payload,
        id: `notification-${Date.now()}-${Math.random()}`,
        timestamp: new Date(),
        read: false
      };
      return {
        ...state,
        notifications: [...state.notifications, newNotification]
      };

    case 'MARK_NOTIFICATION_READ':
      return {
        ...state,
        notifications: state.notifications.map(notification =>
          notification.id === action.payload ? { ...notification, read: true } : notification
        )
      };

    case 'CLEAR_NOTIFICATIONS':
      return { ...state, notifications: [] };

    case 'UPDATE_NETWORK_STATUS':
      return {
        ...state,
        networkStatus: { ...state.networkStatus, ...action.payload }
      };

    case 'SET_CACHE':
      const ttl = action.payload.ttl || 5 * 60 * 1000; // 5 minutes default
      const expiresAt = new Date(Date.now() + ttl);
      return {
        ...state,
        contentCache: {
          ...state.contentCache,
          [action.payload.key]: {
            data: action.payload.data,
            timestamp: new Date(),
            expiresAt
          }
        }
      };

    case 'CLEAR_CACHE':
      if (action.payload) {
        const { [action.payload]: removed, ...rest } = state.contentCache;
        return { ...state, contentCache: rest };
      }
      return { ...state, contentCache: {} };

    case 'SET_CURRENT_TAB':
      return {
        ...state,
        currentTab: action.payload,
        appMetrics: { ...state.appMetrics, lastActiveTab: action.payload }
      };

    case 'ADD_TO_SEARCH_HISTORY':
      const newSearchHistory = [action.payload, ...state.searchHistory.filter(s => s !== action.payload)].slice(0, 20);
      return { ...state, searchHistory: newSearchHistory };

    case 'ADD_TO_FAVORITES':
      if (!state.favorites.includes(action.payload)) {
        return { ...state, favorites: [...state.favorites, action.payload] };
      }
      return state;

    case 'REMOVE_FROM_FAVORITES':
      return { ...state, favorites: state.favorites.filter(f => f !== action.payload) };

    case 'ADD_TO_RECENTLY_VIEWED':
      const newRecentlyViewed = [action.payload, ...state.recentlyViewed.filter(r => r !== action.payload)].slice(0, 50);
      return { ...state, recentlyViewed: newRecentlyViewed };

    default:
      return state;
  }
}

export const EnhancedAppProvider = ({ children }: { children: ReactNode }) => {
  const [state, dispatch] = useReducer(enhancedAppReducer, initialState);

  // Initialize app with persisted data
  useEffect(() => {
    const initializeApp = async () => {
      try {
        const savedPreferences = await storageService.getItem('userPreferences');
        const savedMetrics = await storageService.getItem('appMetrics');
        const savedSearchHistory = await storageService.getItem('searchHistory');
        const savedFavorites = await storageService.getItem('favorites');

        dispatch({
          type: 'INITIALIZE_APP',
          payload: {
            userPreferences: (savedPreferences as UserPreferences) || defaultUserPreferences,
            appMetrics: (savedMetrics as AppMetrics) || defaultAppMetrics,
            searchHistory: (savedSearchHistory as string[]) || [],
            favorites: (savedFavorites as string[]) || []
          }
        });
      } catch (error) {
        console.error('Failed to initialize app:', error);
        dispatch({
          type: 'ADD_ERROR',
          payload: {
            message: 'Failed to load user preferences',
            component: 'AppInitialization',
            severity: 'medium',
            timestamp: new Date(),
            resolved: false
          }
        });
      }
    };

    initializeApp();
  }, []);

  // Persist important data changes
  useEffect(() => {
    if (state.isInitialized) {
      storageService.setItem('userPreferences', state.userPreferences);
    }
  }, [state.userPreferences, state.isInitialized]);

  useEffect(() => {
    if (state.isInitialized) {
      storageService.setItem('appMetrics', state.appMetrics);
    }
  }, [state.appMetrics, state.isInitialized]);

  useEffect(() => {
    if (state.isInitialized) {
      storageService.setItem('searchHistory', state.searchHistory);
    }
  }, [state.searchHistory, state.isInitialized]);

  useEffect(() => {
    if (state.isInitialized) {
      storageService.setItem('favorites', state.favorites);
    }
  }, [state.favorites, state.isInitialized]);

  // Network status monitoring
  useEffect(() => {
    const handleOnline = () => {
      dispatch({
        type: 'UPDATE_NETWORK_STATUS',
        payload: { isOnline: true, lastSync: new Date() }
      });
    };

    const handleOffline = () => {
      dispatch({
        type: 'UPDATE_NETWORK_STATUS',
        payload: { isOnline: false }
      });
    };

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, []);

  // Session duration tracking
  useEffect(() => {
    const startTime = Date.now();
    const interval = setInterval(() => {
      const duration = Date.now() - startTime;
      dispatch({
        type: 'UPDATE_METRICS',
        payload: { sessionDuration: duration }
      });
    }, 60000); // Update every minute

    return () => clearInterval(interval);
  }, []);

  return (
    <EnhancedAppContext.Provider value={{ state, dispatch }}>
      {children}
    </EnhancedAppContext.Provider>
  );
};

export const useEnhancedApp = () => {
  const context = useContext(EnhancedAppContext);
  if (!context) {
    throw new Error('useEnhancedApp must be used within EnhancedAppProvider');
  }
  return context;
};

// Utility hooks for common operations
export const useAppError = () => {
  const { dispatch } = useEnhancedApp();
  
  const addError = (error: Omit<GlobalError, 'id'>) => {
    dispatch({ type: 'ADD_ERROR', payload: error });
  };
  
  const resolveError = (errorId: string) => {
    dispatch({ type: 'RESOLVE_ERROR', payload: errorId });
  };
  
  return { addError, resolveError };
};

export const useAppNotification = () => {
  const { dispatch } = useEnhancedApp();
  
  const addNotification = (notification: Omit<AppNotification, 'id'>) => {
    dispatch({ type: 'ADD_NOTIFICATION', payload: notification });
  };
  
  const markAsRead = (notificationId: string) => {
    dispatch({ type: 'MARK_NOTIFICATION_READ', payload: notificationId });
  };
  
  return { addNotification, markAsRead };
};
