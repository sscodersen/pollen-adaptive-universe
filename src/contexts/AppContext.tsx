
import React, { createContext, useContext, useReducer, ReactNode } from 'react';

interface Conversation {
  id: string;
  messages: Array<{
    role: 'user' | 'assistant';
    content: string;
    timestamp: Date;
  }>;
  title: string;
}

interface AppState {
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
}

type AppAction = 
  | { type: 'ADD_CONVERSATION'; payload: Conversation }
  | { type: 'SET_CURRENT_CONVERSATION'; payload: string }
  | { type: 'ADD_MESSAGE'; payload: { conversationId: string; message: any } }
  | { type: 'SET_TYPING'; payload: boolean }
  | { type: 'ADD_FILE'; payload: any }
  | { type: 'REMOVE_FILE'; payload: string }
  | { type: 'ADD_GENERATED_CONTENT'; payload: any };

const initialState: AppState = {
  conversations: [],
  currentConversation: null,
  isTyping: false,
  uploadedFiles: [],
  generatedContent: []
};

const AppContext = createContext<{
  state: AppState;
  dispatch: React.Dispatch<AppAction>;
} | null>(null);

function appReducer(state: AppState, action: AppAction): AppState {
  switch (action.type) {
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
        )
      };
    case 'SET_TYPING':
      return { ...state, isTyping: action.payload };
    case 'ADD_FILE':
      return { ...state, uploadedFiles: [...state.uploadedFiles, action.payload] };
    case 'REMOVE_FILE':
      return { ...state, uploadedFiles: state.uploadedFiles.filter(f => f.id !== action.payload) };
    case 'ADD_GENERATED_CONTENT':
      return { ...state, generatedContent: [...state.generatedContent, action.payload] };
    default:
      return state;
  }
}

export const AppProvider = ({ children }: { children: ReactNode }) => {
  const [state, dispatch] = useReducer(appReducer, initialState);

  return (
    <AppContext.Provider value={{ state, dispatch }}>
      {children}
    </AppContext.Provider>
  );
};

export const useApp = () => {
  const context = useContext(AppContext);
  if (!context) {
    throw new Error('useApp must be used within AppProvider');
  }
  return context;
};
