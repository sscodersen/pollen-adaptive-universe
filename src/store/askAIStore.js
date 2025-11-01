import { create } from 'zustand';

const useAskAIStore = create((set, get) => ({
      currentMode: 'chat',
      modes: [
        { id: 'chat', name: 'Chat', icon: 'MessageCircle', description: 'General AI conversation' },
        { id: 'code', name: 'Code', icon: 'Code', description: 'Code generation & debugging' },
        { id: 'shopping', name: 'Shopping', icon: 'ShoppingBag', description: 'Product recommendations' },
        { id: 'travel', name: 'Travel', icon: 'Plane', description: 'Trip planning' },
        { id: 'health', name: 'Health', icon: 'Heart', description: 'Health & wellness' },
        { id: 'education', name: 'Learn', icon: 'GraduationCap', description: 'Learning assistant' },
        { id: 'finance', name: 'Finance', icon: 'TrendingUp', description: 'Financial advice' },
        { id: 'image', name: 'Image', icon: 'Image', description: 'Generate images' },
        { id: 'video', name: 'Video', icon: 'Video', description: 'Generate videos' },
        { id: 'audio', name: 'Audio', icon: 'Mic', description: 'Voice & audio' },
        { id: 'music', name: 'Music', icon: 'Music', description: 'Generate music' },
        { id: 'react', name: 'ReAct', icon: 'Brain', description: 'Reasoning + Acting' },
        { id: 'automation', name: 'Automate', icon: 'Zap', description: 'Task automation' },
      ],
      
      sessions: [],
      currentSession: null,
      isStreaming: false,
      streamingData: '',
      
      mediaOutputs: [],
      taskStatus: {},
      
      setMode: (mode) => set({ currentMode: mode }),
      
      createSession: (mode) => {
        const session = {
          id: Date.now().toString(),
          mode,
          messages: [],
          createdAt: new Date().toISOString(),
        };
        set((state) => ({
          sessions: [session, ...state.sessions],
          currentSession: session.id,
        }));
        return session.id;
      },
      
      addMessage: (sessionId, message) =>
        set((state) => ({
          sessions: state.sessions.map((s) =>
            s.id === sessionId
              ? { ...s, messages: [...s.messages, message] }
              : s
          ),
        })),
      
      setStreaming: (isStreaming) => set({ isStreaming }),
      setStreamingData: (data) => set({ streamingData: data }),
      appendStreamingData: (chunk) =>
        set((state) => ({ streamingData: state.streamingData + chunk })),
      clearStreamingData: () => set({ streamingData: '' }),
      
      addMediaOutput: (output) =>
        set((state) => ({
          mediaOutputs: [output, ...state.mediaOutputs],
        })),
      
      setTaskStatus: (taskId, status) =>
        set((state) => ({
          taskStatus: { ...state.taskStatus, [taskId]: status },
        })),
      
      clearSession: (sessionId) =>
        set((state) => ({
          sessions: state.sessions.filter((s) => s.id !== sessionId),
          currentSession: state.currentSession === sessionId ? null : state.currentSession,
        })),
      
      clearAllSessions: () => set({ sessions: [], currentSession: null }),
    }));

export default useAskAIStore;
