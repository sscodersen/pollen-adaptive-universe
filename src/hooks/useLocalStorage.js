import { useState, useEffect } from 'react';

export function useLocalStorage(key, initialValue) {
  const [storedValue, setStoredValue] = useState(() => {
    try {
      const item = window.localStorage.getItem(key);
      return item ? JSON.parse(item) : initialValue;
    } catch (error) {
      console.warn(`Error loading ${key} from localStorage:`, error);
      return initialValue;
    }
  });

  const setValue = (value) => {
    try {
      const valueToStore = value instanceof Function ? value(storedValue) : value;
      setStoredValue(valueToStore);
      window.localStorage.setItem(key, JSON.stringify(valueToStore));
    } catch (error) {
      console.warn(`Error saving ${key} to localStorage:`, error);
    }
  };

  return [storedValue, setValue];
}

export function useConversationHistory() {
  const [history, setHistory] = useLocalStorage('pollen_conversation_history', []);

  const addConversation = (query, response, feature) => {
    const conversation = {
      id: Date.now(),
      query,
      response,
      feature,
      timestamp: new Date().toISOString(),
    };

    setHistory((prev) => [conversation, ...prev].slice(0, 50)); // Keep last 50
  };

  const clearHistory = () => {
    setHistory([]);
  };

  const deleteConversation = (id) => {
    setHistory((prev) => prev.filter((conv) => conv.id !== id));
  };

  return { history, addConversation, clearHistory, deleteConversation };
}

export function useFavorites() {
  const [favorites, setFavorites] = useLocalStorage('pollen_favorites', []);

  const addFavorite = (query, response, feature) => {
    const favorite = {
      id: Date.now(),
      query,
      response,
      feature,
      timestamp: new Date().toISOString(),
    };

    setFavorites((prev) => [favorite, ...prev]);
  };

  const removeFavorite = (id) => {
    setFavorites((prev) => prev.filter((fav) => fav.id !== id));
  };

  const isFavorite = (query) => {
    return favorites.some((fav) => fav.query === query);
  };

  return { favorites, addFavorite, removeFavorite, isFavorite };
}

export function useUserStats() {
  const [stats, setStats] = useLocalStorage('pollen_user_stats', {
    queriesCount: 0,
    favoritesCount: 0,
    lastActive: null,
    streakDays: 0,
  });

  const incrementQueries = () => {
    setStats((prev) => ({
      ...prev,
      queriesCount: prev.queriesCount + 1,
      lastActive: new Date().toISOString(),
    }));
  };

  const updateStreak = () => {
    const today = new Date().toDateString();
    const lastActive = stats.lastActive ? new Date(stats.lastActive).toDateString() : null;

    if (lastActive !== today) {
      setStats((prev) => ({
        ...prev,
        streakDays: prev.streakDays + 1,
        lastActive: new Date().toISOString(),
      }));
    }
  };

  return { stats, incrementQueries, updateStreak };
}
