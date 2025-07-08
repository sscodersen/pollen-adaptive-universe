// Unified storage service using Local Storage + IndexedDB
interface StorageData {
  [key: string]: any;
}

interface UserPreferences {
  theme: 'light' | 'dark' | 'auto';
  notifications: boolean;
  autoplay: boolean;
  contentFilter: 'all' | 'curated' | 'strict';
  language: string;
  privacy: {
    analytics: boolean;
    recommendations: boolean;
    dataCollection: boolean;
  };
}

interface AnalyticsEvent {
  id: string;
  type: string;
  data: any;
  timestamp: number;
  sessionId: string;
}

interface UserContent {
  id: string;
  type: 'review' | 'rating' | 'comment' | 'upload';
  content: any;
  timestamp: number;
  approved: boolean;
}

class StorageService {
  private dbName = 'PollenPlatformDB';
  private dbVersion = 1;
  private db: IDBDatabase | null = null;

  async init(): Promise<void> {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open(this.dbName, this.dbVersion);
      
      request.onerror = () => reject(request.error);
      request.onsuccess = () => {
        this.db = request.result;
        resolve();
      };
      
      request.onupgradeneeded = (event) => {
        const db = (event.target as IDBOpenDBRequest).result;
        
        // Analytics store
        if (!db.objectStoreNames.contains('analytics')) {
          const analyticsStore = db.createObjectStore('analytics', { keyPath: 'id' });
          analyticsStore.createIndex('type', 'type', { unique: false });
          analyticsStore.createIndex('timestamp', 'timestamp', { unique: false });
        }
        
        // User content store
        if (!db.objectStoreNames.contains('userContent')) {
          const contentStore = db.createObjectStore('userContent', { keyPath: 'id' });
          contentStore.createIndex('type', 'type', { unique: false });
          contentStore.createIndex('approved', 'approved', { unique: false });
        }
        
        // Cache store
        if (!db.objectStoreNames.contains('cache')) {
          db.createObjectStore('cache', { keyPath: 'key' });
        }
      };
    });
  }

  // Local Storage methods
  setItem(key: string, value: any): void {
    try {
      localStorage.setItem(key, JSON.stringify(value));
    } catch (error) {
      console.error('Failed to set localStorage item:', error);
    }
  }

  getItem<T>(key: string, defaultValue?: T): T | null {
    try {
      const item = localStorage.getItem(key);
      return item ? JSON.parse(item) : defaultValue || null;
    } catch (error) {
      console.error('Failed to get localStorage item:', error);
      return defaultValue || null;
    }
  }

  removeItem(key: string): void {
    localStorage.removeItem(key);
  }

  // User preferences
  getUserPreferences(): UserPreferences {
    return this.getItem('userPreferences', {
      theme: 'dark',
      notifications: true,
      autoplay: false,
      contentFilter: 'all',
      language: 'en',
      privacy: {
        analytics: true,
        recommendations: true,
        dataCollection: false
      }
    });
  }

  setUserPreferences(preferences: UserPreferences): void {
    this.setItem('userPreferences', preferences);
  }

  // Analytics methods
  async trackEvent(type: string, data: any): Promise<void> {
    if (!this.db) await this.init();
    
    const event: AnalyticsEvent = {
      id: `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      type,
      data,
      timestamp: Date.now(),
      sessionId: this.getSessionId()
    };

    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction(['analytics'], 'readwrite');
      const store = transaction.objectStore('analytics');
      const request = store.add(event);
      
      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
    });
  }

  async getAnalytics(type?: string, limit = 1000): Promise<AnalyticsEvent[]> {
    if (!this.db) await this.init();
    
    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction(['analytics'], 'readonly');
      const store = transaction.objectStore('analytics');
      const request = type 
        ? store.index('type').getAll(type, limit)
        : store.getAll(undefined, limit);
      
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }

  // User content methods
  async saveUserContent(content: Omit<UserContent, 'id' | 'timestamp' | 'approved'>): Promise<string> {
    if (!this.db) await this.init();
    
    const userContent: UserContent = {
      ...content,
      id: `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      timestamp: Date.now(),
      approved: false
    };

    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction(['userContent'], 'readwrite');
      const store = transaction.objectStore('userContent');
      const request = store.add(userContent);
      
      request.onsuccess = () => resolve(userContent.id);
      request.onerror = () => reject(request.error);
    });
  }

  async getUserContent(type?: string): Promise<UserContent[]> {
    if (!this.db) await this.init();
    
    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction(['userContent'], 'readonly');
      const store = transaction.objectStore('userContent');
      const request = type 
        ? store.index('type').getAll(type)
        : store.getAll();
      
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }

  // Cache methods
  async setCache(key: string, value: any, ttl = 3600000): Promise<void> { // 1 hour default TTL
    if (!this.db) await this.init();
    
    const cacheItem = {
      key,
      value,
      expires: Date.now() + ttl
    };

    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction(['cache'], 'readwrite');
      const store = transaction.objectStore('cache');
      const request = store.put(cacheItem);
      
      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
    });
  }

  async getCache<T>(key: string): Promise<T | null> {
    if (!this.db) await this.init();
    
    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction(['cache'], 'readonly');
      const store = transaction.objectStore('cache');
      const request = store.get(key);
      
      request.onsuccess = () => {
        const result = request.result;
        if (result && result.expires > Date.now()) {
          resolve(result.value);
        } else {
          resolve(null);
        }
      };
      request.onerror = () => reject(request.error);
    });
  }

  // Session management
  private getSessionId(): string {
    let sessionId = sessionStorage.getItem('sessionId');
    if (!sessionId) {
      sessionId = `session-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
      sessionStorage.setItem('sessionId', sessionId);
    }
    return sessionId;
  }

  // Onboarding
  isFirstVisit(): boolean {
    return !this.getItem('hasVisited');
  }

  markAsVisited(): void {
    this.setItem('hasVisited', true);
    this.setItem('firstVisitDate', Date.now());
  }

  // Clear all data
  async clearAllData(): Promise<void> {
    localStorage.clear();
    sessionStorage.clear();
    
    if (this.db) {
      this.db.close();
      await new Promise<void>((resolve) => {
        const deleteRequest = indexedDB.deleteDatabase(this.dbName);
        deleteRequest.onsuccess = () => resolve();
        deleteRequest.onerror = () => resolve(); // Continue even if delete fails
      });
      this.db = null;
    }
  }
}

export const storageService = new StorageService();
export type { UserPreferences, AnalyticsEvent, UserContent };