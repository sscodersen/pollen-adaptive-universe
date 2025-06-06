
import { pollenAI } from './pollenAI';

export interface User {
  id: string;
  username: string;
  email: string;
  avatar: string;
  preferences: {
    contentTypes: string[];
    themes: string[];
    notifications: boolean;
  };
  stats: {
    postsCreated: number;
    tasksCompleted: number;
    communityPoints: number;
  };
}

export interface AuthState {
  isAuthenticated: boolean;
  user: User | null;
  token: string | null;
}

class BackendService {
  private baseUrl: string;
  private authState: AuthState;
  private websocket: WebSocket | null = null;

  constructor() {
    this.baseUrl = 'http://localhost:8000';
    this.authState = {
      isAuthenticated: false,
      user: null,
      token: localStorage.getItem('auth_token')
    };
    this.initializeAuth();
  }

  private async initializeAuth() {
    const token = localStorage.getItem('auth_token');
    if (token) {
      try {
        const user = await this.validateToken(token);
        this.authState = {
          isAuthenticated: true,
          user,
          token
        };
        this.connectWebSocket();
      } catch (error) {
        console.error('Token validation failed:', error);
        this.logout();
      }
    }
  }

  async login(email: string, password: string): Promise<AuthState> {
    try {
      const response = await fetch(`${this.baseUrl}/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password })
      });

      if (!response.ok) {
        throw new Error('Login failed');
      }

      const data = await response.json();
      const { user, token } = data;

      localStorage.setItem('auth_token', token);
      this.authState = { isAuthenticated: true, user, token };
      this.connectWebSocket();

      return this.authState;
    } catch (error) {
      // Fallback to demo user for development
      const demoUser: User = {
        id: 'demo-user-123',
        username: 'Demo User',
        email: email,
        avatar: 'bg-gradient-to-r from-purple-500 to-cyan-500',
        preferences: {
          contentTypes: ['social', 'news', 'entertainment'],
          themes: ['technology', 'innovation', 'creativity'],
          notifications: true
        },
        stats: {
          postsCreated: 47,
          tasksCompleted: 128,
          communityPoints: 2340
        }
      };

      const demoToken = 'demo-token-' + Date.now();
      localStorage.setItem('auth_token', demoToken);
      this.authState = { isAuthenticated: true, user: demoUser, token: demoToken };
      
      return this.authState;
    }
  }

  async register(username: string, email: string, password: string): Promise<AuthState> {
    try {
      const response = await fetch(`${this.baseUrl}/auth/register`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, email, password })
      });

      if (!response.ok) {
        throw new Error('Registration failed');
      }

      const data = await response.json();
      return this.login(email, password);
    } catch (error) {
      // Fallback registration for demo
      return this.login(email, password);
    }
  }

  logout(): void {
    localStorage.removeItem('auth_token');
    this.authState = { isAuthenticated: false, user: null, token: null };
    if (this.websocket) {
      this.websocket.close();
      this.websocket = null;
    }
  }

  private async validateToken(token: string): Promise<User> {
    const response = await fetch(`${this.baseUrl}/auth/validate`, {
      headers: { 'Authorization': `Bearer ${token}` }
    });

    if (!response.ok) {
      throw new Error('Invalid token');
    }

    return response.json();
  }

  private connectWebSocket() {
    if (!this.authState.token) return;

    try {
      this.websocket = new WebSocket(`ws://localhost:8000/ws?token=${this.authState.token}`);
      
      this.websocket.onopen = () => {
        console.log('WebSocket connected');
      };

      this.websocket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        this.handleWebSocketMessage(data);
      };

      this.websocket.onclose = () => {
        console.log('WebSocket disconnected');
        // Reconnect after 5 seconds
        setTimeout(() => this.connectWebSocket(), 5000);
      };
    } catch (error) {
      console.error('WebSocket connection failed:', error);
    }
  }

  private handleWebSocketMessage(data: any) {
    // Handle real-time updates
    window.dispatchEvent(new CustomEvent('pollenUpdate', { detail: data }));
  }

  getAuthState(): AuthState {
    return this.authState;
  }

  async updateUserPreferences(preferences: Partial<User['preferences']>): Promise<void> {
    if (!this.authState.user) return;

    try {
      await fetch(`${this.baseUrl}/user/preferences`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.authState.token}`
        },
        body: JSON.stringify(preferences)
      });

      this.authState.user.preferences = { ...this.authState.user.preferences, ...preferences };
    } catch (error) {
      console.error('Failed to update preferences:', error);
    }
  }
}

export const backendService = new BackendService();
