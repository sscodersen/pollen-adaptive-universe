import { storageService } from './storageService';

export interface AnonymousUser {
  id: string;
  username: string;
  avatar: string;
  preferences: UserPreferences;
  joinedAt: string;
  lastActive: string;
  stats: UserStats;
}

export interface UserPreferences {
  theme: 'light' | 'dark' | 'auto';
  interests: string[];
  language: string;
  notifications: {
    feeds: boolean;
    recommendations: boolean;
    trending: boolean;
  };
  privacy: {
    analytics: boolean;
    personalization: boolean;
  };
}

export interface UserStats {
  totalSessions: number;
  totalTimeSpent: number; // in minutes
  contentViewed: number;
  searchesPerformed: number;
  itemsGenerated: number;
}

class AnonymousAuthService {
  private static instance: AnonymousAuthService;
  private currentUser: AnonymousUser | null = null;
  private sessionStartTime: number = Date.now();

  static getInstance(): AnonymousAuthService {
    if (!AnonymousAuthService.instance) {
      AnonymousAuthService.instance = new AnonymousAuthService();
    }
    return AnonymousAuthService.instance;
  }

  async initialize(): Promise<AnonymousUser> {
    // Try to load existing user
    const existingUser = await storageService.getData<AnonymousUser>('anonymous_user');
    
    if (existingUser) {
      this.currentUser = existingUser;
      this.updateLastActive();
      this.incrementSessionCount();
    } else {
      // Create new anonymous user
      this.currentUser = this.createNewUser();
      await this.saveUser();
    }

    return this.currentUser;
  }

  private createNewUser(): AnonymousUser {
    const adjectives = ['Creative', 'Smart', 'Curious', 'Brilliant', 'Innovative', 'Dynamic', 'Clever', 'Wise'];
    const nouns = ['Explorer', 'Thinker', 'Creator', 'Innovator', 'Dreamer', 'Builder', 'Pioneer', 'Visionary'];
    
    const randomAdjective = adjectives[Math.floor(Math.random() * adjectives.length)];
    const randomNoun = nouns[Math.floor(Math.random() * nouns.length)];
    const randomNumber = Math.floor(Math.random() * 1000);

    return {
      id: `anon_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      username: `${randomAdjective}${randomNoun}${randomNumber}`,
      avatar: `https://api.dicebear.com/7.x/bottts-neutral/svg?seed=${randomAdjective}${randomNoun}`,
      joinedAt: new Date().toISOString(),
      lastActive: new Date().toISOString(),
      preferences: {
        theme: 'auto',
        interests: [],
        language: 'en',
        notifications: {
          feeds: true,
          recommendations: true,
          trending: true,
        },
        privacy: {
          analytics: true,
          personalization: true,
        },
      },
      stats: {
        totalSessions: 1,
        totalTimeSpent: 0,
        contentViewed: 0,
        searchesPerformed: 0,
        itemsGenerated: 0,
      },
    };
  }

  getCurrentUser(): AnonymousUser | null {
    return this.currentUser;
  }

  async updatePreferences(preferences: Partial<UserPreferences>): Promise<void> {
    if (!this.currentUser) return;

    this.currentUser.preferences = { ...this.currentUser.preferences, ...preferences };
    await this.saveUser();
  }

  async trackActivity(activity: keyof UserStats, increment: number = 1): Promise<void> {
    if (!this.currentUser) return;

    this.currentUser.stats[activity] += increment;
    await this.saveUser();
  }

  async addInterest(interest: string): Promise<void> {
    if (!this.currentUser) return;

    if (!this.currentUser.preferences.interests.includes(interest)) {
      this.currentUser.preferences.interests.push(interest);
      await this.saveUser();
    }
  }

  async removeInterest(interest: string): Promise<void> {
    if (!this.currentUser) return;

    this.currentUser.preferences.interests = this.currentUser.preferences.interests.filter(
      i => i !== interest
    );
    await this.saveUser();
  }

  private async updateLastActive(): Promise<void> {
    if (!this.currentUser) return;

    this.currentUser.lastActive = new Date().toISOString();
    await this.saveUser();
  }

  private async incrementSessionCount(): Promise<void> {
    if (!this.currentUser) return;

    this.currentUser.stats.totalSessions += 1;
    await this.saveUser();
  }

  async endSession(): Promise<void> {
    if (!this.currentUser) return;

    const sessionDuration = Math.floor((Date.now() - this.sessionStartTime) / 1000 / 60); // minutes
    this.currentUser.stats.totalTimeSpent += sessionDuration;
    await this.saveUser();
  }

  private async saveUser(): Promise<void> {
    if (this.currentUser) {
      await storageService.setData('anonymous_user', this.currentUser);
    }
  }

  async resetUser(): Promise<AnonymousUser> {
    await storageService.removeData('anonymous_user');
    this.currentUser = this.createNewUser();
    await this.saveUser();
    return this.currentUser;
  }

  // Export user data for portability
  async exportUserData(): Promise<string> {
    if (!this.currentUser) return '';
    return JSON.stringify(this.currentUser, null, 2);
  }

  // Import user data
  async importUserData(userData: string): Promise<boolean> {
    try {
      const user = JSON.parse(userData) as AnonymousUser;
      // Validate the structure
      if (user.id && user.username && user.preferences && user.stats) {
        this.currentUser = user;
        await this.saveUser();
        return true;
      }
      return false;
    } catch {
      return false;
    }
  }
}

export const anonymousAuth = AnonymousAuthService.getInstance();