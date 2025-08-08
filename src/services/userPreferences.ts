import { storageService } from './storageService';

export interface UserPreferences {
  enablePollen: boolean;
  pollenEndpoint?: string;
  enableSSE?: boolean;
  enableCrawler?: boolean;
  crawlerEndpoint?: string;
}

const PREFERENCES_KEY = 'user_preferences';

const defaultPreferences: UserPreferences = {
  enablePollen: false,
  pollenEndpoint: undefined,
  enableSSE: true,
  enableCrawler: false,
  crawlerEndpoint: undefined,
};

export const userPreferences = {
  async get(): Promise<UserPreferences> {
    const saved = await storageService.getData<UserPreferences>(PREFERENCES_KEY);
    return { ...defaultPreferences, ...(saved || {}) };
  },

  async update(partial: Partial<UserPreferences>): Promise<UserPreferences> {
    const current = await this.get();
    const updated = { ...current, ...partial };
    await storageService.setData(PREFERENCES_KEY, updated);
    return updated;
  },

  async reset(): Promise<void> {
    await storageService.setData(PREFERENCES_KEY, defaultPreferences);
  }
};
