import { storageService } from './storageService';

export interface CollectionItem {
  id: string;
  title: string;
  description?: string;
  imageUrl?: string;
  source?: string;
  category: string;
  addedAt: string;
  metadata?: Record<string, any>;
}

export interface Collection {
  id: string;
  name: string;
  category: 'travel' | 'food' | 'goals' | 'events' | 'shopping' | 'custom';
  items: CollectionItem[];
  icon?: string;
  color?: string;
  createdAt: string;
  updatedAt: string;
}

class CollectionsService {
  private static instance: CollectionsService;
  private collections: Collection[] = [];
  private readonly STORAGE_KEY = 'user_collections';

  static getInstance(): CollectionsService {
    if (!CollectionsService.instance) {
      CollectionsService.instance = new CollectionsService();
    }
    return CollectionsService.instance;
  }

  async initialize(): Promise<void> {
    const stored = await storageService.getData<Collection[]>(this.STORAGE_KEY);
    if (stored && Array.isArray(stored)) {
      this.collections = stored;
    } else {
      this.collections = this.getDefaultCollections();
      await this.save();
    }
  }

  private getDefaultCollections(): Collection[] {
    return [
      {
        id: 'travel',
        name: 'Travel',
        category: 'travel',
        items: [],
        icon: '✈️',
        color: '#FF6B9D',
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString()
      },
      {
        id: 'food',
        name: 'Food',
        category: 'food',
        items: [],
        icon: '🍽️',
        color: '#FFA07A',
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString()
      },
      {
        id: 'goals',
        name: 'Goals',
        category: 'goals',
        items: [],
        icon: '🎯',
        color: '#FFD700',
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString()
      },
      {
        id: 'events',
        name: 'Events',
        category: 'events',
        items: [],
        icon: '📅',
        color: '#87CEEB',
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString()
      },
      {
        id: 'shopping',
        name: 'Shopping',
        category: 'shopping',
        items: [],
        icon: '🛍️',
        color: '#DDA0DD',
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString()
      }
    ];
  }

  async getAllCollections(): Promise<Collection[]> {
    return this.collections;
  }

  async getCollection(id: string): Promise<Collection | null> {
    return this.collections.find(c => c.id === id) || null;
  }

  async addToCollection(collectionId: string, item: Omit<CollectionItem, 'id' | 'addedAt'>): Promise<void> {
    const collection = this.collections.find(c => c.id === collectionId);
    if (!collection) {
      throw new Error('Collection not found');
    }

    const newItem: CollectionItem = {
      ...item,
      id: `item_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      addedAt: new Date().toISOString()
    };

    collection.items.push(newItem);
    collection.updatedAt = new Date().toISOString();
    await this.save();

    storageService.trackEvent('collection_item_added', {
      collectionId,
      itemId: newItem.id,
      category: collection.category
    });
  }

  async removeFromCollection(collectionId: string, itemId: string): Promise<void> {
    const collection = this.collections.find(c => c.id === collectionId);
    if (!collection) {
      throw new Error('Collection not found');
    }

    collection.items = collection.items.filter(item => item.id !== itemId);
    collection.updatedAt = new Date().toISOString();
    await this.save();

    storageService.trackEvent('collection_item_removed', {
      collectionId,
      itemId
    });
  }

  async createCollection(name: string, category: Collection['category']): Promise<Collection> {
    const newCollection: Collection = {
      id: `col_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      name,
      category,
      items: [],
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString()
    };

    this.collections.push(newCollection);
    await this.save();

    storageService.trackEvent('collection_created', {
      collectionId: newCollection.id,
      category
    });

    return newCollection;
  }

  async deleteCollection(id: string): Promise<void> {
    const index = this.collections.findIndex(c => c.id === id);
    if (index === -1) {
      throw new Error('Collection not found');
    }

    this.collections.splice(index, 1);
    await this.save();

    storageService.trackEvent('collection_deleted', { collectionId: id });
  }

  async getCollectionStats(): Promise<{
    totalCollections: number;
    totalItems: number;
    itemsByCategory: Record<string, number>;
  }> {
    const totalCollections = this.collections.length;
    const totalItems = this.collections.reduce((sum, col) => sum + col.items.length, 0);
    const itemsByCategory = this.collections.reduce((acc, col) => {
      acc[col.category] = col.items.length;
      return acc;
    }, {} as Record<string, number>);

    return {
      totalCollections,
      totalItems,
      itemsByCategory
    };
  }

  async getSuggestedItems(category: string): Promise<CollectionItem[]> {
    return [];
  }

  private async save(): Promise<void> {
    await storageService.setData(this.STORAGE_KEY, this.collections);
  }

  async clearAll(): Promise<void> {
    this.collections = this.getDefaultCollections();
    await this.save();
  }
}

export const collectionsService = CollectionsService.getInstance();
