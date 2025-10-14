/**
 * Enhanced Shop Features - Phase 15
 * Price alerts, wishlist management, and GEO optimization
 */

import { loggingService } from './loggingService';
import { Product } from '../types/shop';

export interface PriceAlert {
  id: string;
  productId: string;
  productName: string;
  targetPrice: number;
  currentPrice: number;
  active: boolean;
  createdAt: string;
  triggeredAt?: string;
}

export interface WishlistItem {
  id: string;
  product: Product;
  addedAt: string;
  priceHistory: { price: number; date: string }[];
  notes?: string;
}

class ShopEnhancementsService {
  private priceAlerts: Map<string, PriceAlert> = new Map();
  private wishlist: Map<string, WishlistItem> = new Map();
  private checkInterval: NodeJS.Timeout | null = null;

  constructor() {
    this.loadFromStorage();
    this.startPriceMonitoring();
  }

  // Price Alerts
  createPriceAlert(product: Product, targetPrice: number): PriceAlert {
    const alert: PriceAlert = {
      id: `alert-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      productId: product.id,
      productName: product.name,
      targetPrice,
      currentPrice: parseFloat(product.price.replace('$', '')),
      active: true,
      createdAt: new Date().toISOString()
    };

    this.priceAlerts.set(alert.id, alert);
    this.saveToStorage();

    loggingService.logUserInteraction('create_price_alert', 'shop', {
      productId: product.id,
      targetPrice,
      currentPrice: alert.currentPrice
    });

    return alert;
  }

  getPriceAlerts(): PriceAlert[] {
    return Array.from(this.priceAlerts.values());
  }

  removePriceAlert(alertId: string): void {
    this.priceAlerts.delete(alertId);
    this.saveToStorage();
    loggingService.logUserInteraction('remove_price_alert', 'shop', { alertId });
  }

  private startPriceMonitoring(): void {
    // Check prices every 5 minutes
    this.checkInterval = setInterval(() => {
      this.checkPriceAlerts();
    }, 5 * 60 * 1000);
  }

  private async checkPriceAlerts(): Promise<void> {
    const activeAlerts = Array.from(this.priceAlerts.values()).filter(a => a.active);
    
    for (const alert of activeAlerts) {
      // In a real app, fetch current price from API
      const simulatedCurrentPrice = alert.currentPrice + (Math.random() - 0.5) * 20;
      
      if (simulatedCurrentPrice <= alert.targetPrice) {
        this.triggerPriceAlert(alert, simulatedCurrentPrice);
      }
    }
  }

  private triggerPriceAlert(alert: PriceAlert, currentPrice: number): void {
    alert.active = false;
    alert.triggeredAt = new Date().toISOString();
    
    loggingService.log('info', 'user_interaction', 
      `Price alert triggered for ${alert.productName}`, {
        targetPrice: alert.targetPrice,
        currentPrice,
        savings: alert.targetPrice - currentPrice
      });

    // Show notification
    if ('Notification' in window && Notification.permission === 'granted') {
      new Notification('Price Alert!', {
        body: `${alert.productName} is now $${currentPrice.toFixed(2)} (Target: $${alert.targetPrice})`,
        icon: '/shop-icon.png'
      });
    }

    this.saveToStorage();
  }

  // Wishlist Management
  addToWishlist(product: Product, notes?: string): WishlistItem {
    const existing = Array.from(this.wishlist.values()).find(
      item => item.product.id === product.id
    );

    if (existing) {
      existing.notes = notes || existing.notes;
      this.saveToStorage();
      return existing;
    }

    const item: WishlistItem = {
      id: `wish-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      product,
      addedAt: new Date().toISOString(),
      priceHistory: [{
        price: parseFloat(product.price.replace('$', '')),
        date: new Date().toISOString()
      }],
      notes
    };

    this.wishlist.set(item.id, item);
    this.saveToStorage();

    loggingService.logUserInteraction('add_to_wishlist', 'shop', {
      productId: product.id,
      productName: product.name
    });

    return item;
  }

  removeFromWishlist(itemId: string): void {
    this.wishlist.delete(itemId);
    this.saveToStorage();
    loggingService.logUserInteraction('remove_from_wishlist', 'shop', { itemId });
  }

  getWishlist(): WishlistItem[] {
    return Array.from(this.wishlist.values());
  }

  updatePriceHistory(productId: string, newPrice: number): void {
    const item = Array.from(this.wishlist.values()).find(
      i => i.product.id === productId
    );

    if (item) {
      item.priceHistory.push({
        price: newPrice,
        date: new Date().toISOString()
      });

      // Keep only last 30 days of history
      if (item.priceHistory.length > 30) {
        item.priceHistory = item.priceHistory.slice(-30);
      }

      this.saveToStorage();
    }
  }

  getPriceDrops(): WishlistItem[] {
    return Array.from(this.wishlist.values()).filter(item => {
      if (item.priceHistory.length < 2) return false;
      
      const currentPrice = item.priceHistory[item.priceHistory.length - 1].price;
      const previousPrice = item.priceHistory[item.priceHistory.length - 2].price;
      
      return currentPrice < previousPrice;
    });
  }

  // GEO Optimization Helpers
  generateGEOMetadata(product: Product): Record<string, any> {
    return {
      '@context': 'https://schema.org',
      '@type': 'Product',
      name: product.name,
      description: product.description,
      brand: product.brand,
      offers: {
        '@type': 'Offer',
        price: product.price.replace('$', ''),
        priceCurrency: 'USD',
        availability: product.inStock ? 'InStock' : 'OutOfStock',
        url: product.link
      },
      aggregateRating: {
        '@type': 'AggregateRating',
        ratingValue: product.rating,
        reviewCount: product.reviews
      },
      category: product.category,
      keywords: product.tags.join(', ')
    };
  }

  // Storage
  private saveToStorage(): void {
    try {
      localStorage.setItem('price_alerts', JSON.stringify(Array.from(this.priceAlerts.entries())));
      localStorage.setItem('wishlist', JSON.stringify(Array.from(this.wishlist.entries())));
    } catch (error) {
      console.error('Failed to save shop data:', error);
    }
  }

  private loadFromStorage(): void {
    try {
      const alertsData = localStorage.getItem('price_alerts');
      if (alertsData) {
        this.priceAlerts = new Map(JSON.parse(alertsData));
      }

      const wishlistData = localStorage.getItem('wishlist');
      if (wishlistData) {
        this.wishlist = new Map(JSON.parse(wishlistData));
      }
    } catch (error) {
      console.error('Failed to load shop data:', error);
    }
  }

  cleanup(): void {
    if (this.checkInterval) {
      clearInterval(this.checkInterval);
    }
  }
}

export const shopEnhancements = new ShopEnhancementsService();
