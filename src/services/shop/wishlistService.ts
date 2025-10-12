import { personalizationEngine } from '../personalizationEngine';
import { Product } from '@/types/shop';

export interface WishlistItem {
  product: Product;
  addedAt: string;
  priceWhenAdded: number;
  currentPrice: number;
  priceDropAlert: boolean;
  notes?: string;
}

export interface PriceAlert {
  id: string;
  productId: string;
  productName: string;
  targetPrice: number;
  currentPrice: number;
  triggered: boolean;
  createdAt: string;
}

class WishlistService {
  private readonly WISHLIST_KEY = 'shop_wishlist';
  private readonly ALERTS_KEY = 'price_alerts';

  addToWishlist(product: Product, notes?: string): void {
    const wishlist = this.getWishlist();
    
    if (wishlist.some(item => item.product.id === product.id)) {
      return;
    }

    const currentPrice = this.extractPrice(product.price);
    
    const wishlistItem: WishlistItem = {
      product,
      addedAt: new Date().toISOString(),
      priceWhenAdded: currentPrice,
      currentPrice,
      priceDropAlert: true,
      notes
    };

    wishlist.push(wishlistItem);
    localStorage.setItem(this.WISHLIST_KEY, JSON.stringify(wishlist));

    personalizationEngine.trackBehavior({
      action: 'save',
      contentId: product.id,
      contentType: 'shop',
      metadata: { category: product.category, price: currentPrice }
    });
  }

  removeFromWishlist(productId: string): void {
    const wishlist = this.getWishlist();
    const filtered = wishlist.filter(item => item.product.id !== productId);
    localStorage.setItem(this.WISHLIST_KEY, JSON.stringify(filtered));
  }

  getWishlist(): WishlistItem[] {
    try {
      const stored = localStorage.getItem(this.WISHLIST_KEY);
      return stored ? JSON.parse(stored) : [];
    } catch {
      return [];
    }
  }

  isInWishlist(productId: string): boolean {
    return this.getWishlist().some(item => item.product.id === productId);
  }

  updatePrice(productId: string, newPrice: number): void {
    const wishlist = this.getWishlist();
    const item = wishlist.find(w => w.product.id === productId);

    if (!item) return;

    const oldPrice = item.currentPrice;
    item.currentPrice = newPrice;

    if (item.priceDropAlert && newPrice < oldPrice) {
      this.createPriceAlert(item.product, oldPrice, newPrice);
    }

    localStorage.setItem(this.WISHLIST_KEY, JSON.stringify(wishlist));
  }

  createPriceAlert(product: Product, oldPrice: number, newPrice: number): void {
    const alerts = this.getPriceAlerts();
    
    const alert: PriceAlert = {
      id: `alert_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      productId: product.id,
      productName: product.name,
      targetPrice: oldPrice,
      currentPrice: newPrice,
      triggered: true,
      createdAt: new Date().toISOString()
    };

    alerts.push(alert);
    localStorage.setItem(this.ALERTS_KEY, JSON.stringify(alerts));
  }

  getPriceAlerts(): PriceAlert[] {
    try {
      const stored = localStorage.getItem(this.ALERTS_KEY);
      return stored ? JSON.parse(stored) : [];
    } catch {
      return [];
    }
  }

  getActiveAlerts(): PriceAlert[] {
    return this.getPriceAlerts().filter(alert => alert.triggered);
  }

  dismissAlert(alertId: string): void {
    const alerts = this.getPriceAlerts();
    const alert = alerts.find(a => a.id === alertId);
    if (alert) {
      alert.triggered = false;
      localStorage.setItem(this.ALERTS_KEY, JSON.stringify(alerts));
    }
  }

  getSimilarProducts(product: Product, allProducts: Product[]): Product[] {
    return allProducts
      .filter(p => p.id !== product.id)
      .filter(p => 
        p.category === product.category ||
        p.tags?.some(tag => product.tags?.includes(tag))
      )
      .sort((a, b) => {
        const aScore = this.calculateSimilarityScore(product, a);
        const bScore = this.calculateSimilarityScore(product, b);
        return bScore - aScore;
      })
      .slice(0, 6);
  }

  private calculateSimilarityScore(productA: Product, productB: Product): number {
    let score = 0;

    if (productA.category === productB.category) score += 3;
    if (productA.brand === productB.brand) score += 2;

    const sharedTags = productA.tags?.filter(tag => 
      productB.tags?.includes(tag)
    ).length || 0;
    score += sharedTags;

    const priceA = this.extractPrice(productA.price);
    const priceB = this.extractPrice(productB.price);
    const priceDiff = Math.abs(priceA - priceB) / priceA;
    if (priceDiff < 0.3) score += 2;

    return score;
  }

  private extractPrice(priceString: string): number {
    const price = parseFloat(priceString.replace('$', '').replace(',', ''));
    return isNaN(price) ? 0 : price;
  }

  updateNotes(productId: string, notes: string): void {
    const wishlist = this.getWishlist();
    const item = wishlist.find(w => w.product.id === productId);

    if (!item) return;

    item.notes = notes;
    localStorage.setItem(this.WISHLIST_KEY, JSON.stringify(wishlist));
  }

  getPriceDrop(item: WishlistItem): number {
    return item.priceWhenAdded - item.currentPrice;
  }

  getPriceDropPercentage(item: WishlistItem): number {
    if (item.priceWhenAdded === 0) return 0;
    return ((item.priceWhenAdded - item.currentPrice) / item.priceWhenAdded) * 100;
  }
}

export const wishlistService = new WishlistService();
