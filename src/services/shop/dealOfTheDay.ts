import { Product } from '@/types/shop';
import { contentOrchestrator } from '../contentOrchestrator';

export interface DealOfTheDay {
  product: Product;
  originalPrice: number;
  dealPrice: number;
  savings: number;
  savingsPercentage: number;
  expiresAt: string;
  claimed: number;
  available: number;
  featured: boolean;
}

class DealOfTheDayService {
  private readonly DEALS_KEY = 'daily_deals';

  async generateDailyDeals(count: number = 3): Promise<DealOfTheDay[]> {
    const today = new Date().toDateString();
    const cached = this.getCachedDeals(today);
    
    if (cached && cached.length > 0) {
      return cached;
    }

    const response = await contentOrchestrator.generateContent({
      type: 'shop',
      count: count * 2,
      strategy: {
        diversity: 0.9,
        freshness: 1.0,
        personalization: 0.7,
        qualityThreshold: 8.0,
        trendingBoost: 0.9
      }
    });

    const deals: DealOfTheDay[] = response.content
      .slice(0, count)
      .map((product: any) => {
        const originalPrice = this.extractPrice(product.price);
        const discountPercentage = 20 + Math.random() * 50;
        const dealPrice = originalPrice * (1 - discountPercentage / 100);
        const savings = originalPrice - dealPrice;

        const expiresAt = new Date();
        expiresAt.setHours(23, 59, 59, 999);

        return {
          product,
          originalPrice,
          dealPrice,
          savings,
          savingsPercentage: discountPercentage,
          expiresAt: expiresAt.toISOString(),
          claimed: Math.floor(Math.random() * 50),
          available: 100,
          featured: Math.random() > 0.5
        };
      });

    this.cacheDeals(today, deals);
    return deals;
  }

  private extractPrice(priceString: string): number {
    const price = parseFloat(priceString.replace('$', '').replace(',', ''));
    return isNaN(price) ? 99.99 : price;
  }

  private getCachedDeals(date: string): DealOfTheDay[] | null {
    try {
      const stored = localStorage.getItem(this.DEALS_KEY);
      if (!stored) return null;

      const cache = JSON.parse(stored);
      if (cache.date === date) {
        return cache.deals;
      }
      return null;
    } catch {
      return null;
    }
  }

  private cacheDeals(date: string, deals: DealOfTheDay[]): void {
    const cache = { date, deals };
    localStorage.setItem(this.DEALS_KEY, JSON.stringify(cache));
  }

  getTimeRemaining(deal: DealOfTheDay): { hours: number; minutes: number; seconds: number } {
    const now = new Date().getTime();
    const expiry = new Date(deal.expiresAt).getTime();
    const diff = expiry - now;

    if (diff <= 0) {
      return { hours: 0, minutes: 0, seconds: 0 };
    }

    return {
      hours: Math.floor(diff / (1000 * 60 * 60)),
      minutes: Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60)),
      seconds: Math.floor((diff % (1000 * 60)) / 1000)
    };
  }

  claimDeal(dealId: string): void {
    const today = new Date().toDateString();
    const deals = this.getCachedDeals(today);
    
    if (!deals) return;

    const deal = deals.find(d => d.product.id === dealId);
    if (deal && deal.claimed < deal.available) {
      deal.claimed++;
      this.cacheDeals(today, deals);
    }
  }
}

export const dealOfTheDayService = new DealOfTheDayService();
