// Smart Shop Feed Algorithm - Similar to Social Feed but optimized for products
import { significanceAlgorithm } from './significanceAlgorithm';
import { contentOrchestrator } from './contentOrchestrator';
import { Product } from '../types/shop';

export interface ShopFeedStrategy {
  diversityBoost: number;      // 0-1: Preference for diverse product categories
  discountWeight: number;      // 0-1: How much to prioritize discounted items
  trendingBoost: number;       // 0-1: Boost for trending products
  qualityThreshold: number;    // 0-10: Minimum quality score
  personalizedWeight: number;  // 0-1: Future: personalization based on user behavior
  freshnessPriority: number;   // 0-1: Preference for newly added products
}

export interface ShopFeedFilters {
  category?: string;
  priceRange?: { min: number; max: number };
  rating?: number;
  trending?: boolean;
  inStock?: boolean;
  discount?: boolean;
}

export interface ShopFeedResponse {
  products: Product[];
  metadata: {
    totalProducts: number;
    featuredCount: number;
    trendingCount: number;
    discountedCount: number;
    avgSignificance: number;
    strategy: ShopFeedStrategy;
    timestamp: string;
  };
}

class SmartShopFeed {
  private defaultStrategy: ShopFeedStrategy = {
    diversityBoost: 0.7,
    discountWeight: 0.6,
    trendingBoost: 0.8,
    qualityThreshold: 7.0,
    personalizedWeight: 0.5,
    freshnessPriority: 0.4
  };

  // Main feed generation - like social feed but for products
  async generateProductFeed(
    count: number = 20,
    filters?: ShopFeedFilters,
    strategy?: Partial<ShopFeedStrategy>
  ): Promise<ShopFeedResponse> {
    const effectiveStrategy = { ...this.defaultStrategy, ...strategy };
    
    // Generate products using content orchestrator
    const { content: rawProducts } = await contentOrchestrator.generateContent({
      type: 'shop',
      count: count * 2, // Generate extra for better filtering
      strategy: {
        diversity: effectiveStrategy.diversityBoost,
        freshness: effectiveStrategy.freshnessPriority,
        personalization: effectiveStrategy.personalizedWeight,
        qualityThreshold: effectiveStrategy.qualityThreshold,
        trendingBoost: effectiveStrategy.trendingBoost
      }
    });

    // Convert and enhance products
    let products = rawProducts.map(item => this.enhanceProduct(item as any));

    // Apply filters
    if (filters) {
      products = this.applyFilters(products, filters);
    }

    // Apply feed algorithm ranking
    products = this.rankProductsForFeed(products, effectiveStrategy);

    // Ensure diversity in categories (feed-like algorithm)
    products = this.ensureCategoryDiversity(products, effectiveStrategy.diversityBoost);

    // Apply final selection
    const finalProducts = products.slice(0, count);

    return {
      products: finalProducts,
      metadata: {
        totalProducts: products.length,
        featuredCount: finalProducts.filter(p => p.significance > 8.5).length,
        trendingCount: finalProducts.filter(p => p.trending).length,
        discountedCount: finalProducts.filter(p => p.discount && p.discount > 0).length,
        avgSignificance: this.calculateAverageSignificance(finalProducts),
        strategy: effectiveStrategy,
        timestamp: new Date().toISOString()
      }
    };
  }

  // Feed-style ranking algorithm for products
  private rankProductsForFeed(products: Product[], strategy: ShopFeedStrategy): Product[] {
    return products.sort((a, b) => {
      // 1. Premium/Featured products first (like pinned posts)
      if (a.significance > 9 && b.significance <= 9) return -1;
      if (b.significance > 9 && a.significance <= 9) return 1;

      // 2. Trending boost (like viral posts)
      const aTrendingScore = a.trending ? strategy.trendingBoost : 0;
      const bTrendingScore = b.trending ? strategy.trendingBoost : 0;

      // 3. Discount appeal (special promotions)
      const aDiscountScore = (a.discount || 0) * strategy.discountWeight / 100;
      const bDiscountScore = (b.discount || 0) * strategy.discountWeight / 100;

      // 4. Base significance (like post engagement)
      const aSignificance = a.significance * 0.4;
      const bSignificance = b.significance * 0.4;

      // 5. Quality and ratings (like post quality)
      const aQualityScore = a.rating * 0.15;
      const bQualityScore = b.rating * 0.15;

      // 6. Stock availability (live updates)
      const aStockBonus = a.inStock ? 0.1 : -0.5;
      const bStockBonus = b.inStock ? 0.1 : -0.5;

      // 7. Views/popularity (like post views)
      const aViewsScore = (a.views || 0) / 10000 * 0.05;
      const bViewsScore = (b.views || 0) / 10000 * 0.05;

      const aFinalScore = aSignificance + aTrendingScore + aDiscountScore + 
                         aQualityScore + aStockBonus + aViewsScore;
      const bFinalScore = bSignificance + bTrendingScore + bDiscountScore + 
                         bQualityScore + bStockBonus + bViewsScore;

      return bFinalScore - aFinalScore;
    });
  }

  // Ensure category diversity like social feed ensures content diversity
  private ensureCategoryDiversity(products: Product[], diversityBoost: number): Product[] {
    if (diversityBoost < 0.5) return products;

    const categorized: { [key: string]: Product[] } = {};
    const result: Product[] = [];
    
    // Group by category
    products.forEach(product => {
      if (!categorized[product.category]) {
        categorized[product.category] = [];
      }
      categorized[product.category].push(product);
    });

    // Distribute products across categories (round-robin style)
    const categories = Object.keys(categorized);
    let maxRounds = Math.max(...Object.values(categorized).map(arr => arr.length));
    
    for (let round = 0; round < maxRounds; round++) {
      for (const category of categories) {
        if (categorized[category][round]) {
          result.push(categorized[category][round]);
        }
      }
    }

    return result;
  }

  private enhanceProduct(product: any): Product {
    // Enhance product with feed-specific metadata
    const enhancedProduct: Product = {
      ...product,
      views: product.views || Math.floor(Math.random() * 10000) + 100,
      rank: 0, // Will be set after ranking
      quality: Math.floor(product.significance * 10) / 10,
      impact: product.significance > 8.5 ? 'premium' : 
              product.significance > 7.5 ? 'high' : 
              product.significance > 6.5 ? 'medium' : 'low'
    };

    return enhancedProduct;
  }

  private applyFilters(products: Product[], filters: ShopFeedFilters): Product[] {
    return products.filter(product => {
      if (filters.category && product.category !== filters.category) return false;
      if (filters.trending !== undefined && product.trending !== filters.trending) return false;
      if (filters.inStock !== undefined && product.inStock !== filters.inStock) return false;
      if (filters.discount && (!product.discount || product.discount <= 0)) return false;
      if (filters.rating && product.rating < filters.rating) return false;
      
      if (filters.priceRange) {
        const price = parseFloat(product.price.replace('$', ''));
        if (price < filters.priceRange.min || price > filters.priceRange.max) return false;
      }
      
      return true;
    });
  }

  private calculateAverageSignificance(products: Product[]): number {
    if (products.length === 0) return 0;
    const sum = products.reduce((acc, product) => acc + product.significance, 0);
    return Math.round((sum / products.length) * 100) / 100;
  }

  // Get trending products (like trending posts)
  async getTrendingProducts(count: number = 10): Promise<Product[]> {
    const response = await this.generateProductFeed(count * 2, { trending: true }, {
      trendingBoost: 1.0,
      qualityThreshold: 8.0
    });
    
    return response.products.slice(0, count);
  }

  // Get personalized recommendations (future feature)
  async getPersonalizedProducts(
    userId: string, 
    count: number = 15,
    userPreferences?: any
  ): Promise<Product[]> {
    // This would use user behavior in the future
    const response = await this.generateProductFeed(count, undefined, {
      personalizedWeight: 0.8,
      diversityBoost: 0.6
    });
    
    return response.products;
  }

  // Get featured/premium products (like pinned posts)
  async getFeaturedProducts(count: number = 5): Promise<Product[]> {
    const response = await this.generateProductFeed(count * 2, undefined, {
      qualityThreshold: 9.0,
      trendingBoost: 0.9
    });
    
    return response.products
      .filter(p => p.significance > 8.5)
      .slice(0, count);
  }
}

export const smartShopFeed = new SmartShopFeed();