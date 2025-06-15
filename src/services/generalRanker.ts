
/**
 * General purpose item ranking utility.
 * Can rank shop products, news articles, entertainment, etc.
 * 
 * Usage:
 *   rankItems(items, { type: "shop" }) // returns sorted array.
 */

type ShopItem = {
  id: string;
  significance?: number;
  trending?: boolean;
  discount?: number;
  rating?: number;
  price?: string;
  quality?: number;
  views?: number;
  impact?: string;
};

type NewsItem = {
  id: string;
  significance?: number;
  trending?: boolean;
  date?: string;
  rating?: number;
  views?: number;
  impact?: string;
};

type RankType = "shop" | "news" | "entertainment" | "social";

interface RankOptions {
  type: RankType;
  sortBy?: string; // e.g., "significance", "rating", etc.
  prioritizeQuality?: boolean;
}

export function rankItems<T extends object>(items: T[], options: RankOptions): T[] {
  if (!items || !items.length) return [];

  switch (options.type) {
    case "shop": {
      return [...items].sort((a: any, b: any) => {
        // Multi-factor ranking for shop items
        
        // 1. Premium/High impact items first
        if (a.impact === 'premium' && b.impact !== 'premium') return -1;
        if (b.impact === 'premium' && a.impact !== 'premium') return 1;
        
        // 2. Trending items get priority
        if (a.trending !== b.trending) return b.trending - a.trending;
        
        // 3. Significance score (weighted)
        const aSignificance = (a.significance ?? 0) * 0.4;
        const bSignificance = (b.significance ?? 0) * 0.4;
        
        // 4. Quality score (weighted)
        const aQuality = (a.quality ?? 0) * 0.3;
        const bQuality = (b.quality ?? 0) * 0.3;
        
        // 5. Rating (weighted)
        const aRating = (a.rating ?? 0) * 0.2;
        const bRating = (b.rating ?? 0) * 0.2;
        
        // 6. Views (weighted)
        const aViews = (a.views ?? 0) / 1000 * 0.1; // Normalize views
        const bViews = (b.views ?? 0) / 1000 * 0.1;
        
        const aScore = aSignificance + aQuality + aRating + aViews;
        const bScore = bSignificance + bQuality + bRating + bViews;
        
        return bScore - aScore;
      });
    }
    case "social": 
    case "news": {
      return [...items].sort((a: any, b: any) => {
        // 1. Critical/High impact first
        if (a.impact === 'critical' && b.impact !== 'critical') return -1;
        if (b.impact === 'critical' && a.impact !== 'critical') return 1;
        
        // 2. Trending priority
        if (a.trending !== b.trending) return b.trending - a.trending;
        
        // 3. Significance weighted heavily
        const aSignificance = (a.significance ?? 0) * 0.5;
        const bSignificance = (b.significance ?? 0) * 0.5;
        
        // 4. Views for engagement
        const aViews = (a.views ?? 0) / 1000 * 0.3;
        const bViews = (b.views ?? 0) / 1000 * 0.3;
        
        // 5. Quality
        const aQuality = (a.quality ?? 0) * 0.2;
        const bQuality = (b.quality ?? 0) * 0.2;
        
        const aScore = aSignificance + aViews + aQuality;
        const bScore = bSignificance + bViews + bQuality;
        
        return bScore - aScore;
      });
    }
    case "entertainment": {
      return [...items].sort((a: any, b: any) => {
        // Entertainment focuses on engagement and quality
        if (a.trending !== b.trending) return b.trending - a.trending;
        
        const aScore = (a.significance ?? 0) * 0.3 + (a.views ?? 0) / 1000 * 0.4 + (a.rating ?? 0) * 0.3;
        const bScore = (b.significance ?? 0) * 0.3 + (b.views ?? 0) / 1000 * 0.4 + (b.rating ?? 0) * 0.3;
        
        return bScore - aScore;
      });
    }
    default:
      return items;
  }
}
