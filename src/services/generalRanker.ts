
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
};

type NewsItem = {
  id: string;
  significance?: number;
  trending?: boolean;
  date?: string;
  rating?: number;
};

type RankType = "shop" | "news" | "entertainment";

interface RankOptions {
  type: RankType;
  sortBy?: string; // e.g., "significance", "rating", etc.
}

export function rankItems<T extends object>(items: T[], options: RankOptions): T[] {
  if (!items || !items.length) return [];

  switch (options.type) {
    case "shop": {
      const sortBy = options.sortBy || "significance";
      return [...items].sort((a: any, b: any) => {
        // Trending, then discount, then chosen sort field
        if (a.trending !== b.trending) return b.trending - a.trending;
        if ((b.discount ?? 0) !== (a.discount ?? 0)) return (b.discount ?? 0) - (a.discount ?? 0);
        if (sortBy === "price" && a.price && b.price) {
          return parseFloat(a.price.replace(/[^0-9.]/g, '')) - parseFloat(b.price.replace(/[^0-9.]/g, ''));
        }
        if (sortBy === "rating") return (b.rating ?? 0) - (a.rating ?? 0);
        if (sortBy === "significance") return (b.significance ?? 0) - (a.significance ?? 0);
        return 0;
      });
    }
    case "news": {
      // Rank news by significance, then trending, then recency
      return [...items].sort((a: any, b: any) => {
        if (a.trending !== b.trending) return b.trending - a.trending;
        if ((b.significance ?? 0) !== (a.significance ?? 0)) return (b.significance ?? 0) - (a.significance ?? 0);
        if (a.date && b.date) return new Date(b.date).getTime() - new Date(a.date).getTime();
        return 0;
      });
    }
    case "entertainment": {
      // Example: just sort by significance
      return [...items].sort((a: any, b: any) => (b.significance ?? 0) - (a.significance ?? 0));
    }
    default:
      return items;
  }
}
