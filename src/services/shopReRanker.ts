import { Product } from '@/types/shop';
import { TrendData } from './enhancedTrendEngine';

// Re-rank products using multi-factor score: significance, discount/value, rating/reviews,
// trend relevance, brand/category diversity.
export const reRankProducts = (products: Product[], trends: TrendData[]): Product[] => {
  if (!products?.length) return products;

  // Build simple trend keyword set
  const topTrends = trends.slice(0, 30);
  const trendKeywords = new Set<string>();
  topTrends.forEach(t => t.keywords.forEach(k => trendKeywords.add(k.toLowerCase())));

  // Count per brand/category for diversity penalty
  const brandCount = new Map<string, number>();
  const categoryCount = new Map<string, number>();

  const scored = products.map((p) => {
    const price = Number((p.price || '').replace(/[^0-9.]/g, '')) || 0;
    const orig = Number((p.originalPrice || '').replace(/[^0-9.]/g, '')) || price;
    const discountPct = p.discount ?? (orig > price && orig > 0 ? Math.round((1 - price / orig) * 100) : 0);

    const ratingScore = Math.min(1, Math.max(0, (p.rating - 3) / 2)); // 0..1 for 3-5 stars
    const reviewBoost = Math.min(1, Math.log10((p.reviews || 1) + 1) / 3);

    // Trend relevance: check name/category/tags against trend keywords
    const text = `${p.name} ${p.description} ${p.category} ${p.brand} ${p.tags.join(' ')}`.toLowerCase();
    const matches = Array.from(trendKeywords).filter(k => text.includes(k)).length;
    const trendRelevance = Math.min(1, matches / 4);

    // Value score favors bigger discounts and reasonable price
    const valueScore = Math.min(1, (discountPct / 60)) * 0.7 + (price > 0 ? Math.max(0, 1 - price / 1000) * 0.3 : 0);

    const base = 0.35 * (p.significance / 10) + 0.25 * valueScore + 0.25 * (0.7 * ratingScore + 0.3 * reviewBoost) + 0.15 * trendRelevance;

    // Diversity penalty will be applied after initial sort
    return { product: p, base, discountPct };
  });

  // Initial sort by base score and keep brand/category tallies
  scored.sort((a, b) => b.base - a.base);

  const adjusted: { product: Product; score: number }[] = [];
  for (const s of scored) {
    const b = s.product.brand || 'Unknown';
    const c = s.product.category || 'Misc';
    const bCount = brandCount.get(b) || 0;
    const cCount = categoryCount.get(c) || 0;

    // Apply mild penalty if a brand/category repeats too much, to ensure variety
    const diversityPenalty = 1 - Math.min(0.3, bCount * 0.05 + cCount * 0.04);
    const finalScore = s.base * diversityPenalty;

    brandCount.set(b, bCount + 1);
    categoryCount.set(c, cCount + 1);

    adjusted.push({ product: s.product, score: finalScore });
  }

  const ranked = adjusted
    .sort((a, b) => b.score - a.score)
    .map((s, idx) => ({ ...s.product, rank: idx + 1 }));

  return ranked;
};
