
import React from 'react';
import { Product } from '../../types';
import { ProductCard } from './ProductCard';
import { Frown } from 'lucide-react';

interface ProductGridProps {
  isLoading: boolean;
  products: Product[];
}

const SkeletonCard = () => (
  <div className="bg-gray-900/50 rounded-xl p-6 border border-gray-800/50 animate-pulse">
    <div className="w-full h-48 bg-gray-700 rounded-lg mb-4"></div>
    <div className="w-3/4 h-4 bg-gray-700 rounded mb-2"></div>
    <div className="w-full h-3 bg-gray-700 rounded mb-4"></div>
    <div className="flex justify-between items-center mb-4">
      <div className="w-20 h-6 bg-gray-700 rounded"></div>
      <div className="w-16 h-4 bg-gray-700 rounded"></div>
    </div>
    <div className="w-full h-10 bg-gray-700 rounded"></div>
  </div>
);

// SMART SHOPPING ALGORITHM
function smartShoppingRecommendations(products: Product[], count: number = 3) {
  // Step 1: handpick trending, discounted, and high-rated first
  let picked: Product[] = [];
  let usedIds = new Set<string>();

  // Trending
  for (const product of products) {
    if (product.trending && !usedIds.has(product.id)) {
      picked.push(product);
      usedIds.add(product.id);
      if (picked.length >= count) break;
    }
  }
  // Highly discounted
  if (picked.length < count) {
    for (const product of products) {
      if ((product.discount ?? 0) >= 15 && !usedIds.has(product.id)) {
        picked.push(product);
        usedIds.add(product.id);
        if (picked.length >= count) break;
      }
    }
  }
  // High rating
  if (picked.length < count) {
    for (const product of products) {
      if (product.rating >= 4.7 && !usedIds.has(product.id)) {
        picked.push(product);
        usedIds.add(product.id);
        if (picked.length >= count) break;
      }
    }
  }
  // Fill remaining with top significance, but limit to one per category for variety
  if (picked.length < count) {
    const sorted = [...products].sort((a, b) => b.significance - a.significance);
    const usedCategories = new Set<string>(picked.map(p => p.category));
    for (const product of sorted) {
      if (!usedIds.has(product.id) && !usedCategories.has(product.category)) {
        picked.push(product);
        usedIds.add(product.id);
        usedCategories.add(product.category);
        if (picked.length >= count) break;
      }
    }
  }
  // As a fallback just add top significance
  if (picked.length < count) {
    for (const product of products) {
      if (!usedIds.has(product.id)) {
        picked.push(product);
        usedIds.add(product.id);
        if (picked.length >= count) break;
      }
    }
  }
  // Guarantee always unique and always some output (if enough products)
  return picked;
}

function smartShoppingRest(products: Product[], recommended: Product[]) {
  const recIds = new Set(recommended.map(p => p.id));
  // Sort rest by significance, discount, then rating
  return products
    .filter(p => !recIds.has(p.id))
    .sort((a, b) =>
      b.significance !== a.significance
        ? b.significance - a.significance
        : (b.discount ?? 0) !== (a.discount ?? 0)
        ? (b.discount ?? 0) - (a.discount ?? 0)
        : b.rating - a.rating
    );
}

export const ProductGrid: React.FC<ProductGridProps> = ({ isLoading, products }) => {
  if (isLoading) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {[...Array(9)].map((_, i) => (
          <SkeletonCard key={i} />
        ))}
      </div>
    );
  }

  if (!products || products.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center text-center text-gray-500 py-20 animate-fade-in">
        <Frown className="w-16 h-16 mb-4" />
        <h3 className="text-xl font-semibold text-white">No products found</h3>
        <p className="max-w-md">Try adjusting your search or filters to find what you're looking for.</p>
      </div>
    );
  }

  // Smart shopping recommendation logic
  const recommended = smartShoppingRecommendations(products, 3);
  const rest = smartShoppingRest(products, recommended);

  return (
    <div className="flex flex-col gap-10">
      {/* Recommended Section */}
      <div>
        <h2 className="text-2xl font-bold text-cyan-200 mb-4 animate-fade-in">Recommended for you</h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-6">
          {recommended.map(product => (
            <ProductCard key={product.id} product={product} />
          ))}
        </div>
      </div>
      {/* Divider */}
      {rest.length > 0 && (
        <div className="border-t border-gray-700 pt-8">
          <h3 className="text-xl font-semibold text-gray-200 mb-3 animate-fade-in">
            Explore More Products
          </h3>
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-6">
            {rest.map(product => (
              <ProductCard key={product.id} product={product} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

