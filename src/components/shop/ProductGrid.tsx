
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

  if (products.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center text-center text-gray-500 py-20 animate-fade-in">
        <Frown className="w-16 h-16 mb-4" />
        <h3 className="text-xl font-semibold text-white">No products found</h3>
        <p className="max-w-md">Try adjusting your search or filters to find what you're looking for.</p>
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
      {products.map((product) => (
        <ProductCard key={product.id} product={product} />
      ))}
    </div>
  );
};
