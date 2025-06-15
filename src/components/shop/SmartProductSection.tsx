
import React from "react";
import { useProducts } from "../../hooks/useProducts";
import { ProductCard } from "./ProductCard";

export const SmartProductSection: React.FC = () => {
  const { isLoading, sortedProducts } = useProducts();

  // "smart" recommendation = top 3 ranked
  const recommended = sortedProducts.slice(0, 3);

  if (isLoading || !recommended.length) return null;

  return (
    <div className="my-10 rounded-xl bg-gradient-to-br from-cyan-950/60 to-gray-900/90 border border-cyan-700/20 p-4 md:p-6 animate-fade-in">
      <h2 className="text-lg md:text-2xl font-bold text-cyan-200 mb-3 flex items-center">
        💡 Smart Products Just For You
      </h2>
      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
        {recommended.map((product) => (
          <ProductCard key={product.id} product={product} />
        ))}
      </div>
    </div>
  );
};
