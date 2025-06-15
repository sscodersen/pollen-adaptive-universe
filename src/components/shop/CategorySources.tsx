
import React from "react";
import { ExternalLink } from "lucide-react";

// Example static category→sources mapping.
// This could come from your backend in the future!
const CATEGORY_SOURCES: Record<
  string,
  { name: string; url: string }[]
> = {
  Electronics: [
    { name: "Amazon", url: "https://amazon.com" },
    { name: "Best Buy", url: "https://bestbuy.com" },
    { name: "Newegg", url: "https://newegg.com" },
  ],
  Fitness: [
    { name: "Nike", url: "https://nike.com" },
    { name: "Rogue Fitness", url: "https://roguefitness.com" },
    { name: "REP Fitness", url: "https://repfitness.com" },
  ],
  Home: [
    { name: "Wayfair", url: "https://wayfair.com" },
    { name: "IKEA", url: "https://ikea.com" },
  ],
  Fashion: [
    { name: "Nordstrom", url: "https://nordstrom.com" },
    { name: "ASOS", url: "https://asos.com" },
  ],
  // Add more category-source mappings as needed
};

interface CategorySourcesProps {
  categories: string[];
  activeCategory: string;
  setCategory: (category: string) => void;
}

export const CategorySources: React.FC<CategorySourcesProps> = ({
  categories,
  activeCategory,
  setCategory,
}) => {
  return (
    <aside className="bg-gray-900/70 border border-gray-800/40 rounded-xl p-4 mb-8">
      <h2 className="text-lg font-bold text-cyan-200 mb-3">Browse by Category</h2>
      <ul className="flex flex-wrap gap-2 mb-3">
        {categories.map((cat) => (
          <li key={cat}>
            <button
              onClick={() => setCategory(cat)}
              className={`px-3 py-1 rounded-full font-medium transition-all text-xs
                ${
                  cat === activeCategory
                    ? "bg-cyan-700/20 text-cyan-300 border border-cyan-600/40"
                    : "bg-gray-800/60 text-gray-400 border border-gray-700/30 hover:bg-gray-700/40"
                }`}
            >
              {cat}
            </button>
          </li>
        ))}
      </ul>
      {CATEGORY_SOURCES[activeCategory] && (
        <div className="mb-2">
          <div className="text-gray-300 text-xs mb-1">Top Sites for <span className="font-semibold">{activeCategory}</span>:</div>
          <ul className="flex flex-wrap gap-2">
            {CATEGORY_SOURCES[activeCategory].map((src) => (
              <li key={src.url}>
                <a
                  href={src.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center px-2 py-1 bg-cyan-800/20 text-cyan-200 rounded text-xs hover:bg-cyan-700/40 transition"
                >
                  {src.name}
                  <ExternalLink className="w-3 h-3 ml-1" />
                </a>
              </li>
            ))}
          </ul>
        </div>
      )}
    </aside>
  );
};
