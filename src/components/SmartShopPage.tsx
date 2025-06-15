
import React, { useState, useEffect, useCallback } from 'react';
import { significanceAlgorithm } from '../services/significanceAlgorithm';
import { rankItems } from '../services/generalRanker';
import { Product } from '../types/shop';
import { ShopHeader } from './shop/ShopHeader';
import { FilterControls } from './shop/FilterControls';
import { ProductGrid } from './shop/ProductGrid';

const productTemplates = [
  {
    name: 'AI-Powered Wireless Earbuds Pro',
    description: 'Advanced noise cancellation with AI-driven sound optimization that adapts to your environment in real-time.',
    category: 'Audio',
    brand: 'TechFlow',
    features: ['Active Noise Cancellation', 'AI Sound Optimization', '30-hour battery life'],
    price: '$199',
    originalPrice: '$299',
    discount: 33,
    tags: ['AI', 'Audio', 'Wireless', 'Premium']
  },
  {
    name: 'Smart Home Security System 2024',
    description: 'Complete home security with AI facial recognition, 24/7 monitoring, and smart alerts.',
    category: 'Smart Home',
    brand: 'SecureLife',
    features: ['AI Facial Recognition', '4K Cameras', 'Mobile App Control'],
    price: '$449',
    originalPrice: '$599',
    discount: 25,
    tags: ['Security', 'Smart Home', 'AI', 'Surveillance']
  },
  {
    name: 'Ergonomic Gaming Chair Pro',
    description: 'Professional gaming chair with memory foam, RGB lighting, and ergonomic design for 12+ hour sessions.',
    category: 'Gaming',
    brand: 'GameThrone',
    features: ['Memory Foam Cushioning', 'RGB Lighting', 'Adjustable Height'],
    price: '$329',
    discount: 0,
    tags: ['Gaming', 'Ergonomic', 'RGB', 'Comfort']
  },
  {
    name: 'Sustainable Water Bottle with UV-C',
    description: 'Self-cleaning water bottle with UV-C sterilization technology and temperature control.',
    category: 'Health',
    brand: 'EcoClean',
    features: ['UV-C Sterilization', 'Temperature Control', 'BPA-Free'],
    price: '$89',
    originalPrice: '$119',
    discount: 25,
    tags: ['Health', 'Sustainable', 'Technology', 'Eco-Friendly']
  },
  {
    name: 'Professional Drone 4K Camera',
    description: 'High-performance drone with 4K recording, obstacle avoidance, and 45-minute flight time.',
    category: 'Photography',
    brand: 'SkyVision',
    features: ['4K Recording', 'Obstacle Avoidance', '45min Flight Time'],
    price: '$899',
    originalPrice: '$1199',
    discount: 25,
    tags: ['Drone', 'Photography', 'Professional', '4K']
  },
  {
    name: 'Smart Fitness Mirror',
    description: 'Interactive fitness mirror with AI personal trainer, real-time form correction, and workout tracking.',
    category: 'Fitness',
    brand: 'FitReflect',
    features: ['AI Personal Trainer', 'Form Correction', 'Workout Tracking'],
    price: '$1299',
    originalPrice: '$1599',
    discount: 19,
    tags: ['Fitness', 'AI', 'Smart Mirror', 'Health']
  },
  {
    name: 'Wireless Charging Desk Pad',
    description: 'Premium leather desk pad with built-in wireless charging zones for multiple devices.',
    category: 'Office',
    brand: 'WorkSpace',
    features: ['Wireless Charging', 'Premium Leather', 'Multiple Device Support'],
    price: '$159',
    discount: 0,
    tags: ['Office', 'Wireless Charging', 'Premium', 'Productivity']
  },
  {
    name: 'Smart Plant Care System',
    description: 'Automated plant care with soil sensors, watering system, and mobile app monitoring.',
    category: 'Home & Garden',
    brand: 'GreenThumb',
    features: ['Automated Watering', 'Soil Sensors', 'Mobile App'],
    price: '$129',
    originalPrice: '$179',
    discount: 28,
    tags: ['Smart Home', 'Plants', 'Automation', 'Gardening']
  }
];

export const SmartShopPage = () => {
  const [products, setProducts] = useState<Product[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [sortBy, setSortBy] = useState('significance');
  const [filter, setFilter] = useState('all');

  const generateProduct = useCallback(async () => {
    const template = productTemplates[Math.floor(Math.random() * productTemplates.length)];
    
    const scored = significanceAlgorithm.scoreContent(template.description, 'shop', template.brand);
    
    const product: Product = {
      id: Date.now().toString() + Math.random(),
      name: template.name,
      description: template.description,
      price: template.price,
      originalPrice: template.originalPrice,
      discount: template.discount,
      rating: Number((Math.random() * 1.5 + 3.5).toFixed(1)),
      reviews: Math.floor(Math.random() * 5000) + 100,
      category: template.category,
      brand: template.brand,
      tags: template.tags,
      link: `https://example.com/product/${template.name.toLowerCase().replace(/\s+/g, '-')}`,
      inStock: Math.random() > 0.1,
      trending: scored.significanceScore > 7.5,
      significance: scored.significanceScore,
      features: template.features,
      seller: template.brand,
      views: Math.floor(Math.random() * 25000) + 500,
      rank: Math.floor(Math.random() * 99) + 1,
      quality: Math.floor(scored.significanceScore * 10),
      impact: scored.significanceScore > 9 ? 'premium' : scored.significanceScore > 8 ? 'high' : scored.significanceScore > 6.5 ? 'medium' : 'low'
    };

    return product;
  }, []);

  const loadProducts = useCallback(async () => {
    setLoading(true);
    const newProducts = await Promise.all(
      Array.from({ length: 16 }, () => generateProduct())
    );
    const rankedProducts = rankItems(newProducts, { type: 'shop' });
    setProducts(rankedProducts.map((product, index) => ({ ...product, rank: index + 1 })));
    setLoading(false);
  }, [generateProduct]);

  useEffect(() => {
    loadProducts();
    const interval = setInterval(loadProducts, 45000);
    return () => clearInterval(interval);
  }, [loadProducts]);

  const categories = [...new Set(products.map(p => p.category))];

  const filteredProducts = products.filter(product => {
    const matchesSearch = !searchQuery || 
      product.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      product.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
      product.category.toLowerCase().includes(searchQuery.toLowerCase()) ||
      product.brand.toLowerCase().includes(searchQuery.toLowerCase()) ||
      product.tags.some(tag => tag.toLowerCase().includes(searchQuery.toLowerCase()));
    
    const matchesFilter = filter === 'all' || 
      filter === 'trending' && product.trending ||
      filter === 'discounted' && (product.discount || 0) > 0 ||
      product.category === filter;
    
    return matchesSearch && matchesFilter;
  }).sort((a, b) => {
    switch (sortBy) {
      case 'price':
        return parseFloat(a.price.replace('$', '')) - parseFloat(b.price.replace('$', ''));
      case 'rating':
        return b.rating - a.rating;
      case 'discount':
        return (b.discount || 0) - (a.discount || 0);
      default:
        return b.significance - a.significance;
    }
  });

  return (
    <div className="flex-1 bg-gray-950 p-6">
      <ShopHeader loading={loading} onRefresh={loadProducts} />
      
      <FilterControls
        searchQuery={searchQuery}
        setSearchQuery={setSearchQuery}
        sortBy={sortBy}
        setSortBy={setSortBy}
        filter={filter}
        setFilter={setFilter}
        categories={categories}
      />

      <ProductGrid isLoading={loading} products={filteredProducts} />
    </div>
  );
};
