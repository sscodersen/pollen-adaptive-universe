import React, { useState, useEffect, useCallback } from 'react';
import { significanceAlgorithm } from '../services/significanceAlgorithm';
import { rankItems } from '../services/generalRanker';
import { contentOrchestrator } from '../services/contentOrchestrator';
import { enhancedTrendEngine } from '../services/enhancedTrendEngine';
import { AppStoreHeader } from './appstore/AppStoreHeader';
import { AppStoreFilters } from './appstore/AppStoreFilters';
import { AppGrid } from './appstore/AppGrid';

export interface App {
  id: string;
  name: string;
  description: string;
  category: string;
  developer: string;
  rating: number;
  reviews: number;
  price: string;
  originalPrice?: string;
  discount?: number;
  downloadLink: string;
  iconUrl: string;
  screenshots: string[];
  size: string;
  version: string;
  updated: string;
  tags: string[];
  inStock: boolean;
  trending: boolean;
  significance: number;
  downloads: string;
  rank: number;
  featured: boolean;
}

const appTemplates = [
  {
    name: 'AI Photo Editor Pro',
    description: 'Advanced photo editing with AI-powered filters, background removal, and smart enhancements.',
    category: 'Photography',
    developer: 'CreativeStudio',
    size: '45.2 MB',
    version: '2.1.0',
    tags: ['AI', 'Photo', 'Editor', 'Creative'],
    price: '$9.99',
    originalPrice: '$14.99',
    discount: 33
  },
  {
    name: 'Smart Fitness Tracker',
    description: 'Track workouts, monitor health metrics, and get AI-powered fitness recommendations.',
    category: 'Health & Fitness',
    developer: 'FitTech Solutions',
    size: '32.1 MB',
    version: '1.8.5',
    tags: ['Fitness', 'Health', 'AI', 'Tracking'],
    price: 'Free',
    discount: 0
  },
  {
    name: 'Code Editor X',
    description: 'Professional code editor with syntax highlighting, Git integration, and AI code completion.',
    category: 'Developer Tools',
    developer: 'DevTools Inc',
    size: '78.9 MB',
    version: '3.2.1',
    tags: ['Development', 'Code', 'Programming', 'Tools'],
    price: '$19.99',
    discount: 0
  },
  {
    name: 'Music Producer Studio',
    description: 'Create, mix, and master music with professional-grade tools and AI-assisted composition.',
    category: 'Music',
    developer: 'AudioWorks',
    size: '156.7 MB',
    version: '4.0.2',
    tags: ['Music', 'Audio', 'Production', 'Creative'],
    price: '$29.99',
    originalPrice: '$39.99',
    discount: 25
  },
  {
    name: 'Language Learning AI',
    description: 'Learn languages faster with AI tutoring, speech recognition, and personalized lessons.',
    category: 'Education',
    developer: 'EduTech Global',
    size: '67.3 MB',
    version: '2.5.0',
    tags: ['Education', 'Language', 'AI', 'Learning'],
    price: '$12.99',
    discount: 0
  },
  {
    name: 'Smart Home Controller',
    description: 'Control all your smart home devices from one app with AI automation and scheduling.',
    category: 'Utilities',
    developer: 'HomeAutomation Co',
    size: '28.4 MB',
    version: '1.9.3',
    tags: ['Smart Home', 'IoT', 'Automation', 'Control'],
    price: 'Free',
    discount: 0
  }
];

export const AppStorePage = () => {
  const [apps, setApps] = useState<App[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [sortBy, setSortBy] = useState('significance');
  const [filter, setFilter] = useState('all');

  const loadApps = useCallback(async () => {
    setLoading(true);
    try {
      // Generate trending apps using content orchestrator for better quality
      const strategy = {
        diversity: 0.8,
        freshness: 0.9,
        personalization: 0.6,
        qualityThreshold: 7.5,
        trendingBoost: 1.5
      } as const;

      // Generate a mix of AI-driven and template-based apps
      const { content: aiApps } = await contentOrchestrator.generateContent({ 
        type: 'shop', 
        count: 8, 
        strategy,
        query: 'trending mobile apps and software applications' 
      });

      // Convert AI generated content to app format
      const convertedAiApps: App[] = aiApps.map((item: any, index: number) => ({
        id: `ai-${Date.now()}-${index}`,
        name: item.name || `Trending App ${index + 1}`,
        description: item.description || 'AI-powered application with advanced features',
        category: item.category || 'Productivity',
        developer: item.brand || item.seller || 'AI Developer',
        rating: item.rating || Number((Math.random() * 1.5 + 3.5).toFixed(1)),
        reviews: item.reviews || Math.floor(Math.random() * 50000) + 1000,
        price: item.price || (Math.random() > 0.3 ? `$${(Math.random() * 50 + 5).toFixed(2)}` : 'Free'),
        originalPrice: item.originalPrice,
        discount: item.discount || 0,
        downloadLink: `https://example.com/download/${(item.name || 'app').toLowerCase().replace(/\s+/g, '-')}`,
        iconUrl: '/placeholder.svg',
        screenshots: ['/placeholder.svg', '/placeholder.svg', '/placeholder.svg'],
        size: `${Math.floor(Math.random() * 100 + 20)}.${Math.floor(Math.random() * 9)} MB`,
        version: `${Math.floor(Math.random() * 3 + 1)}.${Math.floor(Math.random() * 9)}.${Math.floor(Math.random() * 9)}`,
        updated: new Date(Date.now() - Math.random() * 30 * 24 * 60 * 60 * 1000).toLocaleDateString(),
        tags: item.tags || ['Popular', 'Trending', 'New'],
        inStock: item.inStock !== false,
        trending: item.trending || item.significance > 7.5,
        significance: item.significance || 8.0,
        downloads: `${Math.floor(Math.random() * 900 + 100)}K+`,
        rank: 0,
        featured: item.significance > 8.5
      }));

      // Also generate some template-based apps for variety
      const templateApps: App[] = Array.from({ length: 4 }, (_, index) => {
        const template = appTemplates[Math.floor(Math.random() * appTemplates.length)];
        const scored = significanceAlgorithm.scoreContent(template.description, 'entertainment', template.developer);
        
        return {
          id: `template-${Date.now()}-${index}`,
          name: template.name,
          description: template.description,
          category: template.category,
          developer: template.developer,
          rating: Number((Math.random() * 1.5 + 3.5).toFixed(1)),
          reviews: Math.floor(Math.random() * 50000) + 1000,
          price: template.price,
          originalPrice: template.originalPrice,
          discount: template.discount,
          downloadLink: `https://example.com/download/${template.name.toLowerCase().replace(/\s+/g, '-')}`,
          iconUrl: '/placeholder.svg',
          screenshots: ['/placeholder.svg', '/placeholder.svg', '/placeholder.svg'],
          size: template.size,
          version: template.version,
          updated: new Date(Date.now() - Math.random() * 30 * 24 * 60 * 60 * 1000).toLocaleDateString(),
          tags: template.tags,
          inStock: true,
          trending: scored.significanceScore > 7.5,
          significance: scored.significanceScore,
          downloads: `${Math.floor(Math.random() * 900 + 100)}K+`,
          rank: 0,
          featured: scored.significanceScore > 8.5
        };
      });

      // Combine and rank all apps
      const allApps = [...convertedAiApps, ...templateApps];
      const rankedApps = rankItems(allApps, { type: 'entertainment' });
      setApps(rankedApps.map((app, index) => ({ ...app, rank: index + 1 })));
    } catch (error) {
      console.error('Failed to load apps:', error);
      // Fallback to template-based generation
      const fallbackApps = Array.from({ length: 12 }, (_, index) => {
        const template = appTemplates[Math.floor(Math.random() * appTemplates.length)];
        const scored = significanceAlgorithm.scoreContent(template.description, 'entertainment', template.developer);
        
        return {
          id: `fallback-${Date.now()}-${index}`,
          name: template.name,
          description: template.description,
          category: template.category,
          developer: template.developer,
          rating: Number((Math.random() * 1.5 + 3.5).toFixed(1)),
          reviews: Math.floor(Math.random() * 50000) + 1000,
          price: template.price,
          originalPrice: template.originalPrice,
          discount: template.discount,
          downloadLink: `https://example.com/download/${template.name.toLowerCase().replace(/\s+/g, '-')}`,
          iconUrl: '/placeholder.svg',
          screenshots: ['/placeholder.svg', '/placeholder.svg', '/placeholder.svg'],
          size: template.size,
          version: template.version,
          updated: new Date(Date.now() - Math.random() * 30 * 24 * 60 * 60 * 1000).toLocaleDateString(),
          tags: template.tags,
          inStock: true,
          trending: scored.significanceScore > 7.5,
          significance: scored.significanceScore,
          downloads: `${Math.floor(Math.random() * 900 + 100)}K+`,
          rank: index + 1,
          featured: scored.significanceScore > 8.5
        };
      });
      setApps(fallbackApps);
    }
    setLoading(false);
  }, []);

  useEffect(() => {
    loadApps();
    const interval = setInterval(loadApps, 60000);
    return () => clearInterval(interval);
  }, [loadApps]);

  const categories = [...new Set(apps.map(app => app.category))];

  const filteredApps = apps.filter(app => {
    const matchesSearch = !searchQuery || 
      app.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      app.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
      app.category.toLowerCase().includes(searchQuery.toLowerCase()) ||
      app.developer.toLowerCase().includes(searchQuery.toLowerCase()) ||
      app.tags?.some(tag => tag.toLowerCase().includes(searchQuery.toLowerCase()));
    
    const matchesFilter = filter === 'all' || 
      filter === 'trending' && app.trending ||
      filter === 'free' && app.price === 'Free' ||
      filter === 'featured' && app.featured ||
      app.category === filter;
    
    return matchesSearch && matchesFilter;
  }).sort((a, b) => {
    switch (sortBy) {
      case 'price':
        const priceA = a.price === 'Free' ? 0 : parseFloat(a.price.replace('$', ''));
        const priceB = b.price === 'Free' ? 0 : parseFloat(b.price.replace('$', ''));
        return priceA - priceB;
      case 'rating':
        return b.rating - a.rating;
      case 'downloads':
        return parseInt(b.downloads.replace(/[^\d]/g, '')) - parseInt(a.downloads.replace(/[^\d]/g, ''));
      default:
        return b.significance - a.significance;
    }
  });

  return (
    <div className="flex-1 bg-gray-950 p-6">
      <AppStoreHeader loading={loading} onRefresh={loadApps} />
      
      <AppStoreFilters
        searchQuery={searchQuery}
        setSearchQuery={setSearchQuery}
        sortBy={sortBy}
        setSortBy={setSortBy}
        filter={filter}
        setFilter={setFilter}
        categories={categories}
      />

      <AppGrid isLoading={loading} apps={filteredApps} />
    </div>
  );
};