import React, { useState, useEffect, useCallback } from 'react';
import { significanceAlgorithm } from '../services/significanceAlgorithm';
import { rankItems } from '../services/generalRanker';
import { contentOrchestrator } from '../services/contentOrchestrator';
import { enhancedTrendEngine } from '../services/enhancedTrendEngine';
import { insightFlow, ContentItem } from '../services/insightFlow';
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
        diversity: 0.85,
        freshness: 0.95,
        personalization: 0.7,
        qualityThreshold: 8.0, // Increased for InsightFlow compatibility
        trendingBoost: 1.8
      } as const;

      // Generate a mix of AI-driven and template-based apps
      const { content: aiApps } = await contentOrchestrator.generateContent({ 
        type: 'shop', 
        count: 20, // Generate more to have options after filtering
        strategy,
        query: 'innovative mobile apps, software applications, and digital tools' 
      });

      // Convert to InsightFlow ContentItem format for analysis
      const contentItems: ContentItem[] = aiApps.map((item: any, index: number) => ({
        id: `ai-${Date.now()}-${index}`,
        title: item.name || `Trending App ${index + 1}`,
        description: item.description || 'AI-powered application with advanced features',
        category: item.category || 'Productivity',
        tags: item.tags || ['app', 'software', 'productivity'],
        source: item.brand || item.seller || 'AI Developer',
        publishedAt: new Date().toISOString(), // Recent for timeliness
        rating: item.rating || Number((Math.random() * 1.5 + 3.5).toFixed(1)),
        price: typeof item.price === 'string' ? 
          parseFloat(item.price.replace('$', '')) || 0 : (item.price || 0),
        type: 'app' as const
      }));

      // Apply InsightFlow 7-factor significance analysis (only items >7 shown)
      const significantItems = await insightFlow.analyzeContentBatch(contentItems);
      console.log(`InsightFlow filtered ${contentItems.length} apps to ${significantItems.length} significant items (threshold: ${insightFlow.getThreshold()})`);

      // Convert AI generated content to app format
      const convertedAiApps: App[] = significantItems.map((scoredItem) => {
        const originalItem = aiApps.find((item: any) => item.name === scoredItem.title) || aiApps[0] || {};
        return {
        id: scoredItem.id,
        name: scoredItem.title,
        description: scoredItem.description,
        category: scoredItem.category,
        developer: scoredItem.source || 'AI Developer',
        rating: scoredItem.rating || Number((Math.random() * 1.5 + 3.5).toFixed(1)),
        reviews: Math.floor(Math.random() * 50000) + 1000,
        price: scoredItem.price === 0 ? 'Free' : `$${scoredItem.price?.toFixed(2)}`,
        originalPrice: (originalItem as any)?.originalPrice || undefined,
        discount: (originalItem as any)?.discount || 0,
        downloadLink: `https://example.com/download/${(scoredItem.title || 'app').toLowerCase().replace(/\s+/g, '-')}`,
        iconUrl: '/placeholder.svg',
        screenshots: ['/placeholder.svg', '/placeholder.svg', '/placeholder.svg'],
        size: `${Math.floor(Math.random() * 100 + 20)}.${Math.floor(Math.random() * 9)} MB`,
        version: `${Math.floor(Math.random() * 3 + 1)}.${Math.floor(Math.random() * 9)}.${Math.floor(Math.random() * 9)}`,
        updated: new Date(Date.now() - Math.random() * 30 * 24 * 60 * 60 * 1000).toLocaleDateString(),
        tags: scoredItem.tags || ['Popular', 'Trending', 'New'],
        inStock: (originalItem as any)?.inStock !== false,
        trending: (originalItem as any)?.trending || scoredItem.significance > 7.5,
        significance: scoredItem.significance || 8.0,
        downloads: `${Math.floor(Math.random() * 900 + 100)}K+`,
        rank: 0,
        featured: scoredItem.significance > 8.5
      };

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
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadApps();
    // Reduced frequency and added error handling to prevent infinite loops
    const interval = setInterval(() => {
      // Only refresh if not currently loading and no recent errors
      if (!loading) {
        loadApps().catch((error) => {
          console.warn('Scheduled app refresh failed:', error.message);
        });
      }
    }, 5 * 60 * 1000); // Increased to 5 minutes to reduce API load
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