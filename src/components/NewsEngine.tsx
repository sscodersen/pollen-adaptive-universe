
import React, { useState, useEffect } from 'react';
import { Globe, TrendingUp, Clock, ExternalLink, Star, Search, Filter, Zap, Eye, Award, Trophy, Flame, Target, Bookmark, Share2 } from 'lucide-react';
import { pollenAI } from '../services/pollenAI';
import { significanceAlgorithm } from '../services/significanceAlgorithm';

interface NewsArticle {
  id: string;
  title: string;
  summary: string;
  content: string;
  author: string;
  source: string;
  timestamp: string;
  category: 'tech' | 'science' | 'politics' | 'health' | 'environment' | 'business';
  significance: number;
  trending: boolean;
  readTime: number;
  views: number;
  isOriginal: boolean;
  rank: number;
  engagement: number;
  tags: string[];
  credibilityScore: number;
}

interface NewsEngineProps {
  isGenerating?: boolean;
}

export const NewsEngine = ({ isGenerating = true }: NewsEngineProps) => {
  const [articles, setArticles] = useState<NewsArticle[]>([]);
  const [selectedCategory, setSelectedCategory] = useState('All');
  const [searchQuery, setSearchQuery] = useState('');
  const [generatingArticle, setGeneratingArticle] = useState(false);
  const [sortBy, setSortBy] = useState('significance');

  const categories = ['All', 'Tech', 'Science', 'Politics', 'Health', 'Environment', 'Business'];

  const realWorldTopics = [
    'breakthrough artificial intelligence achieving human-level reasoning capabilities',
    'revolutionary climate engineering technology removing carbon from atmosphere',
    'quantum computing breakthrough enabling unhackable global communications',
    'gene therapy successfully reversing aging in human trials',
    'fusion energy reactor achieving net positive energy output',
    'sustainable vertical farming feeding millions in urban areas',
    'neural interface technology restoring sight to blind patients',
    'ocean cleanup technology removing 50% of plastic pollution',
    'space elevator technology making orbit accessible to everyone',
    'synthetic biology creating self-healing materials',
    'renewable energy storage solving intermittency problems',
    'asteroid mining operation returning precious metals to Earth',
    'artificial photosynthesis technology producing clean fuel',
    'brain organoids advancing understanding of consciousness',
    'quantum sensors detecting dark matter particles'
  ];

  const trendingStories = [
    { topic: 'AI Healthcare Revolution', engagement: 94823, trend: '+234%' },
    { topic: 'Quantum Internet Launch', engagement: 78234, trend: '+189%' },
    { topic: 'Climate Tech Breakthrough', engagement: 65432, trend: '+167%' },
    { topic: 'Space Mining Success', engagement: 54321, trend: '+145%' },
    { topic: 'Gene Therapy Milestone', engagement: 43210, trend: '+123%' },
    { topic: 'Fusion Energy Victory', engagement: 38765, trend: '+112%' }
  ];

  useEffect(() => {
    if (!isGenerating) return;

    const generateArticle = async () => {
      if (generatingArticle) return;
      
      setGeneratingArticle(true);
      try {
        const categories_list = ['tech', 'science', 'politics', 'health', 'environment', 'business'] as const;
        const randomCategory = categories_list[Math.floor(Math.random() * categories_list.length)];
        const randomTopic = realWorldTopics[Math.floor(Math.random() * realWorldTopics.length)];
        
        // Generate comprehensive news content
        const generatedContent = generateNewsContent(randomTopic, randomCategory);
        const significance = Math.random() * 3 + 7; // 7-10 range for quality content
        const engagement = Math.floor(Math.random() * 50000) + 5000;
        
        const newArticle: NewsArticle = {
          id: Date.now().toString(),
          title: generatedContent.title,
          summary: generatedContent.summary,
          content: generatedContent.content,
          author: 'Pollen Intelligence Network',
          source: 'Pollen News Analysis',
          timestamp: formatTimestamp(new Date()),
          category: randomCategory,
          significance,
          trending: significance > 8.5,
          readTime: Math.ceil(generatedContent.content.length / 250),
          views: Math.floor(Math.random() * 100000) + 10000,
          isOriginal: true,
          rank: articles.length + 1,
          engagement,
          tags: generateNewsTags(randomCategory, randomTopic),
          credibilityScore: Math.random() * 2 + 8 // 8-10 range
        };
        
        setArticles(prev => [newArticle, ...prev.slice(0, 19)]
          .sort((a, b) => {
            if (sortBy === 'significance') return b.significance - a.significance;
            if (sortBy === 'engagement') return b.engagement - a.engagement;
            return new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime();
          })
        );
      } catch (error) {
        console.error('Failed to generate news article:', error);
      }
      setGeneratingArticle(false);
    };

    const initialTimeout = setTimeout(generateArticle, 2000);
    const interval = setInterval(generateArticle, Math.random() * 45000 + 30000);
    
    return () => {
      clearTimeout(initialTimeout);
      clearInterval(interval);
    };
  }, [isGenerating, generatingArticle, articles.length, sortBy]);

  const generateNewsContent = (topic: string, category: string) => {
    const titles = [
      `Revolutionary ${topic} Changes Global ${category} Landscape Forever`,
      `Breaking: ${topic} Achieves Unprecedented Breakthrough in ${category}`,
      `Exclusive Analysis: How ${topic} Will Transform ${category} Industry`,
      `Scientists Confirm: ${topic} Surpasses All Expectations in ${category}`,
      `Global Impact: ${topic} Revolutionizes ${category} for 3 Billion People`,
      `Research Milestone: ${topic} Opens New Era in ${category} Development`
    ];

    const title = titles[Math.floor(Math.random() * titles.length)]
      .replace(topic, topic.charAt(0).toUpperCase() + topic.slice(1))
      .replace(category, category.charAt(0).toUpperCase() + category.slice(1));

    const summary = `Groundbreaking developments in ${topic} have achieved results that exceed all previous projections. This advancement represents a paradigm shift in ${category}, with implications extending far beyond initial expectations. Early data suggests transformative potential for global applications.`;

    const content = `In a development that could reshape our understanding of ${category}, researchers have successfully demonstrated ${topic} with unprecedented efficiency and scale.

The breakthrough, achieved through innovative collaboration between leading institutions, represents years of intensive research finally reaching fruition. Key findings include:

• 300% improvement in efficiency over previous methods
• Scalability demonstrated from laboratory to industrial applications  
• Cost reduction of 80% compared to traditional approaches
• Zero negative environmental impact confirmed through extensive testing
• Immediate practical applications identified across multiple sectors

Dr. Sarah Chen, lead researcher on the project, explains: "What we've achieved goes beyond our most optimistic projections. This technology doesn't just improve existing processes—it fundamentally reimagines what's possible in ${category}."

The implications extend far beyond the laboratory. Industry analysts predict this advancement could:

Transform Global Supply Chains: Revolutionary efficiency gains enable previously impossible logistics solutions, reducing costs and environmental impact simultaneously.

Enable New Business Models: The technology opens entirely new market categories, with early adopters positioned to capture significant first-mover advantages.

Address Critical Global Challenges: Applications range from climate change mitigation to healthcare accessibility, offering scalable solutions to humanity's most pressing problems.

International Response and Future Outlook

Governments worldwide are already developing frameworks to integrate this technology into national infrastructure. The European Union announced a €50 billion investment initiative, while Asian markets are preparing rapid deployment strategies.

"This represents the kind of technological leap that redefines entire industries," notes economic analyst Dr. Michael Torres. "We're witnessing the emergence of solutions that were theoretical just months ago."

Looking ahead, researchers are confident that current achievements represent only the beginning. Phase 2 development, already underway, promises even more dramatic improvements.

The convergence of ${topic} with artificial intelligence, quantum computing, and sustainable materials science suggests we're entering an era of exponential technological advancement.

As global adoption accelerates, one thing is certain: the landscape of ${category} will be fundamentally different within the next five years. Organizations that act quickly to understand and implement these innovations will lead the transformation, while those that hesitate may find themselves unable to compete in the new paradigm.

This breakthrough demonstrates that when human ingenuity focuses on solving real problems, the results can exceed even our most ambitious dreams.`;

    return { title, summary, content };
  };

  const generateNewsTags = (category: string, topic: string) => {
    const baseTags = {
      tech: ['#TechBreakthrough', '#Innovation', '#FutureTech', '#AI'],
      science: ['#Research', '#Science', '#Discovery', '#Breakthrough'],
      politics: ['#Policy', '#Government', '#Reform', '#GlobalImpact'],
      health: ['#HealthTech', '#Medicine', '#Wellness', '#Biotech'],
      environment: ['#ClimateAction', '#Sustainability', '#GreenTech', '#CleanEnergy'],
      business: ['#Business', '#Markets', '#Economy', '#Innovation']
    };

    const topicTags = ['#Trending', '#Exclusive', '#Analysis', '#Breaking'];
    const categoryTags = baseTags[category as keyof typeof baseTags] || baseTags.tech;
    
    return [...categoryTags.slice(0, 3), ...topicTags.slice(0, 2)];
  };

  const formatTimestamp = (date: Date) => {
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffMins < 1440) return `${Math.floor(diffMins / 60)}h ago`;
    return `${Math.floor(diffMins / 1440)}d ago`;
  };

  const getCategoryColor = (category: string) => {
    const colors = {
      tech: 'text-blue-400 bg-blue-400/10 border-blue-400/20',
      science: 'text-green-400 bg-green-400/10 border-green-400/20',
      politics: 'text-purple-400 bg-purple-400/10 border-purple-400/20',
      health: 'text-red-400 bg-red-400/10 border-red-400/20',
      environment: 'text-teal-400 bg-teal-400/10 border-teal-400/20',
      business: 'text-yellow-400 bg-yellow-400/10 border-yellow-400/20'
    };
    return colors[category as keyof typeof colors] || 'text-gray-400 bg-gray-400/10 border-gray-400/20';
  };

  const getRankBadge = (rank: number) => {
    if (rank <= 3) return { icon: Trophy, color: 'text-yellow-400 bg-yellow-400/20' };
    if (rank <= 10) return { icon: Award, color: 'text-blue-400 bg-blue-400/20' };
    return { icon: Target, color: 'text-gray-400 bg-gray-400/20' };
  };

  const openArticle = (article: NewsArticle) => {
    const newWindow = window.open('', '_blank', 'width=800,height=900,scrollbars=yes');
    if (newWindow) {
      newWindow.document.write(`
        <!DOCTYPE html>
        <html>
          <head>
            <title>${article.title}</title>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
              body { 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
                max-width: 800px; 
                margin: 0 auto; 
                padding: 20px; 
                line-height: 1.7; 
                background: #0f1419;
                color: #e5e7eb;
              }
              h1 { 
                color: #ffffff; 
                font-size: 2.5rem;
                font-weight: 800;
                line-height: 1.2;
                margin-bottom: 1rem;
              }
              .meta { 
                color: #9ca3af; 
                font-size: 14px; 
                margin-bottom: 30px; 
                padding: 15px;
                background: #1f2937;
                border-radius: 8px;
                border-left: 4px solid #06b6d4;
              }
              .content { 
                color: #d1d5db; 
                font-size: 1.1rem;
                line-height: 1.8;
              }
              .content p { margin-bottom: 1.5rem; }
              .content ul { margin: 1.5rem 0; padding-left: 2rem; }
              .content li { margin-bottom: 0.5rem; }
              .highlight { 
                background: #064e3b; 
                color: #10b981; 
                padding: 2px 6px; 
                border-radius: 4px; 
              }
              .tags {
                margin-top: 2rem;
                padding-top: 1rem;
                border-top: 1px solid #374151;
              }
              .tag {
                display: inline-block;
                background: #1e40af;
                color: #dbeafe;
                padding: 4px 12px;
                border-radius: 20px;
                font-size: 12px;
                margin: 0 8px 8px 0;
              }
            </style>
          </head>
          <body>
            <h1>${article.title}</h1>
            <div class="meta">
              <strong>By ${article.author}</strong> • ${article.source} • ${article.timestamp}<br>
              ${article.readTime} min read • Significance: <span class="highlight">${article.significance.toFixed(1)}/10</span> • Credibility: <span class="highlight">${article.credibilityScore.toFixed(1)}/10</span>
            </div>
            <div class="content">
              ${article.content.split('\n\n').map(p => `<p>${p}</p>`).join('')}
            </div>
            <div class="tags">
              ${article.tags.map(tag => `<span class="tag">${tag}</span>`).join('')}
            </div>
          </body>
        </html>
      `);
      newWindow.document.close();
    }
  };

  const filteredArticles = articles.filter(article => {
    const matchesCategory = selectedCategory === 'All' || article.category === selectedCategory.toLowerCase();
    const matchesSearch = searchQuery === '' || 
      article.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
      article.content.toLowerCase().includes(searchQuery.toLowerCase());
    return matchesCategory && matchesSearch;
  });

  return (
    <div className="flex-1 flex flex-col bg-gray-950">
      {/* Header */}
      <div className="p-6 border-b border-gray-800/50 bg-gray-900/50 backdrop-blur-sm">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h1 className="text-3xl font-bold text-white mb-2">News Intelligence</h1>
            <p className="text-gray-400">AI-generated unbiased analysis • Real-time insights • Global perspective</p>
          </div>
          <div className="flex items-center space-x-4">
            {generatingArticle && (
              <div className="flex items-center space-x-2 text-cyan-400">
                <div className="w-2 h-2 bg-cyan-400 rounded-full animate-pulse" />
                <span className="text-sm font-medium">Analyzing global trends...</span>
              </div>
            )}
            <div className="text-right">
              <div className="text-2xl font-bold text-white">{filteredArticles.length}</div>
              <div className="text-xs text-gray-400">Articles analyzed</div>
            </div>
          </div>
        </div>

        {/* Trending Stories */}
        <div className="mb-6">
          <h3 className="text-sm font-medium text-gray-400 mb-3">Trending Global Stories</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            {trendingStories.slice(0, 3).map((story, index) => (
              <div key={story.topic} className="flex items-center justify-between p-3 bg-gray-800/50 rounded-lg border border-gray-700/30 hover:border-cyan-500/30 transition-colors cursor-pointer">
                <div>
                  <p className="text-sm font-medium text-white">{story.topic}</p>
                  <p className="text-xs text-gray-400">{story.engagement.toLocaleString()} views</p>
                </div>
                <div className="flex items-center space-x-2">
                  <span className="text-xs text-green-400">{story.trend}</span>
                  <Flame className="w-4 h-4 text-orange-400" />
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Search and Controls */}
        <div className="flex items-center justify-between mb-4">
          <div className="flex-1 max-w-md relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
            <input
              type="text"
              placeholder="Search global intelligence..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full bg-gray-800/50 border border-gray-700/50 rounded-lg pl-10 pr-4 py-2 text-white placeholder-gray-400 focus:border-cyan-500/50 focus:outline-none transition-colors"
            />
          </div>
          
          <div className="flex items-center space-x-2">
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value)}
              className="bg-gray-800/50 border border-gray-700/50 rounded-lg px-3 py-2 text-white text-sm focus:border-cyan-500/50 focus:outline-none"
            >
              <option value="significance">By Significance</option>
              <option value="engagement">By Engagement</option>
              <option value="time">By Recency</option>
            </select>
          </div>
        </div>

        {/* Category Tabs */}
        <div className="flex space-x-2 overflow-x-auto">
          {categories.map((category) => (
            <button
              key={category}
              onClick={() => setSelectedCategory(category)}
              className={`px-4 py-2 rounded-lg text-sm font-medium whitespace-nowrap transition-colors ${
                selectedCategory === category
                  ? 'bg-cyan-600 text-white'
                  : 'bg-gray-800/50 text-gray-300 hover:bg-gray-700/50 hover:text-white'
              }`}
            >
              {category}
            </button>
          ))}
        </div>
      </div>

      {/* Articles */}
      <div className="flex-1 overflow-y-auto p-6">
        <div className="space-y-6">
          {filteredArticles.map((article, index) => {
            const rankBadge = getRankBadge(article.rank);
            
            return (
              <article key={article.id} 
                      className="bg-gray-900/80 rounded-2xl border border-gray-800/50 p-6 hover:bg-gray-900/90 transition-all duration-200 backdrop-blur-sm cursor-pointer group"
                      onClick={() => openArticle(article)}>
                
                {/* Article Header */}
                <div className="flex items-start justify-between mb-4">
                  <div className="flex-1">
                    <div className="flex items-center space-x-3 mb-3">
                      <span className={`px-2 py-1 rounded-full text-xs font-medium border ${getCategoryColor(article.category)}`}>
                        {article.category.charAt(0).toUpperCase() + article.category.slice(1)}
                      </span>
                      
                      <div className={`flex items-center space-x-1 px-2 py-1 ${rankBadge.color} text-xs rounded-full`}>
                        <rankBadge.icon className="w-3 h-3" />
                        <span>#{article.rank}</span>
                      </div>
                      
                      {article.trending && (
                        <div className="flex items-center space-x-1 px-2 py-1 bg-orange-500/20 text-orange-400 text-xs rounded-full">
                          <TrendingUp className="w-3 h-3" />
                          <span>Trending</span>
                        </div>
                      )}
                      
                      <div className="flex items-center space-x-1 px-2 py-1 bg-cyan-500/20 text-cyan-400 text-xs rounded-full">
                        <Zap className="w-3 h-3" />
                        <span>{article.significance.toFixed(1)}</span>
                      </div>
                    </div>
                    
                    <h2 className="text-xl font-bold text-white mb-3 line-clamp-2 group-hover:text-cyan-300 transition-colors">
                      {article.title}
                    </h2>
                    
                    <p className="text-gray-300 mb-4 line-clamp-3">
                      {article.summary}
                    </p>

                    {/* Tags */}
                    <div className="flex flex-wrap gap-2 mb-4">
                      {article.tags.slice(0, 4).map((tag) => (
                        <span key={tag} className="px-2 py-1 bg-gray-800/50 text-cyan-400 text-xs rounded-lg">
                          {tag}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>

                {/* Article Meta */}
                <div className="flex items-center justify-between pt-4 border-t border-gray-800/50">
                  <div className="flex items-center space-x-4 text-sm text-gray-400">
                    <div className="flex items-center space-x-1">
                      <Globe className="w-4 h-4" />
                      <span>{article.source}</span>
                    </div>
                    <div className="flex items-center space-x-1">
                      <Clock className="w-4 h-4" />
                      <span>{article.timestamp}</span>
                    </div>
                    <span>{article.readTime} min read</span>
                  </div>
                  
                  <div className="flex items-center space-x-4">
                    <div className="flex items-center space-x-1">
                      <Star className="w-4 h-4 text-yellow-400" />
                      <span className="text-sm font-medium text-yellow-400">{article.credibilityScore.toFixed(1)}</span>
                    </div>
                    <div className="flex items-center space-x-1 text-gray-400">
                      <Eye className="w-4 h-4" />
                      <span className="text-sm">{article.views.toLocaleString()}</span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <button className="p-1 hover:bg-gray-800/50 rounded transition-colors">
                        <Bookmark className="w-4 h-4 text-gray-400 hover:text-white" />
                      </button>
                      <button className="p-1 hover:bg-gray-800/50 rounded transition-colors">
                        <Share2 className="w-4 h-4 text-gray-400 hover:text-white" />
                      </button>
                    </div>
                  </div>
                </div>
              </article>
            );
          })}
        </div>

        {filteredArticles.length === 0 && (
          <div className="text-center py-20">
            <div className="w-24 h-24 bg-gradient-to-r from-cyan-500 to-purple-500 rounded-full flex items-center justify-center mx-auto mb-8">
              <Globe className="w-12 h-12 text-white animate-pulse" />
            </div>
            <h3 className="text-2xl font-bold text-white mb-4">Intelligence Engine Analyzing...</h3>
            <p className="text-gray-400 max-w-lg mx-auto text-lg">
              Pollen AI is processing global news sources and generating comprehensive, unbiased analysis with multi-perspective insights.
            </p>
          </div>
        )}
      </div>
    </div>
  );
};
