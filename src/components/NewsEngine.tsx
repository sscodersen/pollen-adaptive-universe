import React, { useState, useEffect, useCallback } from 'react';
import { Search, Clock, Award, Filter, Globe, Eye, BookOpen, TrendingUp } from 'lucide-react';
import { pollenAI } from '../services/pollenAI';
import { significanceAlgorithm } from '../services/significanceAlgorithm';
import { rankItems } from '../services/generalRanker';
import { SignificanceBadge } from "./entertainment/SignificanceBadge";
import { TrendingBadge } from "./entertainment/TrendingBadge";

interface NewsEngineProps {
  isGenerating?: boolean;
}

interface NewsItem {
  id: string;
  title: string;
  summary: string;
  content: string;
  category: string;
  source: string;
  timestamp: string;
  significance: number;
  tags: string[];
  trending: boolean;
  views: number;
  readTime: number;
}

export const NewsEngine = ({ isGenerating = false }: NewsEngineProps) => {
  const [articles, setArticles] = useState<NewsItem[]>([]);
  const [selectedArticle, setSelectedArticle] = useState<NewsItem | null>(null);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [filter, setFilter] = useState('all');
  const [sortBy, setSortBy] = useState('significance');

  const newsTopics = [
    { category: 'AI & Technology', keywords: ['artificial intelligence', 'machine learning', 'quantum computing', 'robotics'] },
    { category: 'Climate & Environment', keywords: ['climate change', 'renewable energy', 'carbon capture', 'sustainability'] },
    { category: 'Biotechnology', keywords: ['gene therapy', 'crispr', 'personalized medicine', 'bioengineering'] },
    { category: 'Space & Science', keywords: ['space exploration', 'mars mission', 'telescope discoveries', 'physics breakthroughs'] },
    { category: 'Economics & Finance', keywords: ['economic trends', 'cryptocurrency', 'market analysis', 'global trade'] },
    { category: 'Health & Medicine', keywords: ['medical breakthroughs', 'pandemic research', 'mental health', 'drug development'] }
  ];

  const generateNewsContent = useCallback(async (topic: typeof newsTopics[0]) => {
    const keyword = topic.keywords[Math.floor(Math.random() * topic.keywords.length)];
    
    const titles = [
      `Breakthrough ${keyword} research reveals unprecedented possibilities for global transformation`,
      `Major ${keyword} initiative launches with potential to impact millions worldwide`,
      `Scientists achieve critical ${keyword} milestone with far-reaching implications`,
      `Revolutionary ${keyword} approach offers solutions to long-standing challenges`,
      `International ${keyword} collaboration yields promising results for future applications`,
      `New ${keyword} technology demonstrates significant advantages over current methods`
    ];

    const title = titles[Math.floor(Math.random() * titles.length)];
    
    const summaries = [
      `Researchers have made a significant advancement in ${keyword}, demonstrating practical applications that could revolutionize the field within the next decade. The breakthrough addresses key limitations that have hindered progress for years.`,
      `A comprehensive study on ${keyword} reveals actionable insights that governments and organizations can implement immediately. The findings suggest substantial benefits for both economic and social outcomes.`,
      `International experts collaborate on ${keyword} research, producing results that exceed expectations. The methodology could be scaled globally, offering sustainable solutions to current challenges.`,
      `Innovative approaches to ${keyword} show remarkable success in controlled trials. The research team reports consistent positive outcomes across diverse testing environments and populations.`,
      `Analysis of ${keyword} trends indicates significant opportunities for advancement. The data suggests that strategic investments in this area could yield substantial returns within five years.`
    ];

    const summary = summaries[Math.floor(Math.random() * summaries.length)];

    const fullContent = `${summary}

The research team, led by international experts, has developed novel methodologies that address core challenges in ${keyword}. Their approach combines theoretical innovation with practical application, resulting in solutions that are both scientifically sound and economically viable.

Key findings include:
• Significant improvement in efficiency compared to existing methods
• Scalable implementation suitable for global deployment
• Positive environmental and social impact assessments
• Strong potential for commercial applications

The study involved collaboration between leading institutions and has undergone rigorous peer review. Results demonstrate consistent positive outcomes across multiple testing phases, with success rates exceeding 85% in controlled environments.

Industry experts consider this development a major step forward, with potential applications spanning multiple sectors. The research methodology could serve as a foundation for future innovations in the field.

Implementation strategies are already being developed for real-world deployment, with pilot programs scheduled to begin within the next six months. The team expects to publish detailed technical specifications and guidelines for adoption by other research groups and organizations.

This breakthrough represents years of dedicated research and international cooperation, highlighting the importance of collaborative scientific efforts in addressing global challenges.`;

    const scored = significanceAlgorithm.scoreContent(fullContent, 'news', 'Pollen Analysis');

    return {
      id: Date.now().toString() + Math.random(),
      title,
      summary,
      content: fullContent,
      category: topic.category,
      source: 'Pollen Intelligence',
      timestamp: `${Math.floor(Math.random() * 24)}h`,
      significance: scored.significanceScore,
      tags: [keyword, topic.category, scored.significanceScore > 8 ? 'Breaking' : 'Trending'],
      trending: scored.significanceScore > 7.5,
      views: Math.floor(Math.random() * 50000) + 1000,
      readTime: Math.floor(Math.random() * 8) + 3
    };
  }, []);

  const loadNews = useCallback(async () => {
    setLoading(true);
    const newsPromises = newsTopics.map(topic => generateNewsContent(topic));
    const newArticles = await Promise.all(newsPromises);
    
    // Add a few more varied articles
    const additionalArticles = await Promise.all([
      generateNewsContent(newsTopics[Math.floor(Math.random() * newsTopics.length)]),
      generateNewsContent(newsTopics[Math.floor(Math.random() * newsTopics.length)])
    ]);
    
    const allArticles = [...newArticles, ...additionalArticles];
    setArticles(allArticles.sort((a, b) => b.significance - a.significance));
    setLoading(false);
  }, [generateNewsContent]);

  useEffect(() => {
    loadNews();
    const interval = setInterval(loadNews, 45000); // Refresh every 45 seconds
    return () => clearInterval(interval);
  }, [loadNews]);

  // Use generalRanker for news sorting
  const filteredArticles = articles.filter(article => {
    if (searchQuery) {
      return article.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
             article.summary.toLowerCase().includes(searchQuery.toLowerCase()) ||
             article.category.toLowerCase().includes(searchQuery.toLowerCase());
    }
    if (filter === 'trending') return article.trending;
    if (filter === 'breaking') return article.significance > 8;
    return true;
  });

  // USE generalRanker for sorting here!
  const sortedArticles = rankItems(filteredArticles, { type: "news", sortBy });

  if (selectedArticle) {
    return (
      <div className="flex-1 bg-gradient-to-br from-slate-950 via-gray-950 to-blue-950 min-h-screen">
        <div className="sticky top-0 z-10 bg-gray-900/95 backdrop-blur-xl border-b border-gray-800/50">
          <div className="p-6">
            <div className="flex items-center justify-between mb-4">
              <button
                onClick={() => setSelectedArticle(null)}
                className="flex items-center space-x-2 text-cyan-400 hover:text-cyan-300 transition-colors font-medium"
              >
                <span>← Back to News Intelligence</span>
              </button>
              <div className="flex items-center space-x-3">
                <SignificanceBadge score={selectedArticle.significance} />
                <TrendingBadge isTrending={selectedArticle.trending} />
                <div className="flex items-center space-x-2 text-gray-400 text-sm">
                  <Eye className="w-4 h-4" />
                  <span>{selectedArticle.views.toLocaleString()}</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="p-6 max-w-5xl mx-auto">
          <div className="mb-8">
            <div className="flex items-center space-x-4 mb-6">
              <span className="px-4 py-2 bg-purple-500/20 text-purple-300 rounded-full text-sm border border-purple-500/30 font-medium">
                {selectedArticle.category}
              </span>
              <span className="text-gray-400 text-sm flex items-center space-x-1">
                <Clock className="w-4 h-4" />
                <span>{selectedArticle.timestamp} ago</span>
              </span>
              <span className="text-gray-400 text-sm flex items-center space-x-1">
                <BookOpen className="w-4 h-4" />
                <span>{selectedArticle.readTime} min read</span>
              </span>
            </div>
            <h1 className="text-4xl md:text-5xl font-bold text-white mb-6 leading-tight">{selectedArticle.title}</h1>
            <p className="text-xl text-gray-300 leading-relaxed font-light">{selectedArticle.summary}</p>
          </div>

          <div className="prose prose-invert prose-lg max-w-none">
            <div className="text-gray-200 leading-relaxed whitespace-pre-line">
              {selectedArticle.content}
            </div>
          </div>

          <div className="flex flex-wrap gap-2 mt-8 pt-6 border-t border-gray-800/50">
            {selectedArticle.tags.map((tag, index) => (
              <span key={index} className="px-3 py-1 bg-gray-700/50 text-gray-300 rounded-full text-sm border border-gray-600/50 hover:bg-gray-600/50 transition-colors">
                #{tag}
              </span>
            ))}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 bg-gradient-to-br from-slate-950 via-gray-950 to-blue-950 min-h-screen">
      <div className="sticky top-0 z-10 bg-gray-900/95 backdrop-blur-xl border-b border-gray-800/50">
        <div className="p-6">
          <div className="flex items-center justify-between mb-6">
            <div>
              <h1 className="text-3xl md:text-4xl font-bold text-white mb-2 bg-gradient-to-r from-cyan-400 to-purple-400 bg-clip-text text-transparent">
                News Intelligence
              </h1>
              <p className="text-gray-400">AI-curated insights • Real-time analysis • Unbiased perspectives</p>
            </div>
            <div className="flex items-center space-x-3">
              <div className="px-4 py-2 bg-green-500/10 text-green-400 rounded-full text-sm font-medium border border-green-500/20 flex items-center space-x-2 animate-pulse-glow">
                <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                <span>Live Analysis</span>
              </div>
            </div>
          </div>

          <div className="flex flex-col lg:flex-row items-stretch lg:items-center justify-between space-y-4 lg:space-y-0 lg:space-x-4">
            <div className="flex-1 relative">
              <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search news, analysis, and insights..."
                className="w-full bg-gray-800/50 border border-gray-700/50 rounded-xl pl-12 pr-4 py-3 text-white placeholder-gray-400 focus:border-cyan-500/50 focus:outline-none focus:ring-2 focus:ring-cyan-500/20 transition-all"
              />
            </div>

            <div className="flex flex-wrap gap-2">
              {[
                { id: 'all', name: 'All News', icon: Globe },
                { id: 'trending', name: 'Trending', icon: TrendingUp },
                { id: 'breaking', name: 'Breaking', icon: Award }
              ].map((filterOption) => (
                <button
                  key={filterOption.id}
                  onClick={() => setFilter(filterOption.id)}
                  className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-all font-medium ${
                    filter === filterOption.id
                      ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/30 shadow-lg'
                      : 'bg-gray-800/50 text-gray-400 hover:bg-gray-700/50 border border-gray-700/30 hover:text-gray-300'
                  }`}
                >
                  <filterOption.icon className="w-4 h-4" />
                  <span className="text-sm">{filterOption.name}</span>
                </button>
              ))}
            </div>

            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value)}
              className="bg-gray-800/50 border border-gray-700/50 rounded-lg px-4 py-2 text-white text-sm focus:outline-none focus:border-cyan-500/50 focus:ring-2 focus:ring-cyan-500/20 transition-all min-w-[160px]"
            >
              <option value="significance">Sort by Significance</option>
              <option value="views">Sort by Views</option>
              <option value="recent">Sort by Recent</option>
            </select>
          </div>
        </div>
      </div>

      <div className="p-6">
        {loading ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {[...Array(6)].map((_, i) => (
              <div key={i} className="bg-gray-900/50 rounded-xl p-6 border border-gray-800/50 animate-pulse">
                <div className="w-full h-4 bg-gray-700/50 rounded mb-4"></div>
                <div className="w-3/4 h-4 bg-gray-700/50 rounded mb-4"></div>
                <div className="space-y-2 mb-4">
                  <div className="w-full h-3 bg-gray-700/50 rounded"></div>
                  <div className="w-full h-3 bg-gray-700/50 rounded"></div>
                  <div className="w-2/3 h-3 bg-gray-700/50 rounded"></div>
                </div>
                <div className="flex space-x-2">
                  <div className="w-16 h-6 bg-gray-700/50 rounded"></div>
                  <div className="w-16 h-6 bg-gray-700/50 rounded"></div>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {sortedArticles.map((article) => (
              <div
                key={article.id}
                onClick={() => setSelectedArticle(article)}
                className="bg-gradient-to-br from-gray-900/60 via-gray-900/50 to-blue-950/50 rounded-xl border border-gray-800/50 p-6 hover:bg-gray-900/80 hover:border-gray-700/50 transition-all cursor-pointer group shadow-lg relative animate-fade-in"
              >
                <div className="absolute top-4 right-4 flex gap-2 z-10">
                  <TrendingBadge isTrending={article.trending} />
                  <SignificanceBadge score={article.significance} />
                </div>

                <div className="flex items-center justify-between mb-4">
                  <span className="px-3 py-1 bg-purple-500/20 text-purple-300 rounded-full text-xs border border-purple-500/30 font-medium">
                    {article.category}
                  </span>
                </div>

                <h3 className="text-lg font-semibold text-white mb-3 group-hover:text-cyan-300 transition-colors line-clamp-3 leading-tight">
                  {article.title}
                </h3>

                <p className="text-gray-400 text-sm mb-4 line-clamp-3 leading-relaxed">
                  {article.summary}
                </p>

                <div className="flex items-center justify-between text-xs text-gray-500 mb-4">
                  <div className="flex items-center space-x-3">
                    <span className="flex items-center space-x-1">
                      <Clock className="w-3 h-3" />
                      <span>{article.timestamp}</span>
                    </span>
                    <span className="flex items-center space-x-1">
                      <BookOpen className="w-3 h-3" />
                      <span>{article.readTime}m</span>
                    </span>
                  </div>
                  <span className="flex items-center space-x-1">
                    <Eye className="w-3 h-3" />
                    <span>{(article.views / 1000).toFixed(1)}k</span>
                  </span>
                </div>

                <div className="flex flex-wrap gap-2">
                  {article.tags.slice(0, 2).map((tag, index) => (
                    <span key={index} className={`px-2 py-1 rounded text-xs font-medium ${
                      tag === 'Breaking' 
                        ? 'bg-red-500/20 text-red-300 border border-red-500/30'
                        : tag === 'Trending'
                        ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/30'
                        : 'bg-gray-600/20 text-gray-400 border border-gray-600/30'
                    }`}>
                      #{tag}
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}

        {!loading && filteredArticles.length === 0 && (
          <div className="text-center py-16 animate-fade-in">
            <Search className="w-16 h-16 text-gray-600 mx-auto mb-4" />
            <h3 className="text-xl font-semibold text-gray-400 mb-2">No articles found</h3>
            <p className="text-gray-500 text-sm">Try adjusting your search or filter criteria</p>
          </div>
        )}
      </div>
    </div>
  );
};
