
import React, { useState, useEffect } from 'react';
import { Heart, MessageCircle, Share2, MoreHorizontal, Eye, Sparkles, TrendingUp, Zap, Send, Image, Video, FileText, Code, BarChart3, Download, Play, Globe, Star, Clock } from 'lucide-react';
import { pollenAI } from '../services/pollenAI';
import { significanceAlgorithm } from '../services/significanceAlgorithm';

interface Post {
  id: string;
  author: string;
  avatar: string;
  content: string;
  likes: number;
  comments: number;
  views: number;
  timestamp: string;
  type: 'breakthrough' | 'analysis' | 'trending' | 'tutorial' | 'discussion' | 'innovation';
  category: 'tech' | 'science' | 'social' | 'business' | 'health' | 'environment';
  significance: number;
  trending?: boolean;
  media?: {
    type: 'image' | 'video' | 'code' | 'document';
    url?: string;
    preview?: string;
  };
}

interface SocialFeedProps {
  isGenerating?: boolean;
}

export const SocialFeed = ({ isGenerating = true }: SocialFeedProps) => {
  const [posts, setPosts] = useState<Post[]>([]);
  const [generatingPost, setGeneratingPost] = useState(false);

  const postTemplates = {
    breakthrough: {
      prompts: [
        "revolutionary discovery in quantum computing",
        "breakthrough medical treatment with 95% success rate",
        "new sustainable energy technology",
        "AI advancement changing industry standards"
      ],
      authors: ["Dr. Sarah Chen", "Research Team Alpha", "Innovation Lab", "Tech Institute"]
    },
    analysis: {
      prompts: [
        "economic impact analysis of emerging technologies",
        "social media trends reshaping communication",
        "climate change solutions effectiveness study",
        "market disruption from AI automation"
      ],
      authors: ["Economic Analyst", "Trend Researcher", "Data Scientist", "Market Expert"]
    },
    trending: {
      prompts: [
        "viral innovation spreading across industries",
        "global movement gaining momentum",
        "technology adoption accelerating worldwide",
        "cultural shift in digital behavior"
      ],
      authors: ["Trend Spotter", "Cultural Analyst", "Tech Observer", "Social Researcher"]
    },
    tutorial: {
      prompts: [
        "step-by-step guide to implementing AI solutions",
        "practical framework for digital transformation",
        "beginner's approach to quantum concepts",
        "actionable strategies for innovation"
      ],
      authors: ["Tech Educator", "Innovation Guide", "Learning Expert", "Skill Builder"]
    },
    discussion: {
      prompts: [
        "ethical implications of emerging technology",
        "future of work in automated society",
        "privacy concerns in digital age",
        "sustainability versus growth debate"
      ],
      authors: ["Ethics Professor", "Future Analyst", "Policy Expert", "Thought Leader"]
    },
    innovation: {
      prompts: [
        "startup revolutionizing traditional industry",
        "open-source project changing development",
        "collaborative platform enabling breakthroughs",
        "community-driven innovation success"
      ],
      authors: ["Startup Founder", "Open Source Dev", "Innovation Hub", "Community Leader"]
    }
  };

  useEffect(() => {
    if (!isGenerating) return;

    const generateContent = async () => {
      if (generatingPost) return;
      
      setGeneratingPost(true);
      try {
        const types = Object.keys(postTemplates) as (keyof typeof postTemplates)[];
        const categories = ['tech', 'science', 'social', 'business', 'health', 'environment'] as const;
        
        const randomType = types[Math.floor(Math.random() * types.length)];
        const randomCategory = categories[Math.floor(Math.random() * categories.length)];
        const template = postTemplates[randomType];
        const randomPrompt = template.prompts[Math.floor(Math.random() * template.prompts.length)];
        const randomAuthor = template.authors[Math.floor(Math.random() * template.authors.length)];
        
        const response = await pollenAI.generate(
          `Create ${randomType} content about ${randomPrompt} for ${randomCategory} category`,
          "social",
          true
        );
        
        const newPost: Post = {
          id: Date.now().toString(),
          author: randomAuthor,
          avatar: generateAvatar(randomCategory),
          content: response.content,
          likes: Math.floor(Math.random() * 10000) + 500,
          comments: Math.floor(Math.random() * 1000) + 50,
          views: Math.floor(Math.random() * 100000) + 5000,
          timestamp: formatTimestamp(new Date()),
          type: randomType,
          category: randomCategory,
          significance: response.significanceScore || 8.5,
          trending: response.significanceScore ? response.significanceScore > 9.0 : Math.random() > 0.7,
          media: Math.random() > 0.6 ? generateMedia(randomType) : undefined
        };
        
        setPosts(prev => [newPost, ...prev.slice(0, 19)]);
      } catch (error) {
        console.error('Failed to generate content:', error);
      }
      setGeneratingPost(false);
    };

    const initialTimeout = setTimeout(generateContent, 2000);
    const interval = setInterval(generateContent, Math.random() * 20000 + 30000);
    
    return () => {
      clearTimeout(initialTimeout);
      clearInterval(interval);
    };
  }, [isGenerating, generatingPost]);

  const generateAvatar = (category: string) => {
    const avatarMap = {
      tech: 'bg-gradient-to-r from-blue-500 to-cyan-500',
      science: 'bg-gradient-to-r from-green-500 to-emerald-500',
      social: 'bg-gradient-to-r from-purple-500 to-pink-500',
      business: 'bg-gradient-to-r from-yellow-500 to-orange-500',
      health: 'bg-gradient-to-r from-red-500 to-rose-500',
      environment: 'bg-gradient-to-r from-teal-500 to-green-500'
    };
    return avatarMap[category as keyof typeof avatarMap] || 'bg-gradient-to-r from-gray-500 to-slate-500';
  };

  const generateMedia = (type: string) => {
    const mediaTypes = {
      breakthrough: { type: 'image' as const, preview: 'Research visualization' },
      analysis: { type: 'document' as const, preview: 'Data analysis report' },
      trending: { type: 'video' as const, preview: 'Trend showcase video' },
      tutorial: { type: 'code' as const, preview: 'Implementation example' },
      discussion: { type: 'document' as const, preview: 'Discussion summary' },
      innovation: { type: 'image' as const, preview: 'Innovation showcase' }
    };
    return mediaTypes[type as keyof typeof mediaTypes] || { type: 'image' as const, preview: 'Visual content' };
  };

  const formatTimestamp = (date: Date) => {
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    
    if (diffMins < 1) return 'now';
    if (diffMins < 60) return `${diffMins}m`;
    if (diffMins < 1440) return `${Math.floor(diffMins / 60)}h`;
    return `${Math.floor(diffMins / 1440)}d`;
  };

  const getCategoryColor = (category: string) => {
    const colors = {
      tech: 'text-blue-400 bg-blue-400/10',
      science: 'text-green-400 bg-green-400/10',
      social: 'text-purple-400 bg-purple-400/10',
      business: 'text-yellow-400 bg-yellow-400/10',
      health: 'text-red-400 bg-red-400/10',
      environment: 'text-teal-400 bg-teal-400/10'
    };
    return colors[category as keyof typeof colors] || 'text-gray-400 bg-gray-400/10';
  };

  const getTypeIcon = (type: string) => {
    const icons = {
      breakthrough: Sparkles,
      analysis: BarChart3,
      trending: TrendingUp,
      tutorial: FileText,
      discussion: MessageCircle,
      innovation: Zap
    };
    const IconComponent = icons[type as keyof typeof icons] || Sparkles;
    return <IconComponent className="w-4 h-4" />;
  };

  return (
    <div className="flex-1 flex flex-col bg-gray-900">
      {/* Enhanced Header */}
      <div className="p-6 border-b border-gray-700/50 bg-gradient-to-r from-gray-800/50 to-gray-900/50">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h1 className="text-3xl font-bold text-white mb-2">Pollen Social Intelligence</h1>
            <p className="text-gray-400">AI-curated high-significance content • Live global analysis</p>
          </div>
          <div className="flex items-center space-x-4">
            {generatingPost && (
              <div className="flex items-center space-x-2 text-cyan-400">
                <div className="w-3 h-3 bg-cyan-400 rounded-full animate-pulse" />
                <span className="text-sm font-medium">Generating insights...</span>
              </div>
            )}
            <div className="text-right">
              <div className="text-2xl font-bold text-white">{posts.length}</div>
              <div className="text-xs text-gray-400">Active posts</div>
            </div>
          </div>
        </div>

        {/* Enhanced Filter Tabs */}
        <div className="flex space-x-8">
          {['All Content', 'Breakthroughs', 'Analysis', 'Trending', 'Tutorials', 'Discussions'].map((tab, index) => (
            <button
              key={tab}
              className={`pb-3 text-sm font-medium transition-all border-b-2 ${
                index === 0
                  ? 'text-cyan-400 border-cyan-400'
                  : 'text-gray-400 border-transparent hover:text-white hover:border-gray-600'
              }`}
            >
              {tab}
            </button>
          ))}
        </div>
      </div>

      {/* Enhanced Feed */}
      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {posts.map((post) => (
          <article key={post.id} className="bg-gradient-to-br from-gray-800/60 to-gray-900/60 rounded-2xl border border-gray-700/50 p-6 hover:border-gray-600/50 transition-all duration-300 backdrop-blur-sm">
            {/* Enhanced Post Header */}
            <div className="flex items-start justify-between mb-5">
              <div className="flex items-center space-x-4">
                <div className={`w-14 h-14 ${post.avatar} rounded-xl flex items-center justify-center text-white font-bold text-lg shadow-lg`}>
                  {post.author.split(' ').map(n => n[0]).join('').slice(0, 2)}
                </div>
                <div>
                  <div className="flex items-center space-x-3 mb-1">
                    <h3 className="font-bold text-white text-lg">{post.author}</h3>
                    <div className={`flex items-center space-x-1 px-2 py-1 rounded-full text-xs font-medium ${getCategoryColor(post.category)}`}>
                      {getTypeIcon(post.type)}
                      <span className="capitalize">{post.type}</span>
                    </div>
                  </div>
                  <div className="flex items-center space-x-3">
                    <p className="text-sm text-gray-400">{post.timestamp}</p>
                    <div className="flex items-center space-x-1">
                      <Star className="w-4 h-4 text-yellow-400" />
                      <span className="text-sm font-medium text-yellow-400">{post.significance.toFixed(1)}</span>
                    </div>
                    {post.trending && (
                      <div className="flex items-center space-x-1 px-2 py-1 bg-orange-500/20 text-orange-400 text-xs rounded-full">
                        <TrendingUp className="w-3 h-3" />
                        <span>Trending</span>
                      </div>
                    )}
                  </div>
                </div>
              </div>
              <button className="text-gray-400 hover:text-white transition-colors p-2 hover:bg-gray-700/50 rounded-lg">
                <MoreHorizontal className="w-5 h-5" />
              </button>
            </div>

            {/* Enhanced Post Content */}
            <div className="mb-6">
              <div className="text-gray-100 leading-relaxed text-lg whitespace-pre-line mb-4">
                {post.content}
              </div>
              
              {/* Media Content */}
              {post.media && (
                <div className="bg-gray-700/30 rounded-xl p-4 border border-gray-600/30">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                      {post.media.type === 'image' && <Image className="w-5 h-5 text-blue-400" />}
                      {post.media.type === 'video' && <Video className="w-5 h-5 text-red-400" />}
                      {post.media.type === 'code' && <Code className="w-5 h-5 text-green-400" />}
                      {post.media.type === 'document' && <FileText className="w-5 h-5 text-yellow-400" />}
                      
                      <div>
                        <p className="text-sm font-medium text-white">{post.media.preview}</p>
                        <p className="text-xs text-gray-400 capitalize">{post.media.type} content</p>
                      </div>
                    </div>
                    
                    <div className="flex items-center space-x-2">
                      {post.media.type === 'video' && (
                        <button className="p-2 bg-gray-600/50 rounded-lg hover:bg-gray-600 transition-colors">
                          <Play className="w-4 h-4 text-white" />
                        </button>
                      )}
                      <button className="p-2 bg-gray-600/50 rounded-lg hover:bg-gray-600 transition-colors">
                        <Download className="w-4 h-4 text-white" />
                      </button>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Enhanced Post Actions */}
            <div className="flex items-center justify-between pt-4 border-t border-gray-700/50">
              <div className="flex items-center space-x-8">
                <button className="flex items-center space-x-2 text-gray-400 hover:text-red-400 transition-colors group">
                  <Heart className="w-5 h-5 group-hover:scale-110 transition-transform" />
                  <span className="text-sm font-medium">{post.likes.toLocaleString()}</span>
                </button>
                <button className="flex items-center space-x-2 text-gray-400 hover:text-blue-400 transition-colors group">
                  <MessageCircle className="w-5 h-5 group-hover:scale-110 transition-transform" />
                  <span className="text-sm font-medium">{post.comments.toLocaleString()}</span>
                </button>
                <button className="flex items-center space-x-2 text-gray-400 hover:text-green-400 transition-colors group">
                  <Share2 className="w-5 h-5 group-hover:scale-110 transition-transform" />
                  <span className="text-sm font-medium">Share</span>
                </button>
              </div>
              <div className="flex items-center space-x-2 text-gray-400">
                <Eye className="w-4 h-4" />
                <span className="text-sm">{post.views.toLocaleString()} views</span>
              </div>
            </div>
          </article>
        ))}

        {posts.length === 0 && (
          <div className="text-center py-20">
            <div className="w-24 h-24 bg-gradient-to-r from-cyan-500 to-purple-500 rounded-full flex items-center justify-center mx-auto mb-8">
              <Sparkles className="w-12 h-12 text-white animate-pulse" />
            </div>
            <h3 className="text-2xl font-bold text-white mb-4">Pollen Intelligence Initializing...</h3>
            <p className="text-gray-400 max-w-lg mx-auto text-lg">
              Our AI is analyzing global patterns, trending topics, and high-significance content across multiple domains to generate meaningful insights.
            </p>
          </div>
        )}
      </div>
    </div>
  );
};
