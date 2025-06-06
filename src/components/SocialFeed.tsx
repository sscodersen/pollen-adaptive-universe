import React, { useState, useEffect } from 'react';
import { Heart, MessageCircle, Share2, MoreHorizontal, Eye, Sparkles, TrendingUp, Zap, Send, Image, Video, FileText, Code, BarChart3, Download, Play } from 'lucide-react';
import { pollenAI } from '../services/pollenAI';
import { significanceAlgorithm } from '../services/significanceAlgorithm';
import { aiChatService, type GenerationRequest, type ChatMessage, type ChatAttachment } from '../services/aiChatService';

interface Post {
  id: string;
  author: string;
  avatar: string;
  content: string;
  images?: string[];
  likes: number;
  comments: number;
  views: number;
  timestamp: string;
  type: 'text' | 'image' | 'video' | 'mixed';
  trending?: boolean;
  attachments?: ChatAttachment[];
}

interface SocialFeedProps {
  isGenerating?: boolean;
}

export const SocialFeed = ({ isGenerating = true }: SocialFeedProps) => {
  const [posts, setPosts] = useState<Post[]>([]);
  const [generatingPost, setGeneratingPost] = useState(false);
  const [chatInput, setChatInput] = useState('');
  const [selectedGenerationType, setSelectedGenerationType] = useState<GenerationRequest['type']>('reasoning');
  const [isProcessingChat, setIsProcessingChat] = useState(false);

  const generationTypes = [
    { type: 'photo' as const, icon: Image, label: 'Photo' },
    { type: 'video' as const, icon: Video, label: 'Video' },
    { type: 'task' as const, icon: Zap, label: 'Task' },
    { type: 'reasoning' as const, icon: BarChart3, label: 'Analysis' },
    { type: 'coding' as const, icon: Code, label: 'Code' },
    { type: 'pdf' as const, icon: FileText, label: 'PDF' },
    { type: 'blog' as const, icon: FileText, label: 'Blog' },
    { type: 'seo' as const, icon: TrendingUp, label: 'SEO' }
  ];

  useEffect(() => {
    if (!isGenerating) return;

    const generateContent = async () => {
      if (generatingPost) return;
      
      setGeneratingPost(true);
      try {
        // Get trending topics from significance algorithm
        const trendingTopics = significanceAlgorithm.getTrendingTopics();
        const randomTopic = trendingTopics[Math.floor(Math.random() * trendingTopics.length)];
        
        // Enhanced prompt for more diverse content generation
        const contentTypes = [
          `breaking news analysis about ${randomTopic}`,
          `actionable insights on ${randomTopic}`,
          `global impact assessment of ${randomTopic}`,
          `innovative solutions in ${randomTopic}`,
          `expert commentary on ${randomTopic}`,
          `practical applications of ${randomTopic}`,
          `emerging trends in ${randomTopic}`,
          `community success stories with ${randomTopic}`
        ];
        
        const randomContentType = contentTypes[Math.floor(Math.random() * contentTypes.length)];
        
        const response = await pollenAI.generate(
          `Create engaging social content about ${randomContentType}`,
          "social",
          true // Use significance filtering
        );
        
        const newPost: Post = {
          id: Date.now().toString(),
          author: generateRandomAuthor(),
          avatar: generateAvatar(),
          content: response.content,
          likes: Math.floor(Math.random() * 5000), // Higher engagement for significant content
          comments: Math.floor(Math.random() * 500),
          views: Math.floor(Math.random() * 50000),
          timestamp: formatTimestamp(new Date()),
          type: 'text',
          trending: response.significanceScore ? response.significanceScore > 8.5 : Math.random() > 0.6
        };
        
        setPosts(prev => [newPost, ...prev.slice(0, 19)]);
      } catch (error) {
        console.error('Failed to generate content:', error);
      }
      setGeneratingPost(false);
    };

    // Generate initial post after a delay
    const initialTimeout = setTimeout(generateContent, 3000);
    
    // Generate new posts every 60-90 seconds (slower as requested)
    const interval = setInterval(generateContent, Math.random() * 30000 + 60000);
    
    return () => {
      clearTimeout(initialTimeout);
      clearInterval(interval);
    };
  }, [isGenerating, generatingPost]);

  const generateRandomAuthor = () => {
    const names = [
      'Alex Chen', 'Maya Rodriguez', 'Jordan Smith', 'Riley Kim', 
      'Sam Johnson', 'Casey Brown', 'Avery Taylor', 'Quinn Davis',
      'Morgan Lee', 'Sage Wilson', 'River Martinez', 'Phoenix Garcia'
    ];
    return names[Math.floor(Math.random() * names.length)];
  };

  const generateAvatar = () => {
    const colors = [
      'bg-blue-500', 'bg-purple-500', 'bg-pink-500', 'bg-green-500', 
      'bg-orange-500', 'bg-cyan-500', 'bg-yellow-500', 'bg-red-500',
      'bg-indigo-500', 'bg-teal-500', 'bg-rose-500', 'bg-emerald-500'
    ];
    return colors[Math.floor(Math.random() * colors.length)];
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

  const handleChatSubmit = async () => {
    if (!chatInput.trim() || isProcessingChat) return;
    
    setIsProcessingChat(true);
    try {
      const request: GenerationRequest = {
        type: selectedGenerationType,
        prompt: chatInput
      };
      
      const response = await aiChatService.processGenerationRequest(request);
      
      // Convert chat response to social post
      const aiPost: Post = {
        id: Date.now().toString(),
        author: 'Pollen AI Assistant',
        avatar: 'bg-gradient-to-r from-cyan-500 to-purple-500',
        content: response.content,
        likes: Math.floor(Math.random() * 100),
        comments: Math.floor(Math.random() * 20),
        views: Math.floor(Math.random() * 1000),
        timestamp: formatTimestamp(new Date()),
        type: response.attachments && response.attachments.length > 0 ? 'mixed' : 'text',
        trending: true,
        attachments: response.attachments
      };
      
      setPosts(prev => [aiPost, ...prev.slice(0, 19)]);
      setChatInput('');
    } catch (error) {
      console.error('Error processing chat:', error);
    }
    setIsProcessingChat(false);
  };

  const handleDownload = (attachment: any) => {
    const blob = new Blob([attachment.content || ''], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = attachment.name;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="flex-1 flex flex-col">
      {/* Header */}
      <div className="p-6 border-b border-gray-700/50">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h1 className="text-2xl font-bold text-white">Social Feed</h1>
            <p className="text-gray-400">Anonymous AI-generated content continuously evolving</p>
          </div>
          <div className="flex items-center space-x-4">
            {generatingPost && (
              <div className="flex items-center space-x-2 text-green-400">
                <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
                <span className="text-sm">Generating...</span>
              </div>
            )}
            <span className="text-sm text-gray-400">{posts.length} posts</span>
          </div>
        </div>

        {/* Filter Tabs */}
        <div className="flex space-x-6">
          {['Recent', 'Trending', 'Discover', 'AI Generated'].map((tab, index) => (
            <button
              key={tab}
              className={`pb-2 text-sm font-medium transition-colors border-b-2 ${
                index === 0
                  ? 'text-blue-400 border-blue-400'
                  : 'text-gray-400 border-transparent hover:text-white'
              }`}
            >
              {tab}
            </button>
          ))}
        </div>
      </div>

      {/* AI Chat Input */}
      <div className="p-4 border-b border-gray-700/50 bg-gray-800/30">
        <div className="max-w-4xl mx-auto">
          <div className="flex items-center space-x-2 mb-3">
            <Sparkles className="w-5 h-5 text-cyan-400" />
            <span className="text-sm font-medium text-white">AI Assistant - Generate Content</span>
          </div>
          
          {/* Generation Type Selector */}
          <div className="flex flex-wrap gap-2 mb-3">
            {generationTypes.map(({ type, icon: Icon, label }) => (
              <button
                key={type}
                onClick={() => setSelectedGenerationType(type)}
                className={`flex items-center space-x-1 px-3 py-1 rounded-full text-xs font-medium transition-colors ${
                  selectedGenerationType === type
                    ? 'bg-cyan-500 text-white'
                    : 'bg-gray-700/50 text-gray-300 hover:bg-gray-600/50'
                }`}
              >
                <Icon className="w-3 h-3" />
                <span>{label}</span>
              </button>
            ))}
          </div>
          
          {/* Chat Input */}
          <div className="flex space-x-3">
            <input
              type="text"
              value={chatInput}
              onChange={(e) => setChatInput(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && handleChatSubmit()}
              placeholder={`Ask AI to generate ${generationTypes.find(t => t.type === selectedGenerationType)?.label.toLowerCase()}...`}
              className="flex-1 bg-gray-700/50 border border-gray-600/50 rounded-lg px-4 py-3 text-white placeholder-gray-400 focus:outline-none focus:border-cyan-500/50"
              disabled={isProcessingChat}
            />
            <button
              onClick={handleChatSubmit}
              disabled={!chatInput.trim() || isProcessingChat}
              className="bg-gradient-to-r from-cyan-500 to-purple-500 px-6 py-3 rounded-lg font-medium text-white transition-all disabled:opacity-50 flex items-center space-x-2"
            >
              {isProcessingChat ? (
                <>
                  <div className="w-4 h-4 border-2 border-white/20 border-t-white rounded-full animate-spin" />
                  <span>Generating...</span>
                </>
              ) : (
                <>
                  <Send className="w-4 h-4" />
                  <span>Generate</span>
                </>
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Feed */}
      <div className="flex-1 overflow-y-auto p-6 space-y-4">
        {posts.map((post) => (
          <div key={post.id} className="bg-gray-800/50 rounded-2xl border border-gray-700/50 p-6 hover:bg-gray-800/70 transition-colors">
            {/* Post Header */}
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center space-x-3">
                <div className={`w-12 h-12 ${post.avatar} rounded-full flex items-center justify-center text-white font-medium text-lg`}>
                  {post.author.split(' ').map(n => n[0]).join('')}
                </div>
                <div>
                  <div className="flex items-center space-x-2">
                    <h3 className="font-semibold text-white">{post.author}</h3>
                    {post.trending && (
                      <div className="flex items-center space-x-1 px-2 py-0.5 bg-orange-500/20 text-orange-400 text-xs rounded-full">
                        <TrendingUp className="w-3 h-3" />
                        <span>Trending</span>
                      </div>
                    )}
                  </div>
                  <p className="text-sm text-gray-400">{post.timestamp}</p>
                </div>
              </div>
              <button className="text-gray-400 hover:text-white transition-colors">
                <MoreHorizontal className="w-5 h-5" />
              </button>
            </div>

            {/* Post Content */}
            <div className="mb-6">
              <div className="text-gray-100 leading-relaxed whitespace-pre-line">
                {post.content}
              </div>
              
              {/* Attachments */}
              {post.attachments && post.attachments.length > 0 && (
                <div className="mt-4 space-y-3">
                  {post.attachments.map((attachment) => (
                    <div key={attachment.id} className="bg-gray-700/30 rounded-lg p-4 border border-gray-600/30">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-3">
                          {attachment.type === 'image' && <Image className="w-5 h-5 text-blue-400" />}
                          {attachment.type === 'video' && <Video className="w-5 h-5 text-red-400" />}
                          {attachment.type === 'code' && <Code className="w-5 h-5 text-green-400" />}
                          {attachment.type === 'document' && <FileText className="w-5 h-5 text-yellow-400" />}
                          {attachment.type === 'analysis' && <BarChart3 className="w-5 h-5 text-purple-400" />}
                          
                          <div>
                            <p className="text-sm font-medium text-white">{attachment.name}</p>
                            <p className="text-xs text-gray-400 capitalize">{attachment.type} • Ready for use</p>
                          </div>
                        </div>
                        
                        <div className="flex items-center space-x-2">
                          {attachment.type === 'video' && (
                            <button className="p-2 bg-gray-600/50 rounded-lg hover:bg-gray-600 transition-colors">
                              <Play className="w-4 h-4 text-white" />
                            </button>
                          )}
                          {attachment.downloadable && (
                            <button
                              onClick={() => handleDownload(attachment)}
                              className="p-2 bg-gray-600/50 rounded-lg hover:bg-gray-600 transition-colors"
                            >
                              <Download className="w-4 h-4 text-white" />
                            </button>
                          )}
                        </div>
                      </div>
                      
                      {attachment.type === 'image' && attachment.url && (
                        <div className="mt-3">
                          <img 
                            src={attachment.url} 
                            alt={attachment.name}
                            className="rounded-lg max-h-64 w-full object-cover"
                          />
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Post Actions */}
            <div className="flex items-center justify-between pt-4 border-t border-gray-700/50">
              <div className="flex items-center space-x-6">
                <button className="flex items-center space-x-2 text-gray-400 hover:text-red-400 transition-colors group">
                  <Heart className="w-5 h-5 group-hover:scale-110 transition-transform" />
                  <span className="text-sm font-medium">{post.likes.toLocaleString()}</span>
                </button>
                <button className="flex items-center space-x-2 text-gray-400 hover:text-blue-400 transition-colors group">
                  <MessageCircle className="w-5 h-5 group-hover:scale-110 transition-transform" />
                  <span className="text-sm font-medium">{post.comments}</span>
                </button>
                <button className="flex items-center space-x-2 text-gray-400 hover:text-green-400 transition-colors group">
                  <Share2 className="w-5 h-5 group-hover:scale-110 transition-transform" />
                  <span className="text-sm font-medium">Share</span>
                </button>
              </div>
              <div className="flex items-center space-x-2 text-gray-400">
                <Eye className="w-4 h-4" />
                <span className="text-sm">{post.views.toLocaleString()}</span>
              </div>
            </div>
          </div>
        ))}

        {posts.length === 0 && (
          <div className="text-center py-16">
            <div className="w-20 h-20 bg-gradient-to-r from-blue-400 to-purple-500 rounded-full flex items-center justify-center mx-auto mb-6">
              <Sparkles className="w-10 h-10 text-white animate-pulse" />
            </div>
            <h3 className="text-xl font-semibold text-white mb-2">Pollen is Thinking...</h3>
            <p className="text-gray-400 max-w-md mx-auto">
              Our AI is analyzing patterns and generating engaging social content tailored to emerging trends and community interests.
            </p>
          </div>
        )}
      </div>
    </div>
  );
};
