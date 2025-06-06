
import React, { useState, useEffect } from 'react';
import { Heart, MessageCircle, Share2, MoreHorizontal, Eye, Sparkles, TrendingUp, Zap, Send, Image, Video, FileText, Code, BarChart3, Download, Play, Globe, Star, Clock, Paperclip } from 'lucide-react';
import { pollenAI } from '../services/pollenAI';
import { significanceAlgorithm } from '../services/significanceAlgorithm';

interface Post {
  id: string;
  author: string;
  username: string;
  avatar: string;
  content: string;
  likes: number;
  comments: number;
  views: number;
  timestamp: string;
  type: 'breakthrough' | 'analysis' | 'trending' | 'tutorial' | 'discussion' | 'innovation' | 'personal' | 'project' | 'review';
  category: 'tech' | 'science' | 'social' | 'business' | 'health' | 'environment' | 'design' | 'creative';
  significance: number;
  trending?: boolean;
  media?: {
    type: 'image' | 'video' | 'code' | 'document';
    url?: string;
    preview?: string;
  };
  replies?: Array<{
    id: string;
    author: string;
    username: string;
    avatar: string;
    content: string;
    timestamp: string;
  }>;
}

interface SocialFeedProps {
  isGenerating?: boolean;
}

export const SocialFeed = ({ isGenerating = true }: SocialFeedProps) => {
  const [posts, setPosts] = useState<Post[]>([]);
  const [generatingPost, setGeneratingPost] = useState(false);
  const [commentTexts, setCommentTexts] = useState<{[key: string]: string}>({});

  const realUsers = [
    { name: "Sarah Chen", username: "sarahc_dev", avatar: "bg-gradient-to-r from-purple-400 to-pink-500" },
    { name: "Marcus Rodriguez", username: "marcus_builds", avatar: "bg-gradient-to-r from-blue-400 to-cyan-500" },
    { name: "Emma Thompson", username: "emmathomps", avatar: "bg-gradient-to-r from-green-400 to-emerald-500" },
    { name: "Alex Kim", username: "alexkim_ai", avatar: "bg-gradient-to-r from-orange-400 to-red-500" },
    { name: "Jordan Lee", username: "jordanlee_", avatar: "bg-gradient-to-r from-indigo-400 to-purple-500" },
    { name: "Taylor Swift", username: "taylorswift", avatar: "bg-gradient-to-r from-pink-400 to-rose-500" },
    { name: "David Zhang", username: "davidz_tech", avatar: "bg-gradient-to-r from-teal-400 to-blue-500" },
    { name: "Maya Patel", username: "mayapatel", avatar: "bg-gradient-to-r from-yellow-400 to-orange-500" },
    { name: "Ryan Wilson", username: "ryanwilson", avatar: "bg-gradient-to-r from-gray-400 to-slate-500" },
    { name: "Lisa Johnson", username: "lisaj_design", avatar: "bg-gradient-to-r from-violet-400 to-purple-500" }
  ];

  const contentTemplates = {
    breakthrough: [
      "Just discovered something incredible in {field}. The implications for {application} are mind-blowing. This could change everything we know about {topic}.",
      "Breakthrough moment: After months of research, we've cracked the code on {technology}. The results are beyond what we expected. Thread below 🧵",
      "Holy grail found! {discovery} is now possible thanks to {method}. This opens doors to applications we never thought feasible.",
      "Game changer alert: {innovation} just became reality. The potential impact on {industry} is enormous. Sharing details..."
    ],
    analysis: [
      "Deep dive analysis: I've been studying {trend} for the past 6 months. Here's what the data reveals about {outcome}. Key insights:",
      "After analyzing {dataset}, some fascinating patterns emerge. {finding} suggests that {implication}. Full breakdown:",
      "Comprehensive review of {topic}: The numbers don't lie. {statistic} indicates {conclusion}. What this means for {future}:",
      "Data-driven insights on {subject}: {percentage} of {group} show {behavior}. This trend suggests {prediction}."
    ],
    personal: [
      "Personal milestone: Just {achievement}! The journey taught me {lesson}. Grateful for {support} along the way.",
      "Reflecting on {experience}: {duration} ago, I decided to {action}. Today, {outcome}. The key was {strategy}.",
      "Life update: {change} has been transformative. {result} exceeded expectations. Sharing my approach:",
      "Lessons learned from {project}: {insight} was the game-changer. If you're considering {similar}, here's my advice..."
    ],
    project: [
      "Project showcase: Spent {timeframe} building {project}. Features include {feature1}, {feature2}, and {feature3}. Feedback welcome!",
      "Side project reveal: {name} is now live! Built with {technology} to solve {problem}. Check it out: {demo}",
      "Open source contribution: Just released {tool} for the {community} community. Enables {functionality}. Hoping it helps others!",
      "Collaboration opportunity: Working on {initiative} with amazing team. Looking for {skill} expertise. DM if interested!"
    ]
  };

  useEffect(() => {
    if (!isGenerating) return;

    const generateContent = async () => {
      if (generatingPost) return;
      
      setGeneratingPost(true);
      try {
        const types = Object.keys(contentTemplates) as (keyof typeof contentTemplates)[];
        const categories = ['tech', 'science', 'social', 'business', 'health', 'environment', 'design', 'creative'] as const;
        
        const randomType = types[Math.floor(Math.random() * types.length)];
        const randomCategory = categories[Math.floor(Math.random() * categories.length)];
        const randomUser = realUsers[Math.floor(Math.random() * realUsers.length)];
        
        const template = contentTemplates[randomType][Math.floor(Math.random() * contentTemplates[randomType].length)];
        
        // Generate dynamic content using Pollen AI
        const response = await pollenAI.generate(
          `Create a natural, engaging ${randomType} social media post about ${randomCategory} topics. Make it sound human and authentic.`,
          "social",
          true
        );
        
        const newPost: Post = {
          id: Date.now().toString(),
          author: randomUser.name,
          username: randomUser.username,
          avatar: randomUser.avatar,
          content: response.content,
          likes: Math.floor(Math.random() * 2500) + 100,
          comments: Math.floor(Math.random() * 150) + 10,
          views: Math.floor(Math.random() * 50000) + 1000,
          timestamp: formatTimestamp(new Date()),
          type: randomType,
          category: randomCategory,
          significance: response.significanceScore || 8.5,
          trending: response.significanceScore ? response.significanceScore > 9.0 : Math.random() > 0.8,
          media: Math.random() > 0.7 ? generateMedia(randomType) : undefined,
          replies: Math.random() > 0.6 ? generateReplies() : undefined
        };
        
        setPosts(prev => [newPost, ...prev.slice(0, 29)]);
      } catch (error) {
        console.error('Failed to generate content:', error);
      }
      setGeneratingPost(false);
    };

    const initialTimeout = setTimeout(generateContent, 1000);
    const interval = setInterval(generateContent, Math.random() * 25000 + 35000);
    
    return () => {
      clearTimeout(initialTimeout);
      clearInterval(interval);
    };
  }, [isGenerating, generatingPost]);

  const generateReplies = () => {
    const numReplies = Math.floor(Math.random() * 3) + 1;
    const replies = [];
    
    for (let i = 0; i < numReplies; i++) {
      const randomUser = realUsers[Math.floor(Math.random() * realUsers.length)];
      replies.push({
        id: `reply-${Date.now()}-${i}`,
        author: randomUser.name,
        username: randomUser.username,
        avatar: randomUser.avatar,
        content: generateReplyContent(),
        timestamp: formatTimestamp(new Date(Date.now() - Math.random() * 3600000))
      });
    }
    
    return replies;
  };

  const generateReplyContent = () => {
    const replyTypes = [
      "Great insights! This aligns with what I've been seeing in my work.",
      "Fascinating approach. Have you considered the implications for accessibility?",
      "This is exactly what we needed. Thanks for sharing!",
      "Incredible work! The attention to detail is impressive.",
      "Love this direction. Would be interested in collaborating on similar projects.",
      "Mind-blowing results! How did you handle the technical challenges?",
      "This could be revolutionary. Any plans for open-sourcing?",
      "Brilliant execution! The user experience looks seamless."
    ];
    
    return replyTypes[Math.floor(Math.random() * replyTypes.length)];
  };

  const generateMedia = (type: string) => {
    const mediaTypes = {
      breakthrough: { type: 'image' as const, preview: 'Research visualization' },
      analysis: { type: 'document' as const, preview: 'Data analysis report' },
      trending: { type: 'video' as const, preview: 'Trend showcase video' },
      tutorial: { type: 'code' as const, preview: 'Implementation example' },
      discussion: { type: 'document' as const, preview: 'Discussion summary' },
      innovation: { type: 'image' as const, preview: 'Innovation showcase' },
      personal: { type: 'image' as const, preview: 'Personal project' },
      project: { type: 'code' as const, preview: 'Project demo' },
      review: { type: 'document' as const, preview: 'Review summary' }
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

  const handleCommentChange = (postId: string, text: string) => {
    setCommentTexts(prev => ({ ...prev, [postId]: text }));
  };

  const handleSubmitComment = (postId: string) => {
    const text = commentTexts[postId]?.trim();
    if (!text) return;
    
    setCommentTexts(prev => ({ ...prev, [postId]: '' }));
  };

  return (
    <div className="flex-1 flex flex-col bg-gray-950">
      {/* Header */}
      <div className="p-6 border-b border-gray-800/50 bg-gray-900/50 backdrop-blur-sm">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h1 className="text-3xl font-bold text-white mb-2">Social Feed</h1>
            <p className="text-gray-400">Real conversations • AI-curated insights • Global community</p>
          </div>
          <div className="flex items-center space-x-4">
            {generatingPost && (
              <div className="flex items-center space-x-2 text-cyan-400">
                <div className="w-2 h-2 bg-cyan-400 rounded-full animate-pulse" />
                <span className="text-sm font-medium">Generating content...</span>
              </div>
            )}
            <div className="text-right">
              <div className="text-2xl font-bold text-white">{posts.length}</div>
              <div className="text-xs text-gray-400">Active posts</div>
            </div>
          </div>
        </div>

        {/* Filter Tabs */}
        <div className="flex space-x-8">
          {['All Posts', 'Trending', 'Breakthroughs', 'Projects', 'Discussions'].map((tab, index) => (
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

      {/* Feed */}
      <div className="flex-1 overflow-y-auto p-6 space-y-4">
        {posts.map((post) => (
          <article key={post.id} className="bg-gray-900/80 rounded-2xl border border-gray-800/50 p-6 hover:bg-gray-900/90 transition-all duration-200 backdrop-blur-sm">
            {/* Post Header */}
            <div className="flex items-start justify-between mb-4">
              <div className="flex items-center space-x-3">
                <div className={`w-12 h-12 ${post.avatar} rounded-full flex items-center justify-center text-white font-semibold shadow-lg`}>
                  {post.author.split(' ').map(n => n[0]).join('')}
                </div>
                <div>
                  <div className="flex items-center space-x-2 mb-1">
                    <h3 className="font-bold text-white">{post.author}</h3>
                    <span className="text-gray-400 text-sm">@{post.username}</span>
                    <span className="text-gray-500 text-sm">•</span>
                    <span className="text-gray-400 text-sm">{post.timestamp}</span>
                  </div>
                  {post.trending && (
                    <div className="flex items-center space-x-1 text-orange-400 text-xs">
                      <TrendingUp className="w-3 h-3" />
                      <span>Trending</span>
                    </div>
                  )}
                </div>
              </div>
              <button className="text-gray-400 hover:text-white transition-colors p-2 hover:bg-gray-800/50 rounded-lg">
                <MoreHorizontal className="w-5 h-5" />
              </button>
            </div>

            {/* Post Content */}
            <div className="mb-4">
              <p className="text-gray-100 leading-relaxed whitespace-pre-line">
                {post.content}
              </p>
            </div>

            {/* Media Content */}
            {post.media && (
              <div className="bg-gray-800/30 rounded-xl p-4 border border-gray-700/30 mb-4">
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
                      <button className="p-2 bg-gray-700/50 rounded-lg hover:bg-gray-700 transition-colors">
                        <Play className="w-4 h-4 text-white" />
                      </button>
                    )}
                    <button className="p-2 bg-gray-700/50 rounded-lg hover:bg-gray-700 transition-colors">
                      <Download className="w-4 h-4 text-white" />
                    </button>
                  </div>
                </div>
              </div>
            )}

            {/* Replies */}
            {post.replies && post.replies.length > 0 && (
              <div className="mb-4 space-y-3 border-l-2 border-gray-700/30 pl-4">
                {post.replies.map((reply) => (
                  <div key={reply.id} className="flex items-start space-x-3 p-3 bg-gray-800/30 rounded-lg">
                    <div className={`w-8 h-8 ${reply.avatar} rounded-full flex items-center justify-center text-white text-sm font-medium`}>
                      {reply.author.split(' ').map(n => n[0]).join('')}
                    </div>
                    <div className="flex-1">
                      <div className="flex items-center space-x-2 mb-1">
                        <span className="font-medium text-white text-sm">{reply.author}</span>
                        <span className="text-gray-400 text-xs">@{reply.username}</span>
                        <span className="text-gray-500 text-xs">•</span>
                        <span className="text-gray-400 text-xs">{reply.timestamp}</span>
                      </div>
                      <p className="text-gray-200 text-sm">{reply.content}</p>
                    </div>
                  </div>
                ))}
              </div>
            )}

            {/* Post Actions */}
            <div className="flex items-center justify-between pt-4 border-t border-gray-800/50">
              <div className="flex items-center space-x-6">
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
                <span className="text-sm">{post.views.toLocaleString()}</span>
              </div>
            </div>

            {/* Comment Input */}
            <div className="flex items-center space-x-3 mt-4">
              <div className="w-8 h-8 bg-gray-600 rounded-full flex items-center justify-center text-white text-sm font-medium">
                U
              </div>
              <div className="flex-1 flex items-center space-x-2 bg-gray-800/50 rounded-xl px-4 py-2 border border-gray-700/30 focus-within:border-cyan-500/50 transition-colors">
                <input
                  type="text"
                  placeholder="Write a comment..."
                  value={commentTexts[post.id] || ''}
                  onChange={(e) => handleCommentChange(post.id, e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && handleSubmitComment(post.id)}
                  className="flex-1 bg-transparent text-white placeholder-gray-400 outline-none text-sm"
                />
                <button className="p-1 hover:bg-gray-700/50 rounded transition-colors">
                  <Paperclip className="w-4 h-4 text-gray-400" />
                </button>
                <button
                  onClick={() => handleSubmitComment(post.id)}
                  className="px-3 py-1 bg-cyan-600 hover:bg-cyan-700 text-white rounded-lg text-sm font-medium transition-colors"
                >
                  Post
                </button>
              </div>
            </div>
          </article>
        ))}

        {posts.length === 0 && (
          <div className="text-center py-20">
            <div className="w-24 h-24 bg-gradient-to-r from-cyan-500 to-purple-500 rounded-full flex items-center justify-center mx-auto mb-8">
              <Sparkles className="w-12 h-12 text-white animate-pulse" />
            </div>
            <h3 className="text-2xl font-bold text-white mb-4">Social Feed Loading...</h3>
            <p className="text-gray-400 max-w-lg mx-auto text-lg">
              Pollen AI is generating authentic conversations and insights from our global community of creators, developers, and innovators.
            </p>
          </div>
        )}
      </div>
    </div>
  );
};
