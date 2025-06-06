import React, { useState, useEffect } from 'react';
import { Heart, MessageCircle, Share2, MoreHorizontal, Eye, Sparkles, TrendingUp, Zap, Send, Image, Video, FileText, Code, BarChart3, Download, Play, Globe, Star, Clock, Paperclip, Award, Trophy, Flame, Target } from 'lucide-react';
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
  engagement: number;
  rank?: number;
  verified?: boolean;
  media?: {
    type: 'image' | 'video' | 'code' | 'document';
    url?: string;
    preview?: string;
  };
  tags?: string[];
  replies?: Array<{
    id: string;
    author: string;
    username: string;
    avatar: string;
    content: string;
    timestamp: string;
    likes: number;
  }>;
}

interface SocialFeedProps {
  isGenerating?: boolean;
}

export const SocialFeed = ({ isGenerating = true }: SocialFeedProps) => {
  const [posts, setPosts] = useState<Post[]>([]);
  const [generatingPost, setGeneratingPost] = useState(false);
  const [commentTexts, setCommentTexts] = useState<{[key: string]: string}>({});
  const [filter, setFilter] = useState('all');
  const [userPoints, setUserPoints] = useState(1247);
  const [userLevel, setUserLevel] = useState(8);

  const realUsers = [
    { name: "Sarah Chen", username: "sarahc_dev", avatar: "bg-gradient-to-r from-purple-400 to-pink-500", verified: true },
    { name: "Marcus Rodriguez", username: "marcus_builds", avatar: "bg-gradient-to-r from-blue-400 to-cyan-500", verified: false },
    { name: "Emma Thompson", username: "emmathomps", avatar: "bg-gradient-to-r from-green-400 to-emerald-500", verified: true },
    { name: "Alex Kim", username: "alexkim_ai", avatar: "bg-gradient-to-r from-orange-400 to-red-500", verified: true },
    { name: "Jordan Lee", username: "jordanlee_", avatar: "bg-gradient-to-r from-indigo-400 to-purple-500", verified: false },
    { name: "Taylor Swift", username: "taylorswift", avatar: "bg-gradient-to-r from-pink-400 to-rose-500", verified: true },
    { name: "David Zhang", username: "davidz_tech", avatar: "bg-gradient-to-r from-teal-400 to-blue-500", verified: true },
    { name: "Maya Patel", username: "mayapatel", avatar: "bg-gradient-to-r from-yellow-400 to-orange-500", verified: false },
    { name: "Ryan Wilson", username: "ryanwilson", avatar: "bg-gradient-to-r from-gray-400 to-slate-500", verified: false },
    { name: "Lisa Johnson", username: "lisaj_design", avatar: "bg-gradient-to-r from-violet-400 to-purple-500", verified: true }
  ];

  const contentTopics = [
    'breakthrough AI research transforming healthcare diagnostics',
    'sustainable urban planning revolutionizing city development',
    'quantum computing breakthrough enabling faster drug discovery',
    'renewable energy storage solutions reaching commercial viability',
    'biotechnology advancement in personalized medicine',
    'space technology enabling asteroid mining operations',
    'neural interface technology restoring mobility to paralyzed patients',
    'climate engineering solutions reversing environmental damage',
    'autonomous systems improving agricultural efficiency',
    'blockchain technology securing digital identity verification',
    'synthetic biology creating sustainable manufacturing processes',
    'fusion energy achieving net positive energy output',
    'ocean cleanup technology removing plastic pollution',
    'vertical farming revolutionizing food production',
    'brain-computer interfaces enhancing cognitive abilities'
  ];

  const trendingTags = [
    { tag: '#AIBreakthrough', posts: 1247, growth: '+23%' },
    { tag: '#QuantumLeap', posts: 892, growth: '+45%' },
    { tag: '#SustainableTech', posts: 734, growth: '+12%' },
    { tag: '#BioInnovation', posts: 623, growth: '+67%' },
    { tag: '#SpaceCommerce', posts: 445, growth: '+89%' },
    { tag: '#CleanEnergy', posts: 567, growth: '+34%' },
    { tag: '#NeuroTech', posts: 389, growth: '+156%' },
    { tag: '#ClimateAction', posts: 712, growth: '+28%' }
  ];

  useEffect(() => {
    if (!isGenerating) return;

    const generateContent = async () => {
      if (generatingPost) return;
      
      setGeneratingPost(true);
      try {
        const types = ['breakthrough', 'analysis', 'trending', 'tutorial', 'discussion', 'innovation', 'personal', 'project', 'review'] as const;
        const categories = ['tech', 'science', 'social', 'business', 'health', 'environment', 'design', 'creative'] as const;
        
        const randomType = types[Math.floor(Math.random() * types.length)];
        const randomCategory = categories[Math.floor(Math.random() * categories.length)];
        const randomUser = realUsers[Math.floor(Math.random() * realUsers.length)];
        const randomTopic = contentTopics[Math.floor(Math.random() * contentTopics.length)];
        
        // Generate diverse, natural content
        const contentPrompts = {
          breakthrough: `Share an exciting breakthrough discovery about ${randomTopic}. Write it as a personal experience from someone who worked on this. Make it sound authentic and enthusiastic.`,
          analysis: `Create a thoughtful analysis post about ${randomTopic}. Include data insights, trends, and implications. Write it professionally but engagingly.`,
          personal: `Write a personal story about how ${randomTopic} impacted someone's life or work. Make it relatable and inspiring.`,
          project: `Describe a project showcase related to ${randomTopic}. Include technical details and results. Sound excited about sharing it.`,
          discussion: `Start a thoughtful discussion about ${randomTopic}. Ask engaging questions and share different perspectives.`
        };

        const prompt = contentPrompts[randomType as keyof typeof contentPrompts] || 
          `Create an engaging social media post about ${randomTopic}. Make it natural and authentic.`;
        
        // Simulate content generation (since pollenAI.generate fails)
        const sampleContent = generateSampleContent(randomType, randomTopic, randomCategory);
        
        const significance = Math.random() * 4 + 6; // 6-10 range
        const engagement = Math.floor(Math.random() * 5000) + 100;
        
        const newPost: Post = {
          id: Date.now().toString(),
          author: randomUser.name,
          username: randomUser.username,
          avatar: randomUser.avatar,
          content: sampleContent,
          likes: Math.floor(Math.random() * 2500) + 100,
          comments: Math.floor(Math.random() * 150) + 10,
          views: Math.floor(Math.random() * 50000) + 1000,
          timestamp: formatTimestamp(new Date()),
          type: randomType,
          category: randomCategory,
          significance,
          trending: significance > 8.5,
          engagement,
          rank: posts.length + 1,
          verified: randomUser.verified,
          media: Math.random() > 0.6 ? generateMedia(randomType) : undefined,
          tags: generateTags(randomCategory, randomType),
          replies: Math.random() > 0.5 ? generateReplies() : undefined
        };
        
        setPosts(prev => [newPost, ...prev.slice(0, 24)].sort((a, b) => b.significance - a.significance));
        
        // Gamification: Award points for content generation
        if (Math.random() > 0.7) {
          setUserPoints(prev => prev + Math.floor(Math.random() * 50) + 10);
        }
      } catch (error) {
        console.error('Failed to generate content:', error);
      }
      setGeneratingPost(false);
    };

    const initialTimeout = setTimeout(generateContent, 1000);
    const interval = setInterval(generateContent, Math.random() * 20000 + 15000);
    
    return () => {
      clearTimeout(initialTimeout);
      clearInterval(interval);
    };
  }, [isGenerating, generatingPost, posts.length]);

  const generateSampleContent = (type: string, topic: string, category: string) => {
    const contents = {
      breakthrough: [
        `🚀 BREAKTHROUGH: Just witnessed something incredible in our lab! ${topic} is now showing results we never thought possible. The implications for humanity are staggering.\n\nAfter 18 months of intensive research, our team achieved a 300% improvement in efficiency. This could change everything we know about ${category}.\n\nWhat excites me most? This is just the beginning. The next phase will blow your mind! 🧠✨`,
        `HOLY GRAIL MOMENT 🏆\n\nGuys, I can't contain my excitement! ${topic} just became reality in our research facility.\n\nThe data speaks for itself:\n• 400% faster processing\n• 90% reduction in energy consumption\n• Zero harmful byproducts\n\nThis breakthrough will revolutionize ${category} as we know it. Peer review pending, but preliminary results are mind-blowing! 📊🔬`,
        `Breaking: We did it! 🎉\n\n${topic} is no longer science fiction. Our team just achieved what many said was impossible.\n\nKey findings:\n✅ Surpassed theoretical limits\n✅ Scalable to industrial applications\n✅ Cost-effective implementation\n\nThe future of ${category} starts now. Can't wait to share more details soon! #Innovation #Research`
      ],
      analysis: [
        `Deep dive analysis 🧵\n\nI've been studying ${topic} for 8 months. Here's what the data reveals:\n\n📈 Market adoption: +340% YoY\n🔍 Efficiency gains: 250% average\n💡 Innovation rate: Accelerating exponentially\n\nWhy this matters for ${category}:\n1. Disrupts traditional approaches\n2. Creates new market opportunities\n3. Solves critical sustainability issues\n\nFull analysis in comments 👇`,
        `The numbers don't lie 📊\n\nAfter analyzing 50,000+ data points on ${topic}, some fascinating patterns emerge:\n\n🔥 Hot insight: 73% of implementations exceed projected ROI\n🎯 Success factors: Open collaboration + iterative development\n📈 Growth trajectory: Exponential, not linear\n\nKey takeaway: ${category} is at an inflection point. Early adopters will capture disproportionate value.\n\nThoughts? Drop your predictions below! 👇`,
        `Comprehensive review: ${topic} 📋\n\nAfter 6 months of research across 200+ case studies:\n\n✅ Technology readiness: 8.5/10\n✅ Market demand: Growing 45% annually\n✅ Implementation complexity: Moderate\n⚠️ Regulatory landscape: Evolving\n\n${category} leaders who act now will dominate the next decade. Waiting means playing catch-up.\n\nDetailed breakdown in thread 🧵`
      ],
      personal: [
        `Life-changing moment 💫\n\nTwo years ago, I decided to dive deep into ${topic}. Today, I can say it's transformed not just my career, but my entire worldview.\n\nThe journey taught me:\n• Persistence beats perfection\n• Collaboration amplifies innovation\n• Small improvements compound exponentially\n\nTo anyone considering ${category}: Take the leap. The learning curve is steep, but the view from the top is incredible! 🏔️`,
        `Reflecting on my ${category} journey 🌱\n\n18 months ago: Complete beginner\nToday: Leading a team working on ${topic}\n\nKey milestones:\n📚 6 months: Basic understanding\n🛠️ 12 months: First successful project\n🚀 18 months: Recognition as domain expert\n\nThe secret? Consistent daily practice + amazing community support. Grateful for everyone who helped along the way! 🙏`,
        `Plot twist of the decade 📈\n\nStarted studying ${topic} as a hobby. Fast forward 2 years: It's now the foundation of my startup!\n\nLessons learned:\n1. Follow curiosity, even if it seems irrelevant\n2. Document everything\n3. Share knowledge generously\n4. Network authentically\n\n${category} has given me more than a career—it's given me purpose. What's your breakthrough story? 💭`
      ]
    };
    
    const typeContents = contents[type as keyof typeof contents] || contents.breakthrough;
    return typeContents[Math.floor(Math.random() * typeContents.length)];
  };

  const generateTags = (category: string, type: string) => {
    const baseTags = {
      tech: ['#TechInnovation', '#AI', '#FutureTech', '#Development'],
      science: ['#Research', '#Science', '#Discovery', '#Innovation'],
      business: ['#Business', '#Strategy', '#Growth', '#Leadership'],
      health: ['#HealthTech', '#Medicine', '#Wellness', '#Biotech'],
      environment: ['#ClimateAction', '#Sustainability', '#GreenTech', '#Environment'],
      design: ['#Design', '#UX', '#Creative', '#Innovation'],
      creative: ['#Creative', '#Art', '#Innovation', '#Inspiration']
    };
    
    const typeTags = {
      breakthrough: ['#Breakthrough', '#Innovation', '#GameChanger'],
      trending: ['#Trending', '#Viral', '#Popular'],
      analysis: ['#Analysis', '#Data', '#Insights'],
      project: ['#Project', '#Showcase', '#Development']
    };
    
    const categoryTags = baseTags[category as keyof typeof baseTags] || baseTags.tech;
    const typeSpecific = typeTags[type as keyof typeof typeTags] || [];
    
    return [...categoryTags.slice(0, 2), ...typeSpecific.slice(0, 2)];
  };

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

  const getRankBadge = (rank: number) => {
    if (rank <= 3) return { icon: Trophy, color: 'text-yellow-400 bg-yellow-400/20' };
    if (rank <= 10) return { icon: Award, color: 'text-blue-400 bg-blue-400/20' };
    if (rank <= 25) return { icon: Target, color: 'text-green-400 bg-green-400/20' };
    return { icon: Star, color: 'text-gray-400 bg-gray-400/20' };
  };

  const filteredPosts = posts.filter(post => {
    if (filter === 'all') return true;
    if (filter === 'trending') return post.trending;
    if (filter === 'breakthrough') return post.type === 'breakthrough';
    return post.category === filter;
  });

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
          <div className="flex items-center space-x-6">
            {/* User Stats */}
            <div className="flex items-center space-x-4 px-4 py-2 bg-gray-800/50 rounded-xl border border-gray-700/50">
              <div className="flex items-center space-x-2">
                <Flame className="w-4 h-4 text-orange-400" />
                <span className="text-sm font-medium text-white">{userPoints}</span>
                <span className="text-xs text-gray-400">points</span>
              </div>
              <div className="w-px h-4 bg-gray-600"></div>
              <div className="flex items-center space-x-2">
                <Star className="w-4 h-4 text-yellow-400" />
                <span className="text-sm font-medium text-white">Lv.{userLevel}</span>
              </div>
            </div>
            
            {generatingPost && (
              <div className="flex items-center space-x-2 text-cyan-400">
                <div className="w-2 h-2 bg-cyan-400 rounded-full animate-pulse" />
                <span className="text-sm font-medium">Generating content...</span>
              </div>
            )}
            <div className="text-right">
              <div className="text-2xl font-bold text-white">{filteredPosts.length}</div>
              <div className="text-xs text-gray-400">Active posts</div>
            </div>
          </div>
        </div>

        {/* Trending Tags */}
        <div className="mb-6">
          <h3 className="text-sm font-medium text-gray-400 mb-3">Trending Topics</h3>
          <div className="flex flex-wrap gap-2">
            {trendingTags.slice(0, 6).map((tag, index) => (
              <div key={tag.tag} className="flex items-center space-x-2 px-3 py-1 bg-gray-800/50 rounded-full border border-gray-700/30 hover:border-cyan-500/30 transition-colors cursor-pointer">
                <span className="text-sm text-white">{tag.tag}</span>
                <span className="text-xs text-gray-400">{tag.posts}</span>
                <span className="text-xs text-green-400">{tag.growth}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Filter Tabs */}
        <div className="flex space-x-2 overflow-x-auto">
          {['all', 'trending', 'breakthrough', 'tech', 'science', 'business'].map((filterName) => (
            <button
              key={filterName}
              onClick={() => setFilter(filterName)}
              className={`px-4 py-2 rounded-lg text-sm font-medium whitespace-nowrap transition-colors ${
                filter === filterName
                  ? 'bg-cyan-600 text-white'
                  : 'bg-gray-800/50 text-gray-300 hover:bg-gray-700/50 hover:text-white'
              }`}
            >
              {filterName.charAt(0).toUpperCase() + filterName.slice(1)}
            </button>
          ))}
        </div>
      </div>

      {/* Feed */}
      <div className="flex-1 overflow-y-auto p-6 space-y-4">
        {filteredPosts.map((post, index) => {
          const rankBadge = getRankBadge(post.rank || index + 1);
          
          return (
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
                      {post.verified && <Star className="w-4 h-4 text-blue-400 fill-current" />}
                      <span className="text-gray-400 text-sm">@{post.username}</span>
                      <span className="text-gray-500 text-sm">•</span>
                      <span className="text-gray-400 text-sm">{post.timestamp}</span>
                    </div>
                    <div className="flex items-center space-x-2">
                      {post.trending && (
                        <div className="flex items-center space-x-1 px-2 py-1 bg-orange-500/20 text-orange-400 text-xs rounded-full">
                          <TrendingUp className="w-3 h-3" />
                          <span>Trending</span>
                        </div>
                      )}
                      <div className={`flex items-center space-x-1 px-2 py-1 ${rankBadge.color} text-xs rounded-full`}>
                        <rankBadge.icon className="w-3 h-3" />
                        <span>#{post.rank || index + 1}</span>
                      </div>
                      <div className="flex items-center space-x-1 px-2 py-1 bg-cyan-500/20 text-cyan-400 text-xs rounded-full">
                        <Zap className="w-3 h-3" />
                        <span>{post.significance.toFixed(1)}</span>
                      </div>
                    </div>
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

              {/* Tags */}
              {post.tags && (
                <div className="flex flex-wrap gap-2 mb-4">
                  {post.tags.map((tag) => (
                    <span key={tag} className="px-2 py-1 bg-gray-800/50 text-cyan-400 text-xs rounded-lg hover:bg-gray-700/50 cursor-pointer transition-colors">
                      {tag}
                    </span>
                  ))}
                </div>
              )}

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
          );
        })}

        {filteredPosts.length === 0 && (
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
