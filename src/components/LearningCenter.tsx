import React, { useState, useEffect } from 'react';
import { 
  GraduationCap, BookOpen, Video, FileText, Search, 
  Clock, Star, TrendingUp, Globe, Play, Bookmark,
  Award, Users, Target, Lightbulb
} from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

interface LearningContent {
  id: string;
  title: string;
  description: string;
  type: 'course' | 'article' | 'video' | 'tutorial' | 'documentation';
  category: string;
  difficulty: 'beginner' | 'intermediate' | 'advanced';
  duration: string;
  rating: number;
  students: number;
  thumbnail: string;
  tags: string[];
  trending: boolean;
  featured: boolean;
}

const categories = [
  { id: 'technology', name: 'Technology', icon: Target, color: 'bg-blue-500/20 text-blue-300' },
  { id: 'science', name: 'Science', icon: Lightbulb, color: 'bg-purple-500/20 text-purple-300' },
  { id: 'programming', name: 'Programming', icon: FileText, color: 'bg-green-500/20 text-green-300' },
  { id: 'business', name: 'Business', icon: TrendingUp, color: 'bg-orange-500/20 text-orange-300' },
  { id: 'ai-ml', name: 'AI & Machine Learning', icon: Globe, color: 'bg-cyan-500/20 text-cyan-300' },
  { id: 'design', name: 'Design', icon: Star, color: 'bg-pink-500/20 text-pink-300' }
];

const generateLearningContent = (): LearningContent[] => {
  const titles = [
    'Introduction to Quantum Computing Fundamentals',
    'Advanced Machine Learning Algorithms',
    'Building Scalable Web Applications',
    'Design Thinking for Innovation',
    'Blockchain Technology and Cryptocurrencies',
    'Neural Networks and Deep Learning',
    'Cloud Architecture Best Practices',
    'Data Science with Python',
    'UI/UX Design Principles',
    'Cybersecurity Fundamentals',
    'Artificial Intelligence Ethics',
    'Modern JavaScript Frameworks',
    'Digital Marketing Strategies',
    'Product Management Essentials',
    'DevOps and Continuous Integration',
    'Mobile App Development',
    'Database Design and Optimization',
    'Virtual Reality Development',
    'Sustainable Technology Solutions',
    'Entrepreneurship and Startups'
  ];

  const descriptions = [
    'Comprehensive introduction to quantum computing principles, quantum gates, and practical applications in modern technology.',
    'Deep dive into advanced ML algorithms including neural networks, decision trees, and ensemble methods.',
    'Learn to build scalable, maintainable web applications using modern frameworks and best practices.',
    'Master the design thinking process to solve complex problems and drive innovation in your organization.',
    'Complete guide to blockchain technology, smart contracts, and cryptocurrency fundamentals.',
    'Understanding neural network architectures and implementing deep learning solutions.',
    'Best practices for designing and implementing cloud-native applications and services.',
    'Practical data science techniques using Python, pandas, and popular ML libraries.',
    'Core principles of user interface and user experience design for digital products.',
    'Essential cybersecurity concepts, threat detection, and protection strategies.',
    'Exploring ethical considerations in AI development and responsible technology practices.',
    'Mastering React, Vue, and Angular for modern web development.',
    'Digital marketing strategies for the modern era including SEO, content marketing, and social media.',
    'Product management methodologies, user research, and go-to-market strategies.',
    'DevOps practices, CI/CD pipelines, and infrastructure automation.',
    'Cross-platform mobile development using React Native and Flutter.',
    'Database design principles, optimization techniques, and performance tuning.',
    'Introduction to VR development using Unity and modern VR frameworks.',
    'Sustainable technology practices and green computing initiatives.',
    'Entrepreneurship fundamentals, startup methodologies, and business model development.'
  ];

  return titles.map((title, index) => ({
    id: `learn-${index + 1}`,
    title,
    description: descriptions[index],
    type: ['course', 'article', 'video', 'tutorial', 'documentation'][Math.floor(Math.random() * 5)] as any,
    category: categories[Math.floor(Math.random() * categories.length)].id,
    difficulty: ['beginner', 'intermediate', 'advanced'][Math.floor(Math.random() * 3)] as any,
    duration: `${Math.floor(Math.random() * 8 + 1)}h ${Math.floor(Math.random() * 60)}m`,
    rating: Number((Math.random() * 1.5 + 3.5).toFixed(1)),
    students: Math.floor(Math.random() * 50000 + 1000),
    thumbnail: `bg-gradient-to-br from-${['blue', 'purple', 'green', 'orange', 'cyan', 'pink'][Math.floor(Math.random() * 6)]}-500 to-${['blue', 'purple', 'green', 'orange', 'cyan', 'pink'][Math.floor(Math.random() * 6)]}-700`,
    tags: ['AI', 'Tech', 'Innovation', 'Future', 'Learning'].slice(0, Math.floor(Math.random() * 3 + 2)),
    trending: Math.random() > 0.7,
    featured: Math.random() > 0.8
  }));
};

export function LearningCenter() {
  const [content, setContent] = useState<LearningContent[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [selectedDifficulty, setSelectedDifficulty] = useState('all');
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // Simulate loading content
    const timer = setTimeout(() => {
      setContent(generateLearningContent());
      setIsLoading(false);
    }, 1000);

    return () => clearTimeout(timer);
  }, []);

  // Load more content periodically
  useEffect(() => {
    const interval = setInterval(() => {
      const newContent = generateLearningContent().slice(0, 5);
      setContent(prev => [...newContent, ...prev].slice(0, 50)); // Keep max 50 items
    }, 30000); // Add new content every 30 seconds

    return () => clearInterval(interval);
  }, []);

  const filteredContent = content.filter(item => {
    const matchesSearch = !searchQuery || 
      item.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
      item.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
      item.tags.some(tag => tag.toLowerCase().includes(searchQuery.toLowerCase()));
    
    const matchesCategory = selectedCategory === 'all' || item.category === selectedCategory;
    const matchesDifficulty = selectedDifficulty === 'all' || item.difficulty === selectedDifficulty;
    
    return matchesSearch && matchesCategory && matchesDifficulty;
  });

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'course': return GraduationCap;
      case 'video': return Video;
      case 'article': return FileText;
      case 'tutorial': return BookOpen;
      default: return FileText;
    }
  };

  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty) {
      case 'beginner': return 'bg-green-500/20 text-green-300 border-green-500/30';
      case 'intermediate': return 'bg-yellow-500/20 text-yellow-300 border-yellow-500/30';
      case 'advanced': return 'bg-red-500/20 text-red-300 border-red-500/30';
      default: return 'bg-gray-500/20 text-gray-300 border-gray-500/30';
    }
  };

  return (
    <div className="flex-1 bg-gray-950 min-h-screen">
      {/* Header */}
      <div className="bg-gray-900/95 backdrop-blur-sm border-b border-gray-800/50 p-6">
        <div className="max-w-7xl mx-auto">
          <h1 className="text-3xl font-bold text-white mb-2 flex items-center gap-3">
            <GraduationCap className="w-8 h-8 text-blue-400" />
            Learning Center
          </h1>
          <p className="text-gray-400">Continuous learning hub • Educational content • Skill development</p>
        </div>
      </div>

      <div className="max-w-7xl mx-auto p-6">
        {/* Search and Filters */}
        <div className="mb-8 space-y-4">
          <div className="relative max-w-2xl">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
            <Input
              type="text"
              placeholder="Search courses, tutorials, articles..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-10 bg-gray-800/50 border-gray-700 text-white placeholder-gray-400"
            />
          </div>

          {/* Filters */}
          <div className="flex flex-wrap gap-4">
            <div className="flex items-center gap-2">
              <span className="text-sm text-gray-400">Category:</span>
              <select
                value={selectedCategory}
                onChange={(e) => setSelectedCategory(e.target.value)}
                className="bg-gray-800 border border-gray-700 text-white rounded px-3 py-1 text-sm"
              >
                <option value="all">All Categories</option>
                {categories.map(cat => (
                  <option key={cat.id} value={cat.id}>{cat.name}</option>
                ))}
              </select>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-sm text-gray-400">Difficulty:</span>
              <select
                value={selectedDifficulty}
                onChange={(e) => setSelectedDifficulty(e.target.value)}
                className="bg-gray-800 border border-gray-700 text-white rounded px-3 py-1 text-sm"
              >
                <option value="all">All Levels</option>
                <option value="beginner">Beginner</option>
                <option value="intermediate">Intermediate</option>
                <option value="advanced">Advanced</option>
              </select>
            </div>
          </div>
        </div>

        {/* Featured Content */}
        <div className="mb-8">
          <h2 className="text-2xl font-bold text-white mb-4 flex items-center gap-2">
            <Star className="w-6 h-6 text-yellow-400" />
            Featured Learning
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {content.filter(item => item.featured).slice(0, 3).map(item => (
              <Card key={item.id} className="bg-gray-900/50 border-gray-800/50 hover:bg-gray-900/70 transition-colors">
                <CardHeader className="pb-3">
                  <div className={`${item.thumbnail} w-full h-32 rounded-lg mb-3 flex items-center justify-center relative`}>
                    <Play className="w-8 h-8 text-white" />
                    {item.trending && (
                      <Badge className="absolute top-2 right-2 bg-red-500 hover:bg-red-500 text-xs">
                        Trending
                      </Badge>
                    )}
                  </div>
                  <CardTitle className="text-white text-lg">{item.title}</CardTitle>
                  <CardDescription className="text-gray-400 text-sm line-clamp-2">
                    {item.description}
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        {React.createElement(getTypeIcon(item.type), { className: "w-4 h-4 text-blue-400" })}
                        <span className="text-sm text-gray-300 capitalize">{item.type}</span>
                      </div>
                      <div className={`px-2 py-1 rounded text-xs border ${getDifficultyColor(item.difficulty)}`}>
                        {item.difficulty}
                      </div>
                    </div>
                    <div className="flex items-center justify-between text-sm text-gray-400">
                      <div className="flex items-center gap-1">
                        <Clock className="w-4 h-4" />
                        {item.duration}
                      </div>
                      <div className="flex items-center gap-1">
                        <Users className="w-4 h-4" />
                        {item.students.toLocaleString()}
                      </div>
                    </div>
                    <Button className="w-full bg-blue-600 hover:bg-blue-700">
                      Start Learning
                    </Button>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>

        {/* All Content */}
        <div>
          <h2 className="text-2xl font-bold text-white mb-4">All Learning Content</h2>
          {isLoading ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {[...Array(9)].map((_, i) => (
                <div key={i} className="bg-gray-900/50 rounded-lg p-6 border border-gray-800/50 animate-pulse">
                  <div className="w-full h-32 bg-gray-700 rounded-lg mb-4"></div>
                  <div className="space-y-2">
                    <div className="w-3/4 h-4 bg-gray-700 rounded"></div>
                    <div className="w-full h-3 bg-gray-700 rounded"></div>
                    <div className="w-1/2 h-3 bg-gray-700 rounded"></div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
              {filteredContent.map(item => (
                <Card key={item.id} className="bg-gray-900/50 border-gray-800/50 hover:bg-gray-900/70 transition-colors">
                  <CardHeader className="pb-3">
                    <div className={`${item.thumbnail} w-full h-24 rounded-lg mb-3 flex items-center justify-center relative`}>
                      {React.createElement(getTypeIcon(item.type), { className: "w-6 h-6 text-white" })}
                      {item.trending && (
                        <Badge className="absolute top-1 right-1 bg-red-500 hover:bg-red-500 text-xs">
                          Hot
                        </Badge>
                      )}
                    </div>
                    <CardTitle className="text-white text-base line-clamp-2">{item.title}</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      <div className="flex items-center justify-between">
                        <div className={`px-2 py-1 rounded text-xs border ${getDifficultyColor(item.difficulty)}`}>
                          {item.difficulty}
                        </div>
                        <div className="flex items-center gap-1 text-xs text-gray-400">
                          <Star className="w-3 h-3 text-yellow-400" />
                          {item.rating}
                        </div>
                      </div>
                      <div className="text-xs text-gray-400">
                        <Clock className="w-3 h-3 inline mr-1" />
                        {item.duration}
                      </div>
                      <Button size="sm" className="w-full bg-blue-600 hover:bg-blue-700">
                        Learn
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}