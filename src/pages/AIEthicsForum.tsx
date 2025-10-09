import { useState, useEffect } from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Badge } from '@/components/ui/badge';
import { Avatar, AvatarFallback } from '@/components/ui/avatar';
import { useToast } from '@/hooks/use-toast';
import { 
  Shield, Users, TrendingUp, MessageSquare, ThumbsUp, Award, 
  AlertTriangle, Eye, Flag, CheckCircle, ArrowUp, Pin 
} from 'lucide-react';
import { EthicsReportForm } from '@/components/ethics/EthicsReportForm';
import { TransparencyDashboard } from '@/components/ethics/TransparencyDashboard';

interface ForumTopic {
  id: string;
  category: string;
  title: string;
  description: string;
  author: string;
  authorType: 'expert' | 'researcher' | 'user';
  votes: number;
  replies: number;
  views: number;
  isPinned: boolean;
  createdAt: string;
}

interface ForumPost {
  id: string;
  topicId: string;
  content: string;
  author: string;
  authorType: 'expert' | 'researcher' | 'user';
  votes: number;
  isAccepted: boolean;
  createdAt: string;
}

export default function AIEthicsForum() {
  const [activeTab, setActiveTab] = useState('forum');
  const [topics, setTopics] = useState<ForumTopic[]>([]);
  const [selectedCategory, setSelectedCategory] = useState<string>('all');
  const [selectedTopic, setSelectedTopic] = useState<ForumTopic | null>(null);
  const [posts, setPosts] = useState<ForumPost[]>([]);
  const [newTopicTitle, setNewTopicTitle] = useState('');
  const [newTopicDescription, setNewTopicDescription] = useState('');
  const [newTopicCategory, setNewTopicCategory] = useState('');
  const [newPostContent, setNewPostContent] = useState('');
  const { toast } = useToast();

  const categories = [
    { value: 'all', label: 'All Topics', icon: MessageSquare },
    { value: 'ai_bias', label: 'AI Bias', icon: AlertTriangle },
    { value: 'privacy', label: 'Privacy', icon: Shield },
    { value: 'transparency', label: 'Transparency', icon: Eye },
    { value: 'fairness', label: 'Fairness', icon: CheckCircle },
    { value: 'accountability', label: 'Accountability', icon: Users },
    { value: 'safety', label: 'Safety', icon: Flag },
  ];

  const mockTopics: ForumTopic[] = [
    {
      id: '1',
      category: 'ai_bias',
      title: 'How to detect and mitigate gender bias in AI systems?',
      description: 'Discussion on best practices for identifying and addressing gender bias in machine learning models',
      author: 'Dr. Sarah Chen',
      authorType: 'expert',
      votes: 142,
      replies: 28,
      views: 1540,
      isPinned: true,
      createdAt: '2025-10-08T10:00:00Z'
    },
    {
      id: '2',
      category: 'privacy',
      title: 'Balancing AI capabilities with user data privacy',
      description: 'Exploring frameworks for maintaining user privacy while enabling AI functionality',
      author: 'Prof. Michael Rodriguez',
      authorType: 'researcher',
      votes: 98,
      replies: 15,
      views: 892,
      isPinned: false,
      createdAt: '2025-10-07T14:30:00Z'
    },
    {
      id: '3',
      category: 'transparency',
      title: 'Should AI decision-making processes be fully transparent?',
      description: 'Debating the level of transparency needed in AI systems and its implications',
      author: 'Alex Johnson',
      authorType: 'user',
      votes: 67,
      replies: 42,
      views: 1203,
      isPinned: false,
      createdAt: '2025-10-06T09:15:00Z'
    },
  ];

  const mockPosts: ForumPost[] = [
    {
      id: 'p1',
      topicId: '1',
      content: 'One effective approach is to use diverse training datasets and regularly audit model outputs for bias patterns. We should also implement bias detection algorithms that can flag potential issues before deployment.',
      author: 'Dr. Sarah Chen',
      authorType: 'expert',
      votes: 45,
      isAccepted: true,
      createdAt: '2025-10-08T10:30:00Z'
    },
    {
      id: 'p2',
      topicId: '1',
      content: 'I agree, but we also need to consider intersectional bias. Gender bias often intersects with racial, age, and cultural biases. Our detection methods should be sophisticated enough to catch these complex patterns.',
      author: 'Prof. Aisha Williams',
      authorType: 'researcher',
      votes: 38,
      isAccepted: false,
      createdAt: '2025-10-08T11:00:00Z'
    },
  ];

  useEffect(() => {
    setTopics(mockTopics);
  }, []);

  const handleVote = (topicId: string) => {
    setTopics(topics.map(t => 
      t.id === topicId ? { ...t, votes: t.votes + 1 } : t
    ));
    toast({
      title: "Vote recorded",
      description: "Thank you for your contribution!",
    });
  };

  const handleCreateTopic = () => {
    if (!newTopicTitle || !newTopicDescription || !newTopicCategory) {
      toast({
        title: "Missing information",
        description: "Please fill in all fields",
        variant: "destructive"
      });
      return;
    }

    const newTopic: ForumTopic = {
      id: `topic_${Date.now()}`,
      category: newTopicCategory,
      title: newTopicTitle,
      description: newTopicDescription,
      author: 'Anonymous User',
      authorType: 'user',
      votes: 0,
      replies: 0,
      views: 0,
      isPinned: false,
      createdAt: new Date().toISOString()
    };

    setTopics([newTopic, ...topics]);
    setNewTopicTitle('');
    setNewTopicDescription('');
    setNewTopicCategory('');

    toast({
      title: "Topic created",
      description: "Your topic has been published to the forum",
    });
  };

  const handlePostReply = (topicId: string) => {
    if (!newPostContent) {
      toast({
        title: "Empty reply",
        description: "Please enter your response",
        variant: "destructive"
      });
      return;
    }

    const newPost: ForumPost = {
      id: `post_${Date.now()}`,
      topicId,
      content: newPostContent,
      author: 'Anonymous User',
      authorType: 'user',
      votes: 0,
      isAccepted: false,
      createdAt: new Date().toISOString()
    };

    setPosts([...posts, newPost]);
    setNewPostContent('');

    toast({
      title: "Reply posted",
      description: "Your response has been added to the discussion",
    });
  };

  const getAuthorBadge = (authorType: string) => {
    switch (authorType) {
      case 'expert':
        return <Badge variant="default" className="ml-2 bg-purple-600"><Award className="w-3 h-3 mr-1" />Expert</Badge>;
      case 'researcher':
        return <Badge variant="secondary" className="ml-2"><Eye className="w-3 h-3 mr-1" />Researcher</Badge>;
      default:
        return null;
    }
  };

  const filteredTopics = selectedCategory === 'all' 
    ? topics 
    : topics.filter(t => t.category === selectedCategory);

  return (
    <div className="container mx-auto p-6 max-w-7xl">
      <div className="mb-8">
        <h1 className="text-4xl font-bold mb-2">AI Ethics & Responsible Innovation Forum</h1>
        <p className="text-muted-foreground">
          Join the conversation on ethical AI development and responsible innovation
        </p>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="forum">
            <MessageSquare className="w-4 h-4 mr-2" />
            Forum
          </TabsTrigger>
          <TabsTrigger value="guidelines">
            <CheckCircle className="w-4 h-4 mr-2" />
            Guidelines
          </TabsTrigger>
          <TabsTrigger value="transparency">
            <Eye className="w-4 h-4 mr-2" />
            Transparency
          </TabsTrigger>
          <TabsTrigger value="report">
            <AlertTriangle className="w-4 h-4 mr-2" />
            Report Issue
          </TabsTrigger>
        </TabsList>

        <TabsContent value="forum" className="mt-6">
          <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
            <div className="lg:col-span-1">
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg">Categories</CardTitle>
                </CardHeader>
                <CardContent className="space-y-2">
                  {categories.map((cat) => {
                    const Icon = cat.icon;
                    return (
                      <Button
                        key={cat.value}
                        variant={selectedCategory === cat.value ? "default" : "ghost"}
                        className="w-full justify-start"
                        onClick={() => setSelectedCategory(cat.value)}
                      >
                        <Icon className="w-4 h-4 mr-2" />
                        {cat.label}
                      </Button>
                    );
                  })}
                </CardContent>
              </Card>

              <Card className="mt-4">
                <CardHeader>
                  <CardTitle className="text-lg">Create Topic</CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div>
                    <Label>Category</Label>
                    <Select value={newTopicCategory} onValueChange={setNewTopicCategory}>
                      <SelectTrigger>
                        <SelectValue placeholder="Select" />
                      </SelectTrigger>
                      <SelectContent>
                        {categories.slice(1).map((cat) => (
                          <SelectItem key={cat.value} value={cat.value}>
                            {cat.label}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                  <div>
                    <Label>Title</Label>
                    <Input
                      placeholder="Topic title"
                      value={newTopicTitle}
                      onChange={(e) => setNewTopicTitle(e.target.value)}
                    />
                  </div>
                  <div>
                    <Label>Description</Label>
                    <Textarea
                      placeholder="Describe the topic..."
                      value={newTopicDescription}
                      onChange={(e) => setNewTopicDescription(e.target.value)}
                      rows={3}
                    />
                  </div>
                  <Button onClick={handleCreateTopic} className="w-full">
                    Create Topic
                  </Button>
                </CardContent>
              </Card>
            </div>

            <div className="lg:col-span-3">
              {!selectedTopic ? (
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <h2 className="text-2xl font-bold">
                      {selectedCategory === 'all' ? 'All Topics' : categories.find(c => c.value === selectedCategory)?.label}
                    </h2>
                    <Badge variant="secondary">{filteredTopics.length} topics</Badge>
                  </div>

                  {filteredTopics.map((topic) => (
                    <Card key={topic.id} className="hover:shadow-md transition-shadow cursor-pointer" onClick={() => setSelectedTopic(topic)}>
                      <CardContent className="p-6">
                        <div className="flex items-start justify-between">
                          <div className="flex-1">
                            <div className="flex items-center gap-2 mb-2">
                              {topic.isPinned && <Pin className="w-4 h-4 text-primary" />}
                              <Badge variant="outline">{categories.find(c => c.value === topic.category)?.label}</Badge>
                            </div>
                            <h3 className="text-xl font-semibold mb-2">{topic.title}</h3>
                            <p className="text-muted-foreground mb-4">{topic.description}</p>
                            <div className="flex items-center gap-6 text-sm text-muted-foreground">
                              <div className="flex items-center">
                                <Avatar className="w-6 h-6 mr-2">
                                  <AvatarFallback>{topic.author[0]}</AvatarFallback>
                                </Avatar>
                                <span>{topic.author}</span>
                                {getAuthorBadge(topic.authorType)}
                              </div>
                              <span className="flex items-center">
                                <MessageSquare className="w-4 h-4 mr-1" />
                                {topic.replies}
                              </span>
                              <span className="flex items-center">
                                <Eye className="w-4 h-4 mr-1" />
                                {topic.views}
                              </span>
                            </div>
                          </div>
                          <div className="flex flex-col items-center ml-4">
                            <Button variant="ghost" size="sm" onClick={(e) => { e.stopPropagation(); handleVote(topic.id); }}>
                              <ArrowUp className="w-4 h-4" />
                            </Button>
                            <span className="font-semibold">{topic.votes}</span>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              ) : (
                <div>
                  <Button variant="ghost" onClick={() => setSelectedTopic(null)} className="mb-4">
                    ← Back to topics
                  </Button>

                  <Card>
                    <CardHeader>
                      <div className="flex items-center gap-2 mb-2">
                        <Badge variant="outline">{categories.find(c => c.value === selectedTopic.category)?.label}</Badge>
                      </div>
                      <CardTitle className="text-2xl">{selectedTopic.title}</CardTitle>
                      <CardDescription>{selectedTopic.description}</CardDescription>
                      <div className="flex items-center gap-4 mt-4 text-sm text-muted-foreground">
                        <div className="flex items-center">
                          <Avatar className="w-6 h-6 mr-2">
                            <AvatarFallback>{selectedTopic.author[0]}</AvatarFallback>
                          </Avatar>
                          <span>{selectedTopic.author}</span>
                          {getAuthorBadge(selectedTopic.authorType)}
                        </div>
                        <span>{new Date(selectedTopic.createdAt).toLocaleDateString()}</span>
                      </div>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-6">
                        <div className="border-t pt-6">
                          <h3 className="text-lg font-semibold mb-4">Responses ({mockPosts.filter(p => p.topicId === selectedTopic.id).length})</h3>
                          
                          {mockPosts.filter(p => p.topicId === selectedTopic.id).map((post) => (
                            <Card key={post.id} className="mb-4">
                              <CardContent className="p-4">
                                <div className="flex justify-between items-start">
                                  <div className="flex-1">
                                    <div className="flex items-center gap-2 mb-3">
                                      <Avatar className="w-8 h-8">
                                        <AvatarFallback>{post.author[0]}</AvatarFallback>
                                      </Avatar>
                                      <div>
                                        <div className="flex items-center">
                                          <span className="font-medium">{post.author}</span>
                                          {getAuthorBadge(post.authorType)}
                                          {post.isAccepted && (
                                            <Badge variant="default" className="ml-2 bg-green-600">
                                              <CheckCircle className="w-3 h-3 mr-1" />
                                              Accepted Answer
                                            </Badge>
                                          )}
                                        </div>
                                        <span className="text-xs text-muted-foreground">
                                          {new Date(post.createdAt).toLocaleDateString()}
                                        </span>
                                      </div>
                                    </div>
                                    <p className="text-sm">{post.content}</p>
                                  </div>
                                  <div className="flex flex-col items-center ml-4">
                                    <Button variant="ghost" size="sm">
                                      <ThumbsUp className="w-4 h-4" />
                                    </Button>
                                    <span className="text-sm">{post.votes}</span>
                                  </div>
                                </div>
                              </CardContent>
                            </Card>
                          ))}
                        </div>

                        <div className="border-t pt-6">
                          <h3 className="text-lg font-semibold mb-4">Add Your Response</h3>
                          <Textarea
                            placeholder="Share your thoughts..."
                            value={newPostContent}
                            onChange={(e) => setNewPostContent(e.target.value)}
                            rows={4}
                            className="mb-3"
                          />
                          <Button onClick={() => handlePostReply(selectedTopic.id)}>
                            Post Reply
                          </Button>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                </div>
              )}
            </div>
          </div>
        </TabsContent>

        <TabsContent value="guidelines" className="mt-6">
          <Card>
            <CardHeader>
              <CardTitle>Ethical Guidelines</CardTitle>
              <CardDescription>Community-driven ethical principles for AI development</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-3">
                <div className="p-4 border rounded-lg">
                  <h3 className="font-semibold mb-2">1. Fairness & Non-Discrimination</h3>
                  <p className="text-sm text-muted-foreground">AI systems should treat all individuals and groups fairly, avoiding bias and discrimination based on protected characteristics.</p>
                </div>
                <div className="p-4 border rounded-lg">
                  <h3 className="font-semibold mb-2">2. Transparency & Explainability</h3>
                  <p className="text-sm text-muted-foreground">AI decision-making processes should be transparent and explainable to affected parties.</p>
                </div>
                <div className="p-4 border rounded-lg">
                  <h3 className="font-semibold mb-2">3. Privacy & Data Protection</h3>
                  <p className="text-sm text-muted-foreground">User data must be protected with strong privacy safeguards and used only for stated purposes.</p>
                </div>
                <div className="p-4 border rounded-lg">
                  <h3 className="font-semibold mb-2">4. Accountability & Responsibility</h3>
                  <p className="text-sm text-muted-foreground">Clear lines of accountability must exist for AI system outcomes and decisions.</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="transparency" className="mt-6">
          <TransparencyDashboard />
        </TabsContent>

        <TabsContent value="report" className="mt-6">
          <EthicsReportForm userId="anonymous_user" />
        </TabsContent>
      </Tabs>
    </div>
  );
}
