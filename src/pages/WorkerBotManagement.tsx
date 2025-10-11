import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Switch } from '@/components/ui/switch';
import { Textarea } from '@/components/ui/textarea';
import { Badge } from '@/components/ui/badge';
import { 
  Bot, Play, Pause, Square, Settings, RefreshCw, Plus, 
  CheckCircle, XCircle, Clock, TrendingUp, Zap 
} from 'lucide-react';
import { sseWorkerBot, WorkerTask, BotConfig } from '@/services/sseWorkerBot';
import { toast } from 'sonner';

export default function WorkerBotManagement() {
  const [config, setConfig] = useState<BotConfig>(sseWorkerBot.getConfig());
  const [tasks, setTasks] = useState<WorkerTask[]>(sseWorkerBot.getAllTasks());
  const [activeTab, setActiveTab] = useState('overview');
  
  // Manual task creation state
  const [ugcProduct, setUgcProduct] = useState('');
  const [ugcPlatform, setUgcPlatform] = useState<'instagram' | 'tiktok' | 'youtube' | 'facebook'>('instagram');
  const [ugcAudience, setUgcAudience] = useState('');
  
  const [suggestionTopic, setSuggestionTopic] = useState('');
  const [suggestionCount, setSuggestionCount] = useState(5);
  
  const [improvementContent, setImprovementContent] = useState('');
  const [improvementAreas, setImprovementAreas] = useState('');

  useEffect(() => {
    const unsubscribe = sseWorkerBot.subscribeToUpdates((task) => {
      setTasks(prev => {
        const index = prev.findIndex(t => t.id === task.id);
        if (index >= 0) {
          const updated = [...prev];
          updated[index] = task;
          return updated;
        }
        return [...prev, task];
      });
    });

    return unsubscribe;
  }, []);

  const handleStartBot = () => {
    sseWorkerBot.start();
    setConfig(sseWorkerBot.getConfig());
    toast.success('Worker Bot started successfully');
  };

  const handleStopBot = () => {
    sseWorkerBot.stop();
    setConfig(sseWorkerBot.getConfig());
    toast.success('Worker Bot stopped');
  };

  const handleSaveConfig = () => {
    sseWorkerBot.updateConfig(config);
    toast.success('Configuration saved successfully');
  };

  const handleCreateUGCAd = async () => {
    if (!ugcProduct || !ugcAudience) {
      toast.error('Please fill in all fields');
      return;
    }

    try {
      toast.info('Creating UGC ad...');
      const result = await sseWorkerBot.createUGCAd({
        product: ugcProduct,
        platform: ugcPlatform,
        targetAudience: ugcAudience
      });
      
      toast.success('UGC ad created successfully!');
      console.log('UGC Ad Result:', result);
      
      // Reset form
      setUgcProduct('');
      setUgcAudience('');
    } catch (error) {
      toast.error('Failed to create UGC ad');
    }
  };

  const handleGenerateContentSuggestions = async () => {
    if (!suggestionTopic) {
      toast.error('Please enter a topic');
      return;
    }

    try {
      toast.info('Generating content suggestions...');
      const results = await sseWorkerBot.generateContentSuggestions(
        suggestionTopic, 
        suggestionCount
      );
      
      toast.success(`Generated ${results.length} content suggestions!`);
      console.log('Content Suggestions:', results);
      
      setSuggestionTopic('');
    } catch (error) {
      toast.error('Failed to generate suggestions');
    }
  };

  const handleImproveContent = async () => {
    if (!improvementContent || !improvementAreas) {
      toast.error('Please provide content and improvement areas');
      return;
    }

    try {
      toast.info('Improving content...');
      const result = await sseWorkerBot.improveContent(
        improvementContent,
        improvementAreas.split(',').map(a => a.trim())
      );
      
      toast.success('Content improved successfully!');
      console.log('Improvement Result:', result);
      
      setImprovementContent('');
      setImprovementAreas('');
    } catch (error) {
      toast.error('Failed to improve content');
    }
  };

  const getStatusBadge = (status: WorkerTask['status']) => {
    const variants: Record<WorkerTask['status'], { variant: any; icon: any }> = {
      pending: { variant: 'secondary', icon: Clock },
      processing: { variant: 'default', icon: RefreshCw },
      completed: { variant: 'success', icon: CheckCircle },
      failed: { variant: 'destructive', icon: XCircle }
    };
    
    const { variant, icon: Icon } = variants[status];
    return (
      <Badge variant={variant as any} className="flex items-center gap-1">
        <Icon className="w-3 h-3" />
        {status}
      </Badge>
    );
  };

  const completedTasks = tasks.filter(t => t.status === 'completed').length;
  const failedTasks = tasks.filter(t => t.status === 'failed').length;
  const processingTasks = tasks.filter(t => t.status === 'processing').length;

  return (
    <div className="container mx-auto px-4 py-8 pt-24">
      <div className="mb-8">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-black dark:text-white mb-2 flex items-center gap-3">
              <Bot className="w-8 h-8 text-blue-600" />
              SSE Worker Bot Management
            </h1>
            <p className="text-gray-600 dark:text-gray-400">
              Configure and control the Pollen AI worker bot for automated content generation
            </p>
          </div>
          <div className="flex gap-2">
            {config.enabled ? (
              <Button onClick={handleStopBot} variant="destructive" className="flex items-center gap-2">
                <Square className="w-4 h-4" />
                Stop Bot
              </Button>
            ) : (
              <Button onClick={handleStartBot} className="flex items-center gap-2">
                <Play className="w-4 h-4" />
                Start Bot
              </Button>
            )}
          </div>
        </div>
      </div>

      {/* Status Overview */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium text-gray-600 dark:text-gray-400">Bot Status</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center gap-2">
              <div className={`w-3 h-3 rounded-full ${config.enabled ? 'bg-green-500 animate-pulse' : 'bg-gray-400'}`} />
              <p className="text-2xl font-bold">{config.enabled ? 'Active' : 'Inactive'}</p>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium text-gray-600 dark:text-gray-400">Tasks Completed</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center justify-between">
              <p className="text-2xl font-bold text-green-600">{completedTasks}</p>
              <CheckCircle className="w-5 h-5 text-green-600" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium text-gray-600 dark:text-gray-400">Processing</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center justify-between">
              <p className="text-2xl font-bold text-blue-600">{processingTasks}</p>
              <RefreshCw className="w-5 h-5 text-blue-600 animate-spin" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium text-gray-600 dark:text-gray-400">Failed</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center justify-between">
              <p className="text-2xl font-bold text-red-600">{failedTasks}</p>
              <XCircle className="w-5 h-5 text-red-600" />
            </div>
          </CardContent>
        </Card>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="tasks">Tasks</TabsTrigger>
          <TabsTrigger value="create">Create Task</TabsTrigger>
          <TabsTrigger value="settings">Settings</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-4 mt-6">
          <Card>
            <CardHeader>
              <CardTitle>Active Configuration</CardTitle>
              <CardDescription>Current worker bot settings and status</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <p className="text-sm text-gray-600 dark:text-gray-400">Auto Generation</p>
                  <p className="text-lg font-semibold">{config.autoGeneration ? 'Enabled' : 'Disabled'}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-600 dark:text-gray-400">Generation Interval</p>
                  <p className="text-lg font-semibold">{config.generationInterval / 60} minutes</p>
                </div>
                <div>
                  <p className="text-sm text-gray-600 dark:text-gray-400">Max Concurrent Tasks</p>
                  <p className="text-lg font-semibold">{config.maxConcurrentTasks}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-600 dark:text-gray-400">Active Platforms</p>
                  <p className="text-lg font-semibold">{config.platforms.join(', ')}</p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Recent Activity</CardTitle>
              <CardDescription>Latest worker bot tasks</CardDescription>
            </CardHeader>
            <CardContent>
              {tasks.length === 0 ? (
                <p className="text-gray-500 text-center py-8">No tasks yet. Create a task to get started.</p>
              ) : (
                <div className="space-y-2">
                  {tasks.slice(-5).reverse().map(task => (
                    <div key={task.id} className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
                      <div className="flex-1">
                        <p className="font-medium">{task.type}</p>
                        <p className="text-sm text-gray-500">{new Date(task.startTime).toLocaleString()}</p>
                      </div>
                      <div className="flex items-center gap-3">
                        {task.status === 'processing' && (
                          <div className="text-sm text-gray-600">{task.progress}%</div>
                        )}
                        {getStatusBadge(task.status)}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="tasks" className="mt-6">
          <Card>
            <CardHeader>
              <CardTitle>All Tasks</CardTitle>
              <CardDescription>View and manage all worker bot tasks</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                {tasks.length === 0 ? (
                  <p className="text-gray-500 text-center py-8">No tasks available</p>
                ) : (
                  tasks.map(task => (
                    <div key={task.id} className="border rounded-lg p-4">
                      <div className="flex items-start justify-between">
                        <div className="flex-1">
                          <div className="flex items-center gap-2 mb-2">
                            <h3 className="font-semibold">{task.type}</h3>
                            {getStatusBadge(task.status)}
                          </div>
                          <div className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                            <p>Started: {new Date(task.startTime).toLocaleString()}</p>
                            {task.completionTime && (
                              <p>Completed: {new Date(task.completionTime).toLocaleString()}</p>
                            )}
                            {task.status === 'processing' && (
                              <div className="mt-2">
                                <div className="w-full bg-gray-200 rounded-full h-2">
                                  <div 
                                    className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                                    style={{ width: `${task.progress}%` }}
                                  />
                                </div>
                              </div>
                            )}
                          </div>
                        </div>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="create" className="space-y-4 mt-6">
          <Card>
            <CardHeader>
              <CardTitle>Create UGC Ad</CardTitle>
              <CardDescription>Generate user-generated content style advertisements</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="ugc-product">Product/Service</Label>
                <Input
                  id="ugc-product"
                  placeholder="e.g., Eco-friendly water bottle"
                  value={ugcProduct}
                  onChange={(e) => setUgcProduct(e.target.value)}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="ugc-platform">Platform</Label>
                <Select value={ugcPlatform} onValueChange={(v: any) => setUgcPlatform(v)}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="instagram">Instagram</SelectItem>
                    <SelectItem value="tiktok">TikTok</SelectItem>
                    <SelectItem value="youtube">YouTube</SelectItem>
                    <SelectItem value="facebook">Facebook</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label htmlFor="ugc-audience">Target Audience</Label>
                <Input
                  id="ugc-audience"
                  placeholder="e.g., Young professionals aged 25-35"
                  value={ugcAudience}
                  onChange={(e) => setUgcAudience(e.target.value)}
                />
              </div>
              <Button onClick={handleCreateUGCAd} className="w-full">
                <Plus className="w-4 h-4 mr-2" />
                Create UGC Ad
              </Button>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Generate Content Suggestions</CardTitle>
              <CardDescription>Get AI-powered content ideas for any topic</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="suggestion-topic">Topic</Label>
                <Input
                  id="suggestion-topic"
                  placeholder="e.g., Sustainable living"
                  value={suggestionTopic}
                  onChange={(e) => setSuggestionTopic(e.target.value)}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="suggestion-count">Number of Suggestions</Label>
                <Input
                  id="suggestion-count"
                  type="number"
                  min="1"
                  max="10"
                  value={suggestionCount}
                  onChange={(e) => setSuggestionCount(parseInt(e.target.value))}
                />
              </div>
              <Button onClick={handleGenerateContentSuggestions} className="w-full">
                <Zap className="w-4 h-4 mr-2" />
                Generate Suggestions
              </Button>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Improve Content</CardTitle>
              <CardDescription>Enhance existing content with AI-powered improvements</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="improvement-content">Content to Improve</Label>
                <Textarea
                  id="improvement-content"
                  placeholder="Paste your content here..."
                  rows={4}
                  value={improvementContent}
                  onChange={(e) => setImprovementContent(e.target.value)}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="improvement-areas">Improvement Areas (comma-separated)</Label>
                <Input
                  id="improvement-areas"
                  placeholder="e.g., clarity, engagement, SEO"
                  value={improvementAreas}
                  onChange={(e) => setImprovementAreas(e.target.value)}
                />
              </div>
              <Button onClick={handleImproveContent} className="w-full">
                <TrendingUp className="w-4 h-4 mr-2" />
                Improve Content
              </Button>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="settings" className="mt-6">
          <Card>
            <CardHeader>
              <CardTitle>Worker Bot Configuration</CardTitle>
              <CardDescription>Configure automated content generation settings</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="flex items-center justify-between">
                <div>
                  <Label htmlFor="auto-generation">Auto Generation</Label>
                  <p className="text-sm text-gray-500">Automatically generate content at intervals</p>
                </div>
                <Switch
                  id="auto-generation"
                  checked={config.autoGeneration}
                  onCheckedChange={(checked) => 
                    setConfig(prev => ({ ...prev, autoGeneration: checked }))
                  }
                  disabled={!config.enabled}
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="interval">Generation Interval (minutes)</Label>
                <Input
                  id="interval"
                  type="number"
                  min="5"
                  value={config.generationInterval / 60}
                  onChange={(e) => 
                    setConfig(prev => ({ ...prev, generationInterval: parseInt(e.target.value) * 60 }))
                  }
                  disabled={!config.enabled}
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="max-tasks">Max Concurrent Tasks</Label>
                <Input
                  id="max-tasks"
                  type="number"
                  min="1"
                  max="10"
                  value={config.maxConcurrentTasks}
                  onChange={(e) => 
                    setConfig(prev => ({ ...prev, maxConcurrentTasks: parseInt(e.target.value) }))
                  }
                  disabled={!config.enabled}
                />
              </div>

              <div className="space-y-2">
                <Label>Enabled Platforms</Label>
                <div className="space-y-2">
                  {['instagram', 'tiktok', 'youtube', 'facebook'].map(platform => (
                    <div key={platform} className="flex items-center gap-2">
                      <input
                        type="checkbox"
                        id={`platform-${platform}`}
                        checked={config.platforms.includes(platform)}
                        onChange={(e) => {
                          if (e.target.checked) {
                            setConfig(prev => ({ ...prev, platforms: [...prev.platforms, platform] }));
                          } else {
                            setConfig(prev => ({ 
                              ...prev, 
                              platforms: prev.platforms.filter(p => p !== platform) 
                            }));
                          }
                        }}
                        disabled={!config.enabled}
                        className="rounded"
                      />
                      <Label htmlFor={`platform-${platform}`} className="capitalize cursor-pointer">
                        {platform}
                      </Label>
                    </div>
                  ))}
                </div>
              </div>

              <Button 
                onClick={handleSaveConfig}
                className="w-full"
              >
                <Settings className="w-4 h-4 mr-2" />
                Save Configuration
              </Button>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
