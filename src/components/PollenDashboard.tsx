import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { 
  Brain, 
  Target, 
  Music, 
  Image, 
  Calendar,
  Share2,
  TrendingUp,
  Loader2,
  CheckCircle,
  AlertTriangle
} from 'lucide-react';
import { pollenAdaptiveService } from '@/services/pollenAdaptiveService';
import type { 
  TaskProposal, 
  TaskSolution, 
  AdCreationResult,
  MusicGenerationResult,
  ImageGenerationResult,
  TaskAutomationResult,
  SocialPostResult,
  TrendAnalysisResult
} from '@/services/pollenAdaptiveService';

export function PollenDashboard() {
  const [activeTab, setActiveTab] = useState('tasks');
  const [loading, setLoading] = useState(false);
  
  // Task Management
  const [taskInput, setTaskInput] = useState('');
  const [taskProposal, setTaskProposal] = useState<TaskProposal | null>(null);
  const [taskSolution, setTaskSolution] = useState<TaskSolution | null>(null);

  // Ad Creation
  const [adInput, setAdInput] = useState('');
  const [adResult, setAdResult] = useState<AdCreationResult | null>(null);

  // Music Generation
  const [musicInput, setMusicInput] = useState('');
  const [musicResult, setMusicResult] = useState<MusicGenerationResult | null>(null);

  // Image Generation
  const [imageInput, setImageInput] = useState('');
  const [imageResult, setImageResult] = useState<ImageGenerationResult | null>(null);

  // Task Automation
  const [automationTask, setAutomationTask] = useState('');
  const [automationSchedule, setAutomationSchedule] = useState('');
  const [automationResult, setAutomationResult] = useState<TaskAutomationResult | null>(null);

  // Social Media
  const [socialInput, setSocialInput] = useState('');
  const [socialResult, setSocialResult] = useState<SocialPostResult | null>(null);

  // Trend Analysis
  const [trendInput, setTrendInput] = useState('');
  const [trendResult, setTrendResult] = useState<TrendAnalysisResult | null>(null);

  const handleProposeTask = async () => {
    if (!taskInput.trim()) return;
    
    setLoading(true);
    try {
      const proposal = await pollenAdaptiveService.proposeTask(taskInput);
      setTaskProposal(proposal);
    } catch (error) {
      console.error('Failed to propose task:', error);
    }
    setLoading(false);
  };

  const handleSolveTask = async () => {
    if (!taskInput.trim()) return;
    
    setLoading(true);
    try {
      const solution = await pollenAdaptiveService.solveTask(taskInput);
      setTaskSolution(solution);
    } catch (error) {
      console.error('Failed to solve task:', error);
    }
    setLoading(false);
  };

  const handleCreateAd = async () => {
    if (!adInput.trim()) return;
    
    setLoading(true);
    try {
      const ad = await pollenAdaptiveService.createAdvertisement(adInput);
      setAdResult(ad);
    } catch (error) {
      console.error('Failed to create ad:', error);
    }
    setLoading(false);
  };

  const handleGenerateMusic = async () => {
    if (!musicInput.trim()) return;
    
    setLoading(true);
    try {
      const music = await pollenAdaptiveService.generateMusic(musicInput);
      setMusicResult(music);
    } catch (error) {
      console.error('Failed to generate music:', error);
    }
    setLoading(false);
  };

  const handleGenerateImage = async () => {
    if (!imageInput.trim()) return;
    
    setLoading(true);
    try {
      const image = await pollenAdaptiveService.generateImage(imageInput);
      setImageResult(image);
    } catch (error) {
      console.error('Failed to generate image:', error);
    }
    setLoading(false);
  };

  const handleAutomateTask = async () => {
    if (!automationTask.trim() || !automationSchedule.trim()) return;
    
    setLoading(true);
    try {
      const automation = await pollenAdaptiveService.automateTask(automationTask, automationSchedule);
      setAutomationResult(automation);
    } catch (error) {
      console.error('Failed to automate task:', error);
    }
    setLoading(false);
  };

  const handleCurateSocial = async () => {
    if (!socialInput.trim()) return;
    
    setLoading(true);
    try {
      const social = await pollenAdaptiveService.curateSocialPost(socialInput);
      setSocialResult(social);
    } catch (error) {
      console.error('Failed to curate social post:', error);
    }
    setLoading(false);
  };

  const handleAnalyzeTrends = async () => {
    if (!trendInput.trim()) return;
    
    setLoading(true);
    try {
      const trends = await pollenAdaptiveService.analyzeTrends(trendInput);
      setTrendResult(trends);
    } catch (error) {
      console.error('Failed to analyze trends:', error);
    }
    setLoading(false);
  };

  return (
    <div className="min-h-screen bg-background p-6">
      <div className="max-w-7xl mx-auto">
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-foreground mb-2">
            Pollen Adaptive Intelligence
          </h1>
          <p className="text-muted-foreground text-lg">
            Advanced AI capabilities for content creation, task automation, and intelligent analysis
          </p>
        </div>

        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
          <TabsList className="grid w-full grid-cols-4 lg:grid-cols-7">
            <TabsTrigger value="tasks" className="flex items-center gap-2">
              <Brain className="w-4 h-4" />
              <span className="hidden sm:inline">Tasks</span>
            </TabsTrigger>
            <TabsTrigger value="ads" className="flex items-center gap-2">
              <Target className="w-4 h-4" />
              <span className="hidden sm:inline">Ads</span>
            </TabsTrigger>
            <TabsTrigger value="music" className="flex items-center gap-2">
              <Music className="w-4 h-4" />
              <span className="hidden sm:inline">Music</span>
            </TabsTrigger>
            <TabsTrigger value="images" className="flex items-center gap-2">
              <Image className="w-4 h-4" />
              <span className="hidden sm:inline">Images</span>
            </TabsTrigger>
            <TabsTrigger value="automation" className="flex items-center gap-2">
              <Calendar className="w-4 h-4" />
              <span className="hidden sm:inline">Automation</span>
            </TabsTrigger>
            <TabsTrigger value="social" className="flex items-center gap-2">
              <Share2 className="w-4 h-4" />
              <span className="hidden sm:inline">Social</span>
            </TabsTrigger>
            <TabsTrigger value="trends" className="flex items-center gap-2">
              <TrendingUp className="w-4 h-4" />
              <span className="hidden sm:inline">Trends</span>
            </TabsTrigger>
          </TabsList>

          <TabsContent value="tasks">
            <div className="grid gap-6 md:grid-cols-2">
              <Card>
                <CardHeader>
                  <CardTitle>Task Proposal</CardTitle>
                  <CardDescription>Let Pollen AI propose and structure tasks for you</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <Textarea
                    placeholder="Describe what you want to accomplish..."
                    value={taskInput}
                    onChange={(e) => setTaskInput(e.target.value)}
                  />
                  <div className="flex gap-2">
                    <Button onClick={handleProposeTask} disabled={loading}>
                      {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Brain className="w-4 h-4" />}
                      Propose Task
                    </Button>
                    <Button onClick={handleSolveTask} disabled={loading} variant="outline">
                      {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <CheckCircle className="w-4 h-4" />}
                      Solve Task
                    </Button>
                  </div>
                </CardContent>
              </Card>

              <div className="space-y-4">
                {taskProposal && (
                  <Card>
                    <CardHeader>
                      <CardTitle>{taskProposal.title}</CardTitle>
                      <div className="flex gap-2">
                        <Badge variant={taskProposal.complexity === 'high' ? 'destructive' : 'default'}>
                          {taskProposal.complexity}
                        </Badge>
                        <Badge variant="outline">{taskProposal.estimatedTime}</Badge>
                      </div>
                    </CardHeader>
                    <CardContent>
                      <p className="text-sm text-muted-foreground">{taskProposal.description}</p>
                    </CardContent>
                  </Card>
                )}

                {taskSolution && (
                  <Card>
                    <CardHeader>
                      <CardTitle>Solution</CardTitle>
                      <CardDescription>Confidence: {Math.round(taskSolution.confidence * 100)}%</CardDescription>
                    </CardHeader>
                    <CardContent>
                      <p className="text-sm mb-4">{taskSolution.solution}</p>
                      {taskSolution.steps.length > 0 && (
                        <div>
                          <h4 className="font-semibold mb-2">Steps:</h4>
                          <ul className="list-disc list-inside space-y-1">
                            {taskSolution.steps.map((step, index) => (
                              <li key={index} className="text-sm text-muted-foreground">{step}</li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </CardContent>
                  </Card>
                )}
              </div>
            </div>
          </TabsContent>

          <TabsContent value="ads">
            <div className="grid gap-6 md:grid-cols-2">
              <Card>
                <CardHeader>
                  <CardTitle>Advertisement Creation</CardTitle>
                  <CardDescription>Generate compelling ads for your products or services</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <Textarea
                    placeholder="Describe your product or service..."
                    value={adInput}
                    onChange={(e) => setAdInput(e.target.value)}
                  />
                  <Button onClick={handleCreateAd} disabled={loading}>
                    {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Target className="w-4 h-4" />}
                    Create Advertisement
                  </Button>
                </CardContent>
              </Card>

              {adResult && (
                <Card>
                  <CardHeader>
                    <CardTitle>{adResult.title}</CardTitle>
                    <div className="flex gap-2">
                      <Badge>CTR: {adResult.estimatedCTR}%</Badge>
                      <Badge variant="outline">{adResult.platform}</Badge>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <p className="text-sm mb-4">{adResult.content}</p>
                    <p className="text-xs text-muted-foreground">
                      Target: {adResult.targetAudience}
                    </p>
                  </CardContent>
                </Card>
              )}
            </div>
          </TabsContent>

          <TabsContent value="music">
            <div className="grid gap-6 md:grid-cols-2">
              <Card>
                <CardHeader>
                  <CardTitle>Music Generation</CardTitle>
                  <CardDescription>Create original music compositions with AI</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <Input
                    placeholder="e.g., Upbeat electronic dance music"
                    value={musicInput}
                    onChange={(e) => setMusicInput(e.target.value)}
                  />
                  <Button onClick={handleGenerateMusic} disabled={loading}>
                    {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Music className="w-4 h-4" />}
                    Generate Music
                  </Button>
                </CardContent>
              </Card>

              {musicResult && (
                <Card>
                  <CardHeader>
                    <CardTitle>{musicResult.title}</CardTitle>
                    <div className="flex gap-2">
                      <Badge>{musicResult.genre}</Badge>
                      <Badge variant="outline">{musicResult.duration}</Badge>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <p className="text-sm mb-4">{musicResult.description}</p>
                    {musicResult.audioUrl && (
                      <audio controls className="w-full">
                        <source src={musicResult.audioUrl} type="audio/mpeg" />
                      </audio>
                    )}
                  </CardContent>
                </Card>
              )}
            </div>
          </TabsContent>

          <TabsContent value="images">
            <div className="grid gap-6 md:grid-cols-2">
              <Card>
                <CardHeader>
                  <CardTitle>Image Generation</CardTitle>
                  <CardDescription>Create stunning visuals with AI</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <Textarea
                    placeholder="Describe the image you want to generate..."
                    value={imageInput}
                    onChange={(e) => setImageInput(e.target.value)}
                  />
                  <Button onClick={handleGenerateImage} disabled={loading}>
                    {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Image className="w-4 h-4" />}
                    Generate Image
                  </Button>
                </CardContent>
              </Card>

              {imageResult && (
                <Card>
                  <CardHeader>
                    <CardTitle>Generated Image</CardTitle>
                    <div className="flex gap-2">
                      <Badge>{imageResult.style}</Badge>
                      <Badge variant="outline">{imageResult.dimensions}</Badge>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <p className="text-sm mb-4">{imageResult.description}</p>
                    {imageResult.imageUrl && (
                      <img 
                        src={imageResult.imageUrl} 
                        alt={imageResult.description}
                        className="w-full rounded-lg"
                      />
                    )}
                  </CardContent>
                </Card>
              )}
            </div>
          </TabsContent>

          <TabsContent value="automation">
            <div className="grid gap-6 md:grid-cols-2">
              <Card>
                <CardHeader>
                  <CardTitle>Task Automation</CardTitle>
                  <CardDescription>Automate repetitive tasks with smart scheduling</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <Input
                    placeholder="Task type (e.g., daily_planning, email_management)"
                    value={automationTask}
                    onChange={(e) => setAutomationTask(e.target.value)}
                  />
                  <Input
                    placeholder="Schedule (e.g., daily at 9 AM)"
                    value={automationSchedule}
                    onChange={(e) => setAutomationSchedule(e.target.value)}
                  />
                  <Button onClick={handleAutomateTask} disabled={loading}>
                    {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Calendar className="w-4 h-4" />}
                    Setup Automation
                  </Button>
                </CardContent>
              </Card>

              {automationResult && (
                <Card>
                  <CardHeader>
                    <CardTitle>{automationResult.taskType}</CardTitle>
                    <div className="flex gap-2">
                      <Badge variant={automationResult.status === 'completed' ? 'default' : 'outline'}>
                        {automationResult.status}
                      </Badge>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <p className="text-sm mb-2"><strong>Schedule:</strong> {automationResult.schedule}</p>
                    <p className="text-sm text-muted-foreground">{automationResult.automationScript}</p>
                  </CardContent>
                </Card>
              )}
            </div>
          </TabsContent>

          <TabsContent value="social">
            <div className="grid gap-6 md:grid-cols-2">
              <Card>
                <CardHeader>
                  <CardTitle>Social Media Curation</CardTitle>
                  <CardDescription>Create engaging social media content</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <Input
                    placeholder="Topic or theme for your post..."
                    value={socialInput}
                    onChange={(e) => setSocialInput(e.target.value)}
                  />
                  <Button onClick={handleCurateSocial} disabled={loading}>
                    {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Share2 className="w-4 h-4" />}
                    Curate Post
                  </Button>
                </CardContent>
              </Card>

              {socialResult && (
                <Card>
                  <CardHeader>
                    <CardTitle>Social Media Post</CardTitle>
                    <div className="flex gap-2">
                      <Badge>{socialResult.platform}</Badge>
                      <Badge variant="outline">Score: {socialResult.engagementScore}/10</Badge>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <p className="text-sm mb-4">{socialResult.content}</p>
                    <div className="flex flex-wrap gap-1 mb-2">
                      {socialResult.hashtags.map((tag, index) => (
                        <Badge key={index} variant="secondary">{tag}</Badge>
                      ))}
                    </div>
                    <p className="text-xs text-muted-foreground">
                      Optimal time: {socialResult.optimalPostTime}
                    </p>
                  </CardContent>
                </Card>
              )}
            </div>
          </TabsContent>

          <TabsContent value="trends">
            <div className="grid gap-6 md:grid-cols-2">
              <Card>
                <CardHeader>
                  <CardTitle>Trend Analysis</CardTitle>
                  <CardDescription>Analyze market trends and get predictions</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <Input
                    placeholder="Topic to analyze (e.g., AI technology, sustainable fashion)"
                    value={trendInput}
                    onChange={(e) => setTrendInput(e.target.value)}
                  />
                  <Button onClick={handleAnalyzeTrends} disabled={loading}>
                    {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <TrendingUp className="w-4 h-4" />}
                    Analyze Trends
                  </Button>
                </CardContent>
              </Card>

              {trendResult && (
                <Card>
                  <CardHeader>
                    <CardTitle>{trendResult.topic}</CardTitle>
                    <div className="flex gap-2">
                      <Badge variant={trendResult.trendScore > 0.7 ? 'default' : 'outline'}>
                        Score: {Math.round(trendResult.trendScore * 100)}%
                      </Badge>
                      <Badge variant="outline">{trendResult.timeframe}</Badge>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      <div>
                        <h4 className="font-semibold mb-2">Key Insights:</h4>
                        <ul className="list-disc list-inside space-y-1">
                          {trendResult.insights.map((insight, index) => (
                            <li key={index} className="text-sm text-muted-foreground">{insight}</li>
                          ))}
                        </ul>
                      </div>
                      <div>
                        <h4 className="font-semibold mb-2">Predictions:</h4>
                        <ul className="list-disc list-inside space-y-1">
                          {trendResult.predictions.map((prediction, index) => (
                            <li key={index} className="text-sm text-muted-foreground">{prediction}</li>
                          ))}
                        </ul>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              )}
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}