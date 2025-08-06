import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Progress } from '@/components/ui/progress';
import { 
  TrendingUp, 
  TrendingDown, 
  BarChart3, 
  Lightbulb, 
  ShoppingCart, 
  AlertTriangle,
  RefreshCw,
  X,
  Eye,
  MessageSquare,
  Share2
} from 'lucide-react';
import { 
  trendBasedContentGenerator,
  type TrendScore,
  type TrendAlert,
  type GeneratedContent,
  type ProductRecommendation
} from '@/services/trendBasedContentGenerator';
import { useToast } from '@/hooks/use-toast';

export default function TrendDashboard() {
  const [trends, setTrends] = useState<TrendScore[]>([]);
  const [alerts, setAlerts] = useState<TrendAlert[]>([]);
  const [content, setContent] = useState<GeneratedContent[]>([]);
  const [recommendations, setRecommendations] = useState<ProductRecommendation[]>([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [progress, setProgress] = useState(0);
  const { toast } = useToast();

  useEffect(() => {
    loadData();
  }, []);

  const loadData = () => {
    setTrends(trendBasedContentGenerator.getTrendScores());
    setAlerts(trendBasedContentGenerator.getAlerts());
    setContent(trendBasedContentGenerator.getGeneratedContent());
  };

  const runTrendAnalysis = async () => {
    setIsAnalyzing(true);
    setProgress(10);
    
    try {
      toast({
        title: "Starting Trend Analysis",
        description: "Analyzing trends from multiple sources...",
      });

      const result = await trendBasedContentGenerator.run();
      
      setProgress(100);
      setTrends(result.trends);
      setAlerts([...alerts, ...result.alerts]);
      setContent([...content, ...result.content]);
      setRecommendations(result.recommendations);
      
      toast({
        title: "Analysis Complete",
        description: `Found ${result.trends.length} trending topics, generated ${result.content.length} content pieces`,
      });
    } catch (error) {
      console.error('Trend analysis failed:', error);
      toast({
        title: "Analysis Failed",
        description: "Unable to complete trend analysis. Please try again.",
        variant: "destructive",
      });
    } finally {
      setIsAnalyzing(false);
      setProgress(0);
    }
  };

  const dismissAlert = async (alertId: string) => {
    await trendBasedContentGenerator.dismissAlert(alertId);
    setAlerts(alerts.filter(alert => alert.id !== alertId));
  };

  const formatScore = (score: number) => {
    return (score * 100).toFixed(1);
  };

  const getSentimentIcon = (sentiment: number) => {
    return sentiment > 0 ? <TrendingUp className="h-4 w-4 text-green-500" /> : <TrendingDown className="h-4 w-4 text-red-500" />;
  };

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold gradient-text">Trend Intelligence Dashboard</h1>
          <p className="text-muted-foreground">AI-powered trend analysis and content generation</p>
        </div>
        <Button 
          onClick={runTrendAnalysis} 
          disabled={isAnalyzing}
          className="gradient-bg"
        >
          <RefreshCw className={`mr-2 h-4 w-4 ${isAnalyzing ? 'animate-spin' : ''}`} />
          {isAnalyzing ? 'Analyzing...' : 'Analyze Trends'}
        </Button>
      </div>

      {isAnalyzing && (
        <Card>
          <CardContent className="pt-6">
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm text-muted-foreground">Analysis Progress</span>
                <span className="text-sm font-medium">{progress}%</span>
              </div>
              <Progress value={progress} className="h-2" />
            </div>
          </CardContent>
        </Card>
      )}

      {/* Active Alerts */}
      {alerts.length > 0 && (
        <div className="space-y-4">
          <h2 className="text-xl font-semibold flex items-center">
            <AlertTriangle className="mr-2 h-5 w-5 text-amber-500" />
            Trending Alerts
          </h2>
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            {alerts.slice(0, 6).map((alert) => (
              <Alert key={alert.id} className="border-amber-200 bg-amber-50 dark:bg-amber-950/20">
                <AlertTriangle className="h-4 w-4 text-amber-500" />
                <AlertDescription className="flex items-start justify-between">
                  <div>
                    <p className="font-medium text-amber-800 dark:text-amber-200">{alert.topic}</p>
                    <p className="text-sm text-amber-600 dark:text-amber-300">Score: {formatScore(alert.score)}%</p>
                  </div>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => dismissAlert(alert.id)}
                    className="text-amber-600 hover:text-amber-800"
                  >
                    <X className="h-4 w-4" />
                  </Button>
                </AlertDescription>
              </Alert>
            ))}
          </div>
        </div>
      )}

      <Tabs defaultValue="trends" className="space-y-4">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="trends">
            <BarChart3 className="mr-2 h-4 w-4" />
            Trends
          </TabsTrigger>
          <TabsTrigger value="content">
            <MessageSquare className="mr-2 h-4 w-4" />
            Generated Content
          </TabsTrigger>
          <TabsTrigger value="products">
            <ShoppingCart className="mr-2 h-4 w-4" />
            Recommendations
          </TabsTrigger>
          <TabsTrigger value="insights">
            <Lightbulb className="mr-2 h-4 w-4" />
            Insights
          </TabsTrigger>
        </TabsList>

        <TabsContent value="trends" className="space-y-4">
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            {trends.slice(0, 12).map((trend, index) => (
              <Card key={index} className="hover:shadow-lg transition-shadow">
                <CardHeader className="pb-3">
                  <div className="flex items-start justify-between">
                    <Badge variant="secondary" className="mb-2">
                      Rank #{index + 1}
                    </Badge>
                    <div className="flex items-center space-x-1">
                      {getSentimentIcon(trend.sentiment)}
                      <span className="text-sm font-medium">{formatScore(trend.score)}%</span>
                    </div>
                  </div>
                  <CardTitle className="text-sm leading-relaxed line-clamp-3">
                    {trend.topic}
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="flex items-center justify-between text-xs text-muted-foreground">
                    <span>{trend.source}</span>
                    <span>{new Date(trend.timestamp).toLocaleDateString()}</span>
                  </div>
                  <div className="mt-2 flex space-x-2">
                    <Button variant="outline" size="sm">
                      <Eye className="mr-1 h-3 w-3" />
                      View
                    </Button>
                    <Button variant="outline" size="sm">
                      <Share2 className="mr-1 h-3 w-3" />
                      Share
                    </Button>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="content" className="space-y-4">
          <div className="grid gap-4 md:grid-cols-2">
            {content.map((item) => (
              <Card key={item.id} className="hover:shadow-lg transition-shadow">
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <Badge variant={item.type === 'ad' ? 'default' : item.type === 'post' ? 'secondary' : 'outline'}>
                      {item.type.toUpperCase()}
                    </Badge>
                    <span className="text-xs text-muted-foreground">
                      {new Date(item.timestamp).toLocaleDateString()}
                    </span>
                  </div>
                  <CardTitle className="text-lg">{item.topic}</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-muted-foreground mb-4 line-clamp-3">
                    {item.content}
                  </p>
                  <div className="space-y-2">
                    <div className="text-xs">
                      <span className="font-medium">Target:</span> {item.targetAudience}
                    </div>
                    <div className="text-xs">
                      <span className="font-medium">Platform:</span> {item.platform}
                    </div>
                  </div>
                  <div className="mt-4 flex space-x-2">
                    <Button variant="outline" size="sm">
                      <Eye className="mr-1 h-3 w-3" />
                      Preview
                    </Button>
                    <Button variant="outline" size="sm">
                      <Share2 className="mr-1 h-3 w-3" />
                      Use Content
                    </Button>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
          {content.length === 0 && (
            <Card>
              <CardContent className="text-center py-12">
                <MessageSquare className="mx-auto h-12 w-12 text-muted-foreground mb-4" />
                <h3 className="text-lg font-medium mb-2">No Content Generated Yet</h3>
                <p className="text-muted-foreground mb-4">Run trend analysis to generate AI-powered content</p>
                <Button onClick={runTrendAnalysis} variant="outline">
                  Start Analysis
                </Button>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="products" className="space-y-4">
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            {recommendations.map((product) => (
              <Card key={product.id} className="hover:shadow-lg transition-shadow">
                <CardHeader>
                  <CardTitle className="text-lg">{product.name}</CardTitle>
                  <Badge variant="outline" className="w-fit">
                    {formatScore(product.relevanceScore)}% relevant
                  </Badge>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-muted-foreground mb-4">{product.description}</p>
                  <div className="space-y-2 text-xs">
                    <div>
                      <span className="font-medium">Trend:</span> {product.trendTopic}
                    </div>
                    <div>
                      <span className="font-medium">Source:</span> {product.source}
                    </div>
                  </div>
                  <Button className="w-full mt-4" variant="outline">
                    <ShoppingCart className="mr-2 h-4 w-4" />
                    View Product
                  </Button>
                </CardContent>
              </Card>
            ))}
          </div>
          {recommendations.length === 0 && (
            <Card>
              <CardContent className="text-center py-12">
                <ShoppingCart className="mx-auto h-12 w-12 text-muted-foreground mb-4" />
                <h3 className="text-lg font-medium mb-2">No Recommendations Yet</h3>
                <p className="text-muted-foreground mb-4">Run trend analysis to get product recommendations</p>
                <Button onClick={runTrendAnalysis} variant="outline">
                  Start Analysis
                </Button>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="insights" className="space-y-4">
          <div className="grid gap-4 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center">
                  <BarChart3 className="mr-2 h-5 w-5" />
                  Trend Statistics
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="flex justify-between">
                    <span>Total Trends Tracked</span>
                    <Badge variant="secondary">{trends.length}</Badge>
                  </div>
                  <div className="flex justify-between">
                    <span>Active Alerts</span>
                    <Badge variant="destructive">{alerts.length}</Badge>
                  </div>
                  <div className="flex justify-between">
                    <span>Content Generated</span>
                    <Badge variant="default">{content.length}</Badge>
                  </div>
                  <div className="flex justify-between">
                    <span>Avg. Trend Score</span>
                    <Badge variant="outline">
                      {trends.length > 0 ? formatScore(trends.reduce((acc, t) => acc + t.score, 0) / trends.length) : '0'}%
                    </Badge>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center">
                  <Lightbulb className="mr-2 h-5 w-5" />
                  Top Categories
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {['Technology', 'AI/ML', 'Business', 'Innovation', 'Startups'].map((category, index) => (
                    <div key={category} className="flex items-center justify-between">
                      <span className="text-sm">{category}</span>
                      <div className="flex items-center space-x-2">
                        <div className="w-20 bg-muted rounded-full h-2">
                          <div 
                            className="h-2 bg-primary rounded-full" 
                            style={{width: `${Math.max(20, 100 - index * 20)}%`}}
                          />
                        </div>
                        <span className="text-xs text-muted-foreground">
                          {Math.max(20, 100 - index * 20)}%
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}