import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Activity, TrendingUp, Users, Brain, AlertCircle, CheckCircle } from 'lucide-react';

interface HealthTrend {
  category: string;
  trend: 'increasing' | 'decreasing' | 'stable';
  change: number;
  dataPoints: number;
}

interface HealthInsight {
  type: 'trend' | 'correlation' | 'recommendation' | 'breakthrough';
  title: string;
  description: string;
  confidence: number;
  category: string;
}

export function HealthDashboard() {
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [trends, setTrends] = useState<HealthTrend[]>([]);
  const [insights, setInsights] = useState<HealthInsight[]>([]);

  const mockTrends: HealthTrend[] = [
    { category: 'fitness', trend: 'increasing', change: 15.3, dataPoints: 1240 },
    { category: 'nutrition', trend: 'stable', change: 2.1, dataPoints: 890 },
    { category: 'mental_health', trend: 'increasing', change: 8.7, dataPoints: 567 },
    { category: 'sleep', trend: 'decreasing', change: -5.2, dataPoints: 1103 },
    { category: 'medical', trend: 'stable', change: 1.4, dataPoints: 423 },
  ];

  const mockInsights: HealthInsight[] = [
    {
      type: 'correlation',
      title: 'Strong correlation between sleep quality and mental health',
      description: 'Analysis of 1,103 data points shows that users reporting better sleep quality (7-9 hours) had 32% better mental health scores.',
      confidence: 0.87,
      category: 'sleep'
    },
    {
      type: 'trend',
      title: 'Fitness activity levels increasing across all age groups',
      description: 'Average weekly exercise duration increased by 15.3% across 1,240 submissions, with the 26-35 age group showing the highest increase.',
      confidence: 0.92,
      category: 'fitness'
    },
    {
      type: 'recommendation',
      title: 'Optimal hydration timing for performance',
      description: 'Data suggests drinking water 30 minutes before exercise improves performance metrics by 18% compared to during-exercise hydration.',
      confidence: 0.79,
      category: 'nutrition'
    },
    {
      type: 'breakthrough',
      title: 'New pattern in stress reduction techniques',
      description: 'Combination of breathing exercises + short walks reduces stress levels 40% more effectively than either method alone.',
      confidence: 0.85,
      category: 'mental_health'
    },
  ];

  useEffect(() => {
    setTrends(mockTrends);
    setInsights(mockInsights);
  }, []);

  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case 'increasing':
        return <TrendingUp className="w-4 h-4 text-green-600" />;
      case 'decreasing':
        return <TrendingUp className="w-4 h-4 text-red-600 rotate-180" />;
      default:
        return <Activity className="w-4 h-4 text-blue-600" />;
    }
  };

  const getInsightIcon = (type: string) => {
    switch (type) {
      case 'breakthrough':
        return <AlertCircle className="w-5 h-5 text-purple-600" />;
      case 'correlation':
        return <Brain className="w-5 h-5 text-blue-600" />;
      case 'trend':
        return <TrendingUp className="w-5 h-5 text-green-600" />;
      default:
        return <CheckCircle className="w-5 h-5 text-orange-600" />;
    }
  };

  const filteredInsights = selectedCategory === 'all' 
    ? insights 
    : insights.filter(i => i.category === selectedCategory);

  const categoryLabels: Record<string, string> = {
    fitness: 'Fitness',
    nutrition: 'Nutrition',
    mental_health: 'Mental Health',
    sleep: 'Sleep',
    medical: 'Medical'
  };

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Submissions</CardTitle>
            <Users className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">4,223</div>
            <p className="text-xs text-muted-foreground">
              +12.5% from last month
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Active Categories</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">5</div>
            <p className="text-xs text-muted-foreground">
              Fitness leading with 1,240 entries
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">AI Insights Generated</CardTitle>
            <Brain className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">127</div>
            <p className="text-xs text-muted-foreground">
              4 breakthroughs this week
            </p>
          </CardContent>
        </Card>
      </div>

      <Tabs defaultValue="trends" className="w-full">
        <TabsList>
          <TabsTrigger value="trends">Data Trends</TabsTrigger>
          <TabsTrigger value="insights">AI Insights</TabsTrigger>
          <TabsTrigger value="correlations">Correlations</TabsTrigger>
        </TabsList>

        <TabsContent value="trends" className="mt-6">
          <Card>
            <CardHeader>
              <CardTitle>Category Trends</CardTitle>
              <CardDescription>Current trends across health data categories</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {trends.map((trend) => (
                  <div key={trend.category} className="flex items-center justify-between p-4 border rounded-lg">
                    <div className="flex items-center space-x-4">
                      {getTrendIcon(trend.trend)}
                      <div>
                        <h3 className="font-semibold capitalize">
                          {categoryLabels[trend.category]}
                        </h3>
                        <p className="text-sm text-muted-foreground">
                          {trend.dataPoints} data points
                        </p>
                      </div>
                    </div>
                    <div className="text-right">
                      <Badge variant={trend.trend === 'increasing' ? 'default' : trend.trend === 'decreasing' ? 'destructive' : 'secondary'}>
                        {trend.change > 0 ? '+' : ''}{trend.change}%
                      </Badge>
                      <p className="text-xs text-muted-foreground mt-1 capitalize">{trend.trend}</p>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="insights" className="mt-6">
          <div className="mb-4">
            <Select value={selectedCategory} onValueChange={setSelectedCategory}>
              <SelectTrigger className="w-[200px]">
                <SelectValue placeholder="Select category" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Categories</SelectItem>
                <SelectItem value="fitness">Fitness</SelectItem>
                <SelectItem value="nutrition">Nutrition</SelectItem>
                <SelectItem value="mental_health">Mental Health</SelectItem>
                <SelectItem value="sleep">Sleep</SelectItem>
                <SelectItem value="medical">Medical</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-4">
            {filteredInsights.map((insight, index) => (
              <Card key={index}>
                <CardHeader>
                  <div className="flex items-start justify-between">
                    <div className="flex items-start space-x-3">
                      {getInsightIcon(insight.type)}
                      <div>
                        <CardTitle className="text-lg">{insight.title}</CardTitle>
                        <div className="flex items-center gap-2 mt-2">
                          <Badge variant="outline" className="capitalize">{insight.type}</Badge>
                          <Badge variant="secondary" className="capitalize">{categoryLabels[insight.category]}</Badge>
                        </div>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="text-sm font-medium">Confidence</div>
                      <div className="text-2xl font-bold">{Math.round(insight.confidence * 100)}%</div>
                    </div>
                  </div>
                </CardHeader>
                <CardContent>
                  <p className="text-muted-foreground">{insight.description}</p>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="correlations" className="mt-6">
          <Card>
            <CardHeader>
              <CardTitle>Data Correlations</CardTitle>
              <CardDescription>Discovered relationships between health metrics</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="p-4 border rounded-lg">
                  <div className="flex items-center justify-between mb-2">
                    <h3 className="font-semibold">Sleep Duration ↔ Fitness Performance</h3>
                    <Badge>Strong (r=0.78)</Badge>
                  </div>
                  <p className="text-sm text-muted-foreground">
                    Users with 7-9 hours of sleep show 34% better fitness performance metrics
                  </p>
                </div>

                <div className="p-4 border rounded-lg">
                  <div className="flex items-center justify-between mb-2">
                    <h3 className="font-semibold">Nutrition Quality ↔ Energy Levels</h3>
                    <Badge>Moderate (r=0.62)</Badge>
                  </div>
                  <p className="text-sm text-muted-foreground">
                    Balanced protein intake correlates with sustained energy throughout the day
                  </p>
                </div>

                <div className="p-4 border rounded-lg">
                  <div className="flex items-center justify-between mb-2">
                    <h3 className="font-semibold">Stress Levels ↔ Sleep Quality</h3>
                    <Badge variant="destructive">Strong Negative (r=-0.71)</Badge>
                  </div>
                  <p className="text-sm text-muted-foreground">
                    Higher stress levels significantly impact sleep quality and duration
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
