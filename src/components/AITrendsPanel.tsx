import { useEffect, useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { useWorkerBot } from '@/hooks/useWorkerBot';
import { analyticsEngine } from '@/services/analyticsEngine';
import { TrendingUp, Sparkles, RefreshCw } from 'lucide-react';
import { toast } from 'sonner';

interface Trend {
  title: string;
  description: string;
  confidence: number;
  category: string;
  momentum: string;
}

export function AITrendsPanel({ category = 'all' }: { category?: string }) {
  const { analyzeTrends, loading } = useWorkerBot();
  const [trends, setTrends] = useState<Trend[]>([]);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);

  useEffect(() => {
    loadTrends();
    const interval = setInterval(loadTrends, 60000); // Update every minute
    return () => clearInterval(interval);
  }, [category]);

  const loadTrends = async () => {
    try {
      // Get recent analytics data
      const globalInsights = analyticsEngine.getGlobalInsights();
      const analyticsSummary = analyticsEngine.getAnalyticsSummary();

      // Prepare data for trend analysis
      const trendData = {
        insights: globalInsights,
        summary: analyticsSummary,
        timestamp: new Date().toISOString()
      };

      const result = await analyzeTrends(
        trendData,
        '24h',
        category
      );

      if (result?.trends) {
        setTrends(result.trends);
        setLastUpdate(new Date());
      }
    } catch (error) {
      console.error('Failed to load trends:', error);
      // Use fallback local trends
      const localTrends = analyticsEngine.getGlobalInsights()
        .filter(i => i.type === 'trend')
        .map(i => ({
          title: i.description,
          description: i.recommendation,
          confidence: i.confidence,
          category: i.data?.category || 'general',
          momentum: i.data?.momentum || 'stable'
        }));
      setTrends(localTrends);
    }
  };

  const handleRefresh = () => {
    loadTrends();
    toast.success('Trends refreshed');
  };

  const getMomentumColor = (momentum: string) => {
    switch (momentum) {
      case 'rising': return 'text-green-600';
      case 'falling': return 'text-red-600';
      case 'stable': return 'text-blue-600';
      default: return 'text-gray-600';
    }
  };

  const getMomentumIcon = (momentum: string) => {
    switch (momentum) {
      case 'rising': return '↗️';
      case 'falling': return '↘️';
      case 'stable': return '→';
      default: return '•';
    }
  };

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <Sparkles className="h-5 w-5" />
              AI-Detected Trends
            </CardTitle>
            <CardDescription>
              {lastUpdate ? `Last updated: ${lastUpdate.toLocaleTimeString()}` : 'Loading trends...'}
            </CardDescription>
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={handleRefresh}
            disabled={loading}
          >
            <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        {trends.length === 0 ? (
          <div className="text-center py-8 text-muted-foreground">
            <TrendingUp className="h-12 w-12 mx-auto mb-4 opacity-50" />
            <p>No trends detected yet</p>
            <p className="text-sm">Keep exploring to discover trends!</p>
          </div>
        ) : (
          <div className="space-y-4">
            {trends.map((trend, index) => (
              <div
                key={index}
                className="p-4 border rounded-lg hover:bg-accent transition-colors cursor-pointer"
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-2">
                      <span className={`text-xl ${getMomentumColor(trend.momentum)}`}>
                        {getMomentumIcon(trend.momentum)}
                      </span>
                      <h4 className="font-semibold">{trend.title}</h4>
                    </div>
                    <p className="text-sm text-muted-foreground mb-2">
                      {trend.description}
                    </p>
                    <div className="flex items-center gap-2">
                      <Badge variant="secondary">{trend.category}</Badge>
                      <Badge variant="outline">
                        {Math.round(trend.confidence * 100)}% confidence
                      </Badge>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
