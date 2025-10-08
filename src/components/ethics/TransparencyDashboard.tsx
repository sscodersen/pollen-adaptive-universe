import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Eye, TrendingUp, Shield } from 'lucide-react';

interface DecisionLog {
  logId: string;
  decisionType: string;
  factors: any[];
  reasoning: string;
  confidence: number;
  createdAt: string;
}

export function TransparencyDashboard({ userId }: { userId?: string }) {
  const [logs, setLogs] = useState<DecisionLog[]>([]);
  const [biasStats, setBiasStats] = useState<any>(null);

  useEffect(() => {
    fetchTransparencyData();
  }, [userId]);

  const fetchTransparencyData = async () => {
    try {
      const logsResponse = await fetch(`/api/ethics/transparency${userId ? `?userId=${userId}` : ''}`);
      const logsData = await logsResponse.json();
      if (logsData.success) {
        setLogs(logsData.logs);
      }

      const statsResponse = await fetch('/api/ethics/bias-stats');
      const statsData = await statsResponse.json();
      if (statsData.success) {
        setBiasStats(statsData.stats);
      }
    } catch (error) {
      console.error('Error fetching transparency data:', error);
    }
  };

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Detections</CardTitle>
            <Shield className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{biasStats?.totalDetections || 0}</div>
            <p className="text-xs text-muted-foreground">Bias incidents detected</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Mitigation Rate</CardTitle>
            <TrendingUp className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{biasStats?.mitigationRate || '0%'}</div>
            <p className="text-xs text-muted-foreground">Successfully mitigated</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Transparency Score</CardTitle>
            <Eye className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">94%</div>
            <p className="text-xs text-muted-foreground">AI decision visibility</p>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>AI Decision Log</CardTitle>
          <CardDescription>
            Transparent view of how AI makes recommendations and decisions
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {logs.map((log) => (
              <div key={log.logId} className="border-b pb-4 last:border-0">
                <div className="flex items-start justify-between">
                  <div className="space-y-1">
                    <div className="flex items-center gap-2">
                      <Badge variant="outline">{log.decisionType}</Badge>
                      <span className="text-sm text-muted-foreground">
                        Confidence: {(log.confidence * 100).toFixed(0)}%
                      </span>
                    </div>
                    <p className="text-sm">{log.reasoning}</p>
                    <p className="text-xs text-muted-foreground">
                      {new Date(log.createdAt).toLocaleString()}
                    </p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
