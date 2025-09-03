import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { 
  Wifi, WifiOff, Settings, TestTube, 
  Server, Activity, Clock, Zap 
} from 'lucide-react';
import { pollenConnection } from '@/services/pollenConnection';
import { pollenAI } from '@/services/pollenAI';

export const PollenStatus: React.FC = () => {
  const [status, setStatus] = useState(pollenConnection.getConnectionStatus());
  const [memoryStats, setMemoryStats] = useState<any>({});
  const [customUrl, setCustomUrl] = useState('');
  const [testResult, setTestResult] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    const updateStatus = async () => {
      setStatus(pollenConnection.getConnectionStatus());
      try {
        const stats = await pollenAI.getMemoryStats();
        setMemoryStats(stats);
      } catch (error) {
        console.warn('Failed to get memory stats:', error);
      }
    };

    updateStatus();
    const interval = setInterval(updateStatus, 10000);
    return () => clearInterval(interval);
  }, []);

  const handleTestConnection = async () => {
    setIsLoading(true);
    setTestResult(null);
    
    try {
      const result = await pollenConnection.testConnection();
      setTestResult(result);
    } catch (error) {
      setTestResult({
        success: false,
        error: 'Test failed'
      });
    }
    
    setIsLoading(false);
  };

  const handleSetCustomUrl = async () => {
    if (!customUrl) return;
    
    setIsLoading(true);
    const success = await pollenConnection.setCustomBackendUrl(customUrl);
    
    if (success) {
      setStatus(pollenConnection.getConnectionStatus());
      setCustomUrl('');
    }
    
    setIsLoading(false);
  };

  const getStatusColor = () => {
    switch (status.status) {
      case 'connected': return 'default';
      case 'connecting': return 'secondary';
      default: return 'destructive';
    }
  };

  const getStatusIcon = () => {
    switch (status.status) {
      case 'connected': return <Wifi className="h-4 w-4" />;
      case 'connecting': return <Activity className="h-4 w-4 animate-pulse" />;
      default: return <WifiOff className="h-4 w-4" />;
    }
  };

  return (
    <Card className="w-full max-w-2xl">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Server className="h-5 w-5" />
          Pollen AI Backend Status
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Connection Status */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Badge variant={getStatusColor()}>
              {getStatusIcon()}
              {status.status}
            </Badge>
            <span className="text-sm text-muted-foreground">
              {status.url}
            </span>
          </div>
          <Button 
            onClick={handleTestConnection} 
            variant="outline" 
            size="sm"
            disabled={isLoading}
          >
            <TestTube className="h-4 w-4 mr-2" />
            Test Connection
          </Button>
        </div>

        {/* Test Results */}
        {testResult && (
          <div className={`p-3 rounded-md border ${
            testResult.success 
              ? 'border-green-200 bg-green-50 dark:border-green-800 dark:bg-green-950' 
              : 'border-red-200 bg-red-50 dark:border-red-800 dark:bg-red-950'
          }`}>
            <div className="flex items-center gap-2 mb-1">
              <Zap className={`h-4 w-4 ${testResult.success ? 'text-green-600' : 'text-red-600'}`} />
              <span className="font-medium">
                {testResult.success ? 'Connection Successful' : 'Connection Failed'}
              </span>
            </div>
            {testResult.latency && (
              <div className="text-sm text-muted-foreground flex items-center gap-1">
                <Clock className="h-3 w-3" />
                Response time: {testResult.latency}ms
              </div>
            )}
            {testResult.error && (
              <div className="text-sm text-red-600 mt-1">
                Error: {testResult.error}
              </div>
            )}
          </div>
        )}

        {/* Memory Stats */}
        {status.status === 'connected' && memoryStats && (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center">
              <div className="text-2xl font-bold text-primary">
                {memoryStats.shortTermSize || 0}
              </div>
              <div className="text-xs text-muted-foreground">Short Term</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-primary">
                {memoryStats.longTermPatterns || 0}
              </div>
              <div className="text-xs text-muted-foreground">Patterns</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-primary">
                {memoryStats.adaptiveSignals || 0}
              </div>
              <div className="text-xs text-muted-foreground">Signals</div>
            </div>
            <div className="text-center">
              <div className={`text-2xl font-bold ${
                memoryStats.isLearning ? 'text-green-500' : 'text-yellow-500'
              }`}>
                {memoryStats.isLearning ? '●' : '◐'}
              </div>
              <div className="text-xs text-muted-foreground">Learning</div>
            </div>
          </div>
        )}

        {/* Custom Backend URL */}
        <div className="space-y-2">
          <label className="text-sm font-medium">Custom Pollen Backend URL:</label>
          <div className="flex gap-2">
            <Input
              placeholder="http://localhost:8000"
              value={customUrl}
              onChange={(e) => setCustomUrl(e.target.value)}
            />
            <Button 
              onClick={handleSetCustomUrl}
              disabled={!customUrl || isLoading}
              variant="outline"
            >
              <Settings className="h-4 w-4" />
            </Button>
          </div>
          <p className="text-xs text-muted-foreground">
            Connect to your own Pollen AI backend instance. Leave empty to use default.
          </p>
        </div>

        {/* Instructions */}
        <div className="text-sm text-muted-foreground space-y-1">
          <p><strong>To run Pollen AI locally:</strong></p>
          <ol className="list-decimal list-inside space-y-1 text-xs">
            <li>Navigate to the docker folder in your project</li>
            <li>Run: <code className="bg-muted px-1 rounded">docker-compose up -d</code></li>
            <li>Wait for the backend to start (check logs with <code className="bg-muted px-1 rounded">docker-compose logs -f</code>)</li>
            <li>The backend will be available at http://localhost:8000</li>
          </ol>
        </div>
      </CardContent>
    </Card>
  );
};