
import React from 'react';
import { useIntelligenceEngine } from '../hooks/useIntelligenceEngine';
import InsightPost from './InsightPost';
import { Button } from './ui/button';
import { Loader, Sparkles } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent } from './ui/card';

const InsightFeed: React.FC = () => {
  const { insights, isGenerating, generateInsights } = useIntelligenceEngine();

  return (
    <Card className="bg-slate-900/50 border-slate-800 h-full flex flex-col">
      <CardHeader className="flex flex-row items-center justify-between p-4 border-b border-slate-800">
        <CardTitle className="text-lg text-white">AI Insights</CardTitle>
        <Button onClick={generateInsights} disabled={isGenerating} size="sm">
          {isGenerating ? (
            <Loader className="w-4 h-4 mr-2 animate-spin" />
          ) : (
            <Sparkles className="w-4 h-4 mr-2" />
          )}
          {isGenerating ? 'Generating...' : 'New Insights'}
        </Button>
      </CardHeader>
      <CardContent className="flex-1 overflow-y-auto space-y-4 p-4 scrollbar-thin scrollbar-track-slate-800/50 scrollbar-thumb-slate-700/50">
        {insights.map(insight => (
          <InsightPost key={insight.id} insight={insight} />
        ))}
      </CardContent>
    </Card>
  );
};

export default InsightFeed;
