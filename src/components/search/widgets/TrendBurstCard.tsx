import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { TrendingUp, Flame } from 'lucide-react';
import type { SearchResult } from '@/types/search';

interface TrendBurstCardProps {
  result: SearchResult;
}

export const TrendBurstCard: React.FC<TrendBurstCardProps> = ({ result }) => {
  const trend = result.content;
  const growth = result.metadata?.growth || 0;
  const volume = result.metadata?.volume || 0;

  return (
    <Card className="group hover:shadow-2xl transition-all duration-300 border-2 hover:border-orange-300 dark:hover:border-orange-700 bg-gradient-to-br from-orange-50 to-red-50 dark:from-orange-900/20 dark:to-red-900/20 backdrop-blur-sm">
      <CardHeader>
        <div className="flex items-start justify-between">
          <Badge className="mb-2 bg-orange-500">
            <Flame className="w-3 h-3 mr-1" />
            Trending
          </Badge>
          <Badge variant="outline" className="text-xs">
            <TrendingUp className="w-3 h-3 mr-1" />
            +{growth}%
          </Badge>
        </div>
        <CardTitle className="text-lg group-hover:text-orange-600 dark:group-hover:text-orange-400 transition-colors">
          {result.title}
        </CardTitle>
        <CardDescription>{result.description}</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="flex items-center gap-4 text-sm text-gray-600 dark:text-gray-400">
          <div className="flex items-center gap-1">
            <span>Volume:</span>
            <span className="font-semibold">{volume.toLocaleString()}</span>
          </div>
          {result.metadata?.category && (
            <Badge variant="secondary" className="text-xs">
              {result.metadata.category}
            </Badge>
          )}
        </div>
      </CardContent>
    </Card>
  );
};
