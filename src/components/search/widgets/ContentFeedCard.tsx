import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Sparkles } from 'lucide-react';
import type { SearchResult } from '@/types/search';

interface ContentFeedCardProps {
  result: SearchResult;
}

export const ContentFeedCard: React.FC<ContentFeedCardProps> = ({ result }) => {
  return (
    <Card className="group hover:shadow-2xl transition-all duration-300 border-2 hover:border-purple-300 dark:hover:border-purple-700 bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm">
      <CardHeader>
        <div className="flex items-start justify-between">
          <Badge variant="secondary" className="mb-2">
            <Sparkles className="w-3 h-3 mr-1" />
            AI Generated
          </Badge>
        </div>
        <CardTitle className="text-lg group-hover:text-purple-600 dark:group-hover:text-purple-400 transition-colors">
          {result.title}
        </CardTitle>
        <CardDescription>{result.description}</CardDescription>
      </CardHeader>
      <CardContent>
        <p className="text-sm text-gray-700 dark:text-gray-300 line-clamp-4">
          {result.content}
        </p>
      </CardContent>
    </Card>
  );
};
