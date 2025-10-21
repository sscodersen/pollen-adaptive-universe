import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Music } from 'lucide-react';
import type { SearchResult } from '@/types/search';

interface MediaCardProps {
  result: SearchResult;
}

export const MediaCard: React.FC<MediaCardProps> = ({ result }) => {
  return (
    <Card className="group hover:shadow-2xl transition-all duration-300 border-2 hover:border-pink-300 dark:hover:border-pink-700 bg-gradient-to-br from-pink-50 to-purple-50 dark:from-pink-900/20 dark:to-purple-900/20 backdrop-blur-sm">
      <CardHeader>
        <div className="flex items-start justify-between">
          <Badge className="mb-2 bg-pink-500">
            <Music className="w-3 h-3 mr-1" />
            Media
          </Badge>
        </div>
        <CardTitle className="text-lg group-hover:text-pink-600 dark:group-hover:text-pink-400 transition-colors">
          {result.title}
        </CardTitle>
        <CardDescription>{result.description}</CardDescription>
      </CardHeader>
      <CardContent>
        <p className="text-sm text-gray-700 dark:text-gray-300 line-clamp-3">
          {result.content}
        </p>
      </CardContent>
    </Card>
  );
};
