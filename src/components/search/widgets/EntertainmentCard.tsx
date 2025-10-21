import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Film } from 'lucide-react';
import type { SearchResult } from '@/types/search';

interface EntertainmentCardProps {
  result: SearchResult;
}

export const EntertainmentCard: React.FC<EntertainmentCardProps> = ({ result }) => {
  return (
    <Card className="group hover:shadow-2xl transition-all duration-300 border-2 hover:border-indigo-300 dark:hover:border-indigo-700 bg-gradient-to-br from-indigo-50 to-violet-50 dark:from-indigo-900/20 dark:to-violet-900/20 backdrop-blur-sm">
      <CardHeader>
        <div className="flex items-start justify-between">
          <Badge className="mb-2 bg-indigo-500">
            <Film className="w-3 h-3 mr-1" />
            Entertainment
          </Badge>
        </div>
        <CardTitle className="text-lg group-hover:text-indigo-600 dark:group-hover:text-indigo-400 transition-colors">
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
