import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Heart } from 'lucide-react';
import type { SearchResult } from '@/types/search';

interface WellnessCardProps {
  result: SearchResult;
}

export const WellnessCard: React.FC<WellnessCardProps> = ({ result }) => {
  const emoji = result.metadata?.emoji || '💚';

  return (
    <Card className="group hover:shadow-2xl transition-all duration-300 border-2 hover:border-green-300 dark:hover:border-green-700 bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 backdrop-blur-sm">
      <CardHeader>
        <div className="flex items-start justify-between">
          <Badge className="mb-2 bg-green-500">
            <Heart className="w-3 h-3 mr-1" />
            Wellness
          </Badge>
          <span className="text-2xl">{emoji}</span>
        </div>
        <CardTitle className="text-lg group-hover:text-green-600 dark:group-hover:text-green-400 transition-colors">
          {result.title}
        </CardTitle>
        <CardDescription>{result.description}</CardDescription>
      </CardHeader>
      <CardContent>
        {result.metadata?.category && (
          <Badge variant="secondary" className="mb-2">
            {result.metadata.category}
          </Badge>
        )}
      </CardContent>
    </Card>
  );
};
