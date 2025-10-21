import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Sparkles, Cloud, ThermometerSun } from 'lucide-react';
import type { SearchResult } from '@/types/search';

interface AssistantCardProps {
  result: SearchResult;
}

export const AssistantCard: React.FC<AssistantCardProps> = ({ result }) => {
  const isWeather = result.content?.temperature !== undefined;

  return (
    <Card className="group hover:shadow-2xl transition-all duration-300 border-2 hover:border-yellow-300 dark:hover:border-yellow-700 bg-gradient-to-br from-yellow-50 to-amber-50 dark:from-yellow-900/20 dark:to-amber-900/20 backdrop-blur-sm">
      <CardHeader>
        <div className="flex items-start justify-between">
          <Badge className="mb-2 bg-yellow-500">
            <Sparkles className="w-3 h-3 mr-1" />
            Assistant
          </Badge>
        </div>
        <CardTitle className="text-lg group-hover:text-yellow-600 dark:group-hover:text-yellow-400 transition-colors flex items-center gap-2">
          {isWeather && <Cloud className="w-5 h-5" />}
          {result.title}
        </CardTitle>
        <CardDescription>{result.description}</CardDescription>
      </CardHeader>
      {isWeather && (
        <CardContent>
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <ThermometerSun className="w-5 h-5 text-orange-500" />
              <span className="text-3xl font-bold">{result.content.temperature}°C</span>
            </div>
            <div className="text-sm text-gray-600 dark:text-gray-400">
              <p>{result.content.condition}</p>
              <p className="text-xs">{result.content.location}</p>
            </div>
          </div>
        </CardContent>
      )}
    </Card>
  );
};
