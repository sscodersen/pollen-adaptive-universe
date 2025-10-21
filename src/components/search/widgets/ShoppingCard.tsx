import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ShoppingBag } from 'lucide-react';
import type { SearchResult } from '@/types/search';

interface ShoppingCardProps {
  result: SearchResult;
}

export const ShoppingCard: React.FC<ShoppingCardProps> = ({ result }) => {
  return (
    <Card className="group hover:shadow-2xl transition-all duration-300 border-2 hover:border-blue-300 dark:hover:border-blue-700 bg-gradient-to-br from-blue-50 to-cyan-50 dark:from-blue-900/20 dark:to-cyan-900/20 backdrop-blur-sm">
      <CardHeader>
        <div className="flex items-start justify-between">
          <Badge className="mb-2 bg-blue-500">
            <ShoppingBag className="w-3 h-3 mr-1" />
            Shopping
          </Badge>
        </div>
        <CardTitle className="text-lg group-hover:text-blue-600 dark:group-hover:text-blue-400 transition-colors">
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
