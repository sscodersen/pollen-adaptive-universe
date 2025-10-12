import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { 
  Calendar, Clock, TrendingUp, BookOpen, Tag, 
  ChevronRight, Sparkles, Mail 
} from 'lucide-react';
import { newsDigestService, NewsDigest } from '@/services/newsDigest';
import { NewsContent } from '@/services/unifiedContentEngine';

interface NewsDigestPanelProps {
  articles: NewsContent[];
  onClose?: () => void;
}

export const NewsDigestPanel: React.FC<NewsDigestPanelProps> = ({ articles, onClose }) => {
  const [digest, setDigest] = useState<NewsDigest | null>(null);
  const [loading, setLoading] = useState(true);
  const [frequency, setFrequency] = useState<'daily' | 'weekly'>('daily');

  useEffect(() => {
    const prefs = newsDigestService.getDigestPreferences();
    setFrequency(prefs.frequency);
    generateDigest(prefs.frequency);
  }, [articles]);

  const generateDigest = async (freq: 'daily' | 'weekly') => {
    setLoading(true);
    try {
      const generated = await newsDigestService.generateDigest(articles, freq);
      setDigest(generated);
    } catch (error) {
      console.error('Failed to generate digest:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleFrequencyChange = async (freq: 'daily' | 'weekly') => {
    setFrequency(freq);
    newsDigestService.saveDigestPreferences(freq, digest?.personalizedCategories || []);
    await generateDigest(freq);
  };

  if (loading) {
    return (
      <Card className="animate-pulse">
        <CardHeader>
          <div className="h-6 bg-gray-300 dark:bg-gray-700 rounded w-1/2 mb-2" />
          <div className="h-4 bg-gray-300 dark:bg-gray-700 rounded w-3/4" />
        </CardHeader>
      </Card>
    );
  }

  if (!digest) return null;

  return (
    <div className="space-y-6">
      <Card className="border-2 border-blue-500/20 bg-gradient-to-br from-blue-50/50 to-purple-50/50 dark:from-blue-950/20 dark:to-purple-950/20">
        <CardHeader>
          <div className="flex items-start justify-between">
            <div className="flex items-center gap-3">
              <div className="p-3 bg-gradient-to-r from-blue-500 to-purple-500 rounded-xl">
                <Mail className="w-6 h-6 text-white" />
              </div>
              <div>
                <CardTitle className="text-2xl">News Digest</CardTitle>
                <CardDescription>
                  AI-curated summary of your personalized news
                </CardDescription>
              </div>
            </div>
            {onClose && (
              <Button variant="ghost" size="sm" onClick={onClose}>
                Close
              </Button>
            )}
          </div>

          <div className="flex gap-2 mt-4">
            <Button
              variant={frequency === 'daily' ? 'default' : 'outline'}
              size="sm"
              onClick={() => handleFrequencyChange('daily')}
            >
              <Calendar className="w-4 h-4 mr-2" />
              Daily
            </Button>
            <Button
              variant={frequency === 'weekly' ? 'default' : 'outline'}
              size="sm"
              onClick={() => handleFrequencyChange('weekly')}
            >
              <Calendar className="w-4 h-4 mr-2" />
              Weekly
            </Button>
          </div>
        </CardHeader>

        <CardContent className="space-y-6">
          <div className="flex items-center gap-4 text-sm text-gray-600 dark:text-gray-400">
            <div className="flex items-center gap-2">
              <Calendar className="w-4 h-4" />
              {new Date(digest.date).toLocaleDateString()}
            </div>
            <div className="flex items-center gap-2">
              <BookOpen className="w-4 h-4" />
              {digest.totalArticles} articles
            </div>
            <div className="flex items-center gap-2">
              <Clock className="w-4 h-4" />
              {digest.readTime} read
            </div>
          </div>

          <div className="space-y-4">
            <h3 className="text-lg font-semibold flex items-center gap-2">
              <TrendingUp className="w-5 h-5 text-blue-600" />
              Top Stories
            </h3>
            <div className="space-y-3">
              {digest.topStories.map(story => (
                <div
                  key={story.id}
                  className="p-4 bg-white/60 dark:bg-gray-800/60 rounded-xl border border-gray-200 dark:border-gray-700 hover:shadow-lg transition-all group cursor-pointer"
                >
                  <div className="flex items-start justify-between mb-2">
                    <Badge variant="secondary" className="text-xs">
                      {story.category}
                    </Badge>
                    <ChevronRight className="w-4 h-4 text-gray-400 group-hover:translate-x-1 transition-transform" />
                  </div>
                  <h4 className="font-semibold text-gray-900 dark:text-white mb-1 line-clamp-2">
                    {story.title}
                  </h4>
                  <p className="text-sm text-gray-600 dark:text-gray-400 line-clamp-2">
                    {story.snippet}
                  </p>
                </div>
              ))}
            </div>
          </div>

          <div className="space-y-4">
            <h3 className="text-lg font-semibold flex items-center gap-2">
              <Sparkles className="w-5 h-5 text-purple-600" />
              Category Summaries
            </h3>
            <div className="space-y-4">
              {digest.summaries.map(summary => (
                <div
                  key={summary.category}
                  className="p-4 bg-white/60 dark:bg-gray-800/60 rounded-xl border border-gray-200 dark:border-gray-700"
                >
                  <div className="flex items-center justify-between mb-3">
                    <h4 className="font-semibold text-gray-900 dark:text-white capitalize">
                      {summary.category}
                    </h4>
                    <Badge variant="outline" className="text-xs">
                      {summary.articleCount} articles
                    </Badge>
                  </div>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
                    {summary.summary}
                  </p>
                  <div className="flex items-center gap-2 flex-wrap">
                    <Tag className="w-3 h-3 text-gray-400" />
                    {summary.topKeywords.map(keyword => (
                      <span
                        key={keyword}
                        className="px-2 py-1 bg-gray-100 dark:bg-gray-700 rounded text-xs text-gray-600 dark:text-gray-300"
                      >
                        {keyword}
                      </span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};
