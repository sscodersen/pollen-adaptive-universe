import React from 'react';
import { Loader, ExternalLink, Shield } from 'lucide-react';
import { useQuery } from '@tanstack/react-query';
import { pollenAI } from '../services/pollenAI';
import { FeedHeader } from '../components/feed/FeedHeader';
import { ActivityPost } from '../components/feed/ActivityPost';

interface NewsArticle {
    id: string;
    title: string;
    source: string;
    snippet: string;
    url: string;
    bias: string;
    timestamp: string;
}

const parseNewsContent = (content: string, id: string): NewsArticle => {
    const sentences = content.split('. ');
    const title = sentences[0] || 'AI Generated News';
    const snippet = sentences.slice(1).join('. ') || 'No snippet available.';
    return {
        id,
        title,
        source: 'Pollen AI News Engine',
        snippet,
        url: '#',
        bias: 'neutral',
        timestamp: 'Just now'
    }
}

const Search = () => {
  const { data: newsItems, isLoading } = useQuery({
      queryKey: ['news-feed'],
      queryFn: async () => {
          const promises = Array.from({ length: 5 }).map((_, i) => pollenAI.generate(`news article ${i}`, 'news'));
          const responses = await Promise.all(promises);
          return responses.map((res, i) => parseNewsContent(res.content, `news-${Date.now()}-${i}`));
      },
      refetchInterval: 15000,
      staleTime: 10000,
  });

  const NewsAvatar = () => (
    <div className="w-10 h-10 bg-gradient-to-r from-green-500 to-emerald-500 rounded-full flex items-center justify-center">
        <Shield className="w-5 h-5 text-white" />
    </div>
  );

  return (
    <div className="flex flex-col h-full">
      <FeedHeader title="News Intelligence" />
      
      {isLoading && !newsItems ? (
           <div className="flex-1 flex justify-center items-center">
              <Loader className="w-12 h-12 animate-spin text-cyan-400" />
           </div>
      ) : (
          <div className="flex-1 overflow-y-auto p-4 md:p-6 space-y-4">
              {newsItems?.map((result) => (
                <ActivityPost 
                    key={result.id}
                    author={{ name: result.source, avatar: <NewsAvatar /> }}
                    timestamp={result.timestamp}
                >
                    <h3 className="text-lg font-medium text-white hover:text-cyan-300 cursor-pointer flex items-center justify-between">
                        {result.title}
                        <ExternalLink className="w-4 h-4 text-slate-400" />
                    </h3>
                    <p>{result.snippet}</p>
                    <div className="text-xs text-green-400 flex items-center space-x-1 mt-2">
                        <Shield className="w-3 h-3" />
                        <span>Verified Neutral</span>
                    </div>
                </ActivityPost>
              ))}
          </div>
      )}
    </div>
  );
};

export default Search;
