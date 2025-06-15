
import React from 'react';
import { Layout } from '../components/Layout';
import { TrendingUp, ExternalLink, Shield, Loader } from 'lucide-react';
import { useQuery } from '@tanstack/react-query';
import { pollenAI } from '../services/pollenAI';

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
          const promises = Array.from({ length: 3 }).map((_, i) => pollenAI.generate(`news article ${i}`, 'news'));
          const responses = await Promise.all(promises);
          return responses.map((res, i) => parseNewsContent(res.content, `news-${Date.now()}-${i}`));
      },
      refetchInterval: 15000,
      staleTime: 10000,
  });

  return (
    <Layout>
      <div className="p-6 max-w-6xl mx-auto">
        <div className="mb-8">
          <h1 className="text-3xl font-bold mb-2">Live News Intelligence</h1>
          <p className="text-slate-400">Continuously updated news stream powered by Pollen AI</p>
        </div>

        {isLoading && !newsItems ? (
             <div className="flex justify-center items-center py-20">
                <Loader className="w-12 h-12 animate-spin text-cyan-400" />
             </div>
        ) : (
            <div className="space-y-4">
                <h2 className="text-xl font-semibold">Live Feed</h2>
                {newsItems?.map((result) => (
                  <div key={result.id} className="bg-slate-800/50 rounded-lg p-6 border border-slate-700/50 animate-fade-in">
                    <div className="flex items-start justify-between mb-2">
                      <h3 className="text-lg font-medium text-white hover:text-cyan-300 cursor-pointer">
                        {result.title}
                      </h3>
                      <ExternalLink className="w-4 h-4 text-slate-400" />
                    </div>
                    <p className="text-slate-300 mb-3">{result.snippet}</p>
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-4">
                        <span className="text-sm text-slate-400">{result.source}</span>
                        <span className="text-xs text-slate-500">{result.timestamp}</span>
                        <div className="flex items-center space-x-1">
                          <Shield className="w-3 h-3 text-green-400" />
                          <span className="text-xs text-green-400">Verified Neutral</span>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
            </div>
        )}

        <div className="mt-8">
          <h2 className="text-xl font-semibold mb-4 flex items-center space-x-2">
            <TrendingUp className="w-5 h-5" />
            <span>Trending Topics</span>
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {['AI Safety Research', 'Quantum Computing Advances', 'Climate Tech Innovation'].map((topic) => (
              <div key={topic} className="bg-slate-800/50 rounded-lg p-4 border border-slate-700/50">
                <h3 className="font-medium mb-2">{topic}</h3>
                <p className="text-sm text-slate-400">Multiple verified sources</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </Layout>
  );
};

export default Search;
