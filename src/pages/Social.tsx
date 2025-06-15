
import React from 'react';
import { Layout } from '../components/Layout';
import { Users, Loader, Repeat } from 'lucide-react';
import { useQuery } from '@tanstack/react-query';
import { pollenAI } from '../services/pollenAI';

const Social = () => {
  const { data: posts, isLoading, refetch, isFetching } = useQuery({
      queryKey: ['social-feed'],
      queryFn: async () => {
          const promises = Array.from({ length: 5 }).map((_, i) => pollenAI.generate(`social post ${i}`, 'social'));
          const responses = await Promise.all(promises);
          return responses.map((res, i) => ({
              id: `post-${Date.now()}-${i}`,
              content: res.content,
              author: 'Pollen AI',
              timestamp: 'Just now'
          }));
      },
      refetchInterval: 20000,
      staleTime: 15000,
  });

  return (
    <Layout>
      <div className="p-6 max-w-3xl mx-auto">
        <div className="mb-8 flex justify-between items-center">
            <div>
                <h1 className="text-3xl font-bold mb-2">Social Intelligence Feed</h1>
                <p className="text-slate-400">AI-curated insights from across the digital sphere</p>
            </div>
            <button onClick={() => refetch()} disabled={isFetching} className="p-2 text-slate-400 hover:text-white hover:bg-slate-700/50 rounded-full transition-colors disabled:opacity-50">
                <Repeat className={`w-5 h-5 ${isFetching ? 'animate-spin' : ''}`} />
            </button>
        </div>

        {isLoading && !posts ? (
             <div className="flex justify-center items-center py-20">
                <Loader className="w-12 h-12 animate-spin text-cyan-400" />
             </div>
        ) : (
            <div className="space-y-6">
                {posts?.map((post) => (
                  <div key={post.id} className="bg-slate-800/50 rounded-xl p-6 border border-slate-700/50 animate-fade-in">
                      <div className="flex items-center space-x-3 mb-4">
                          <div className="w-10 h-10 bg-gradient-to-r from-cyan-500 to-purple-500 rounded-full flex items-center justify-center">
                              <Users className="w-5 h-5 text-white" />
                          </div>
                          <div>
                              <h3 className="font-semibold text-white">{post.author}</h3>
                              <p className="text-xs text-slate-400">{post.timestamp}</p>
                          </div>
                      </div>
                      <p className="text-slate-300 leading-relaxed">{post.content}</p>
                  </div>
                ))}
            </div>
        )}
      </div>
    </Layout>
  );
};

export default Social;
