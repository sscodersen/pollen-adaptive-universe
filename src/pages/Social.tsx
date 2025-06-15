
import React from 'react';
import { Users, Loader, Repeat, Image as ImageIcon } from 'lucide-react';
import { useQuery } from '@tanstack/react-query';
import { pollenAI } from '../services/pollenAI';
import { FeedHeader } from '../components/feed/FeedHeader';
import { ActivityPost } from '../components/feed/ActivityPost';

interface SocialPost {
    id: string;
    content: string;
    author: string;
    timestamp: string;
    image_url?: string;
}

const parseSocialContent = (content: string, id: string): SocialPost => {
    // Simple parser to extract potential image and text
    const hasImage = content.includes("[IMAGE]");
    return {
        id,
        content: content.replace("[IMAGE]", "").trim(),
        author: 'Pollen AI',
        timestamp: 'Just now',
        image_url: hasImage ? `https://picsum.photos/seed/${id}/600/400` : undefined
    }
}


const Social = () => {
  const { data: posts, isLoading, refetch, isFetching } = useQuery({
      queryKey: ['social-feed'],
      queryFn: async () => {
          const promises = Array.from({ length: 5 }).map((_, i) => pollenAI.generate(`social post ${i}`, 'social'));
          const responses = await Promise.all(promises);
          return responses.map((res, i) => parseSocialContent(res.content, `post-${Date.now()}-${i}`));
      },
      refetchInterval: 10000, // Fetch new posts every 10 seconds
      staleTime: 5000,
  });

  const AuthorAvatar = () => (
    <div className="w-10 h-10 bg-gradient-to-r from-cyan-500 to-purple-500 rounded-full flex items-center justify-center">
        <Users className="w-5 h-5 text-white" />
    </div>
  );

  return (
    <div className="flex flex-col h-full">
      <FeedHeader title="Social Feed" />
      
      {isLoading && !posts ? (
           <div className="flex-1 flex justify-center items-center">
              <Loader className="w-12 h-12 animate-spin text-cyan-400" />
           </div>
      ) : (
          <div className="flex-1 overflow-y-auto p-4 md:p-6 space-y-4">
              {posts?.map((post) => (
                <ActivityPost 
                    key={post.id}
                    author={{ name: post.author, avatar: <AuthorAvatar /> }}
                    timestamp={post.timestamp}
                >
                    <p>{post.content}</p>
                    {post.image_url && (
                        <div className="mt-3 rounded-lg overflow-hidden border border-slate-700">
                            <img src={post.image_url} alt="Generated Content" className="w-full h-auto object-cover" />
                        </div>
                    )}
                </ActivityPost>
              ))}
              <div className="flex justify-center py-4">
                  <button onClick={() => refetch()} disabled={isFetching} className="flex items-center space-x-2 px-4 py-2 text-slate-400 hover:text-white bg-slate-800/50 hover:bg-slate-700/50 rounded-full transition-colors disabled:opacity-50">
                      <Repeat className={`w-4 h-4 ${isFetching ? 'animate-spin' : ''}`} />
                      <span>{isFetching ? 'Loading...' : 'Load More'}</span>
                  </button>
              </div>
          </div>
      )}
    </div>
  );
};

export default Social;
