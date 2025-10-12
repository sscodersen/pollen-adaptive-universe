import React, { memo, useState } from 'react';
import { Eye, TrendingUp, Award, Zap, Users, Sparkles, Star, Clock, Target, ChevronUp, ChevronDown } from 'lucide-react';
import { SocialContent } from '../../services/unifiedContentEngine';
import { votingService } from '../../services/votingService';

interface PostCardProps {
  post: SocialContent;
  onPostClick?: (post: SocialContent) => void;
  onVoteChange?: (postId: string, votes: any) => void;
}

const getImpactColor = (impact: string) => {
  switch (impact) {
    case 'critical': return 'bg-red-500/20 text-red-300 border-red-500/30';
    case 'high': return 'bg-orange-500/20 text-orange-300 border-orange-500/30';
    case 'medium': return 'bg-yellow-500/20 text-yellow-300 border-yellow-500/30';
    default: return 'bg-gray-500/20 text-gray-300 border-gray-500/30';
  }
};

const getRankBadge = (rank: number) => {
  if (rank >= 95) return 'bg-gradient-to-r from-yellow-500 to-orange-500 text-white';
  if (rank >= 90) return 'bg-gradient-to-r from-purple-500 to-pink-500 text-white';
  if (rank >= 80) return 'bg-gradient-to-r from-blue-500 to-cyan-500 text-white';
  return 'bg-gradient-to-r from-gray-600 to-gray-500 text-white';
};

export const PostCard = memo(({ post, onPostClick, onVoteChange }: PostCardProps) => {
  const [localVotes, setLocalVotes] = useState(post.votes || { upvotes: 0, downvotes: 0, score: 0, userVote: null });

  const handleClick = () => {
    onPostClick?.(post);
  };

  const handleVote = async (e: React.MouseEvent, voteType: 'up' | 'down') => {
    e.stopPropagation();
    
    // Use localVotes (current state) instead of post.votes (stale prop)
    // localVotes already includes user's vote (from enrichPostsWithVotes or previous vote)
    // We need to get the raw server counts by removing user's existing vote
    const currentUserVote = votingService.getUserVote(post.id);
    
    const serverVotes = {
      upvotes: localVotes.upvotes - (currentUserVote === 'up' ? 1 : 0),
      downvotes: localVotes.downvotes - (currentUserVote === 'down' ? 1 : 0),
      score: localVotes.score
    };
    
    const newVotes = await votingService.vote(post.id, voteType, serverVotes);
    const updatedVotes = {
      ...newVotes,
      userVote: votingService.getUserVote(post.id)
    };
    setLocalVotes(updatedVotes);
    onVoteChange?.(post.id, updatedVotes);
  };

  return (
    <div 
      className="bg-card hover:bg-card/80 border border-border rounded-lg p-6 cursor-pointer transition-all duration-200 hover:shadow-lg hover:shadow-primary/10"
      onClick={handleClick}
    >
      {/* User Info */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-3">
          <div className={`w-10 h-10 rounded-full flex items-center justify-center text-white font-semibold text-sm ${post.user?.avatar || 'bg-gradient-to-r from-blue-500 to-purple-500'}`}>
            {post.user?.name?.slice(0, 2).toUpperCase() || 'AI'}
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <span className="font-semibold text-white">{post.user?.name || 'AI Assistant'}</span>
              {post.user?.verified && (
                <Award className="w-4 h-4 text-blue-400" />
              )}
              {post.user?.rank && (
                <span className={`px-2 py-1 rounded-full text-xs font-bold ${getRankBadge(post.user.rank)}`}>
                  #{post.user.rank}
                </span>
              )}
            </div>
            <div className="flex items-center space-x-2 text-sm text-gray-400">
              <span>@{post.user?.username || 'ai_assistant'}</span>
              <span>•</span>
              <span>{post.timestamp}</span>
              {post.readTime && (
                <>
                  <span>•</span>
                  <Clock className="w-3 h-3" />
                  <span>{post.readTime}</span>
                </>
              )}
            </div>
          </div>
        </div>
        
        {post.trending && (
          <div className="flex items-center space-x-1 text-orange-400">
            <TrendingUp className="w-4 h-4" />
            <span className="text-xs font-semibold">TRENDING</span>
          </div>
        )}
      </div>

      {/* Content */}
      <div className="mb-4">
        {post.title && (
          <h3 className="font-bold text-white mb-2 text-lg leading-tight">
            {post.title}
          </h3>
        )}
        <p className="text-gray-300 leading-relaxed">
          {(() => {
            const content = post.description || post.content;
            if (typeof content === 'string') {
              return content;
            } else if (typeof content === 'object' && content !== null && typeof content === 'object') {
              // Handle object content safely
              const obj = content as any;
              if (obj.content && typeof obj.content === 'string') {
                return obj.content;
              } else if (obj.title && typeof obj.title === 'string') {
                return obj.title;
              } else if (obj.summary && typeof obj.summary === 'string') {
                return obj.summary;
              } else {
                return 'Generated content available';
              }
            } else {
              return 'Generated content available';
            }
          })()}
        </p>
      </div>

      {/* Tags */}
      {post.tags && post.tags.length > 0 && (
        <div className="flex flex-wrap gap-2 mb-4">
          {post.tags.slice(0, 4).map((tag, index) => (
            <span
              key={index}
              className="px-2 py-1 bg-primary/10 text-primary text-xs rounded-full border border-primary/20"
            >
              #{tag}
            </span>
          ))}
          {post.tags.length > 4 && (
            <span className="px-2 py-1 bg-muted text-muted-foreground text-xs rounded-full">
              +{post.tags.length - 4} more
            </span>
          )}
        </div>
      )}

      {/* Badges */}
      {post.user?.badges && post.user.badges.length > 0 && (
        <div className="flex flex-wrap gap-2 mb-4">
          {post.user.badges.map((badge, index) => (
            <span
              key={index}
              className="px-2 py-1 bg-purple-500/10 text-purple-300 text-xs rounded-full border border-purple-500/20 flex items-center space-x-1"
            >
              <Sparkles className="w-3 h-3" />
              <span>{badge}</span>
            </span>
          ))}
        </div>
      )}

      {/* Stats & Voting */}
      <div className="flex items-center justify-between text-sm">
        <div className="flex items-center space-x-4 text-gray-400">
          <div className="flex items-center space-x-1">
            <Eye className="w-4 h-4" />
            <span>{post.views?.toLocaleString() || '0'}</span>
          </div>
          <div className="flex items-center space-x-1">
            <Users className="w-4 h-4" />
            <span>{post.engagement?.toLocaleString() || '0'}</span>
          </div>
          {post.significance && (
            <div className="flex items-center space-x-1">
              <Target className="w-4 h-4" />
              <span>{post.significance.toFixed(1)}/10</span>
            </div>
          )}
          {localVotes.score > 0 && (
            <div className="flex items-center space-x-1 text-primary">
              <Star className="w-4 h-4 fill-current" />
              <span className="font-semibold">{localVotes.score.toFixed(1)}</span>
            </div>
          )}
        </div>

        <div className="flex items-center space-x-2">
          {post.impact && (
            <div className={`px-2 py-1 rounded-full text-xs font-medium border ${getImpactColor(post.impact)}`}>
              {post.impact.toUpperCase()}
            </div>
          )}
          
          {/* Voting Buttons */}
          <div className="flex items-center bg-muted/30 rounded-lg border border-border overflow-hidden">
            <button
              onClick={(e) => handleVote(e, 'up')}
              className={`flex items-center space-x-1 px-3 py-1.5 transition-colors ${
                localVotes.userVote === 'up' 
                  ? 'bg-green-500/20 text-green-400' 
                  : 'hover:bg-green-500/10 text-gray-400 hover:text-green-400'
              }`}
              title="Upvote"
            >
              <ChevronUp className="w-4 h-4" />
              <span className="text-xs font-medium">{localVotes.upvotes}</span>
            </button>
            <div className="w-px h-6 bg-border"></div>
            <button
              onClick={(e) => handleVote(e, 'down')}
              className={`flex items-center space-x-1 px-3 py-1.5 transition-colors ${
                localVotes.userVote === 'down' 
                  ? 'bg-red-500/20 text-red-400' 
                  : 'hover:bg-red-500/10 text-gray-400 hover:text-red-400'
              }`}
              title="Downvote"
            >
              <ChevronDown className="w-4 h-4" />
              <span className="text-xs font-medium">{localVotes.downvotes}</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
});

PostCard.displayName = 'PostCard';