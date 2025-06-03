
import React, { useState } from 'react';
import { Heart, MessageCircle, Share2, MoreHorizontal, Download, Paperclip, Send } from 'lucide-react';

interface Activity {
  id: string;
  type: string;
  user: {
    name: string;
    avatar: string;
    initial: string;
  };
  action: string;
  target: string;
  content?: string;
  image?: string;
  timestamp: string;
  actions?: string[];
  likes?: any[];
  replies?: any[];
  hasNewReply?: boolean;
}

interface ActivityFeedProps {
  activities: Activity[];
  isGenerating: boolean;
}

export const ActivityFeed = ({ activities, isGenerating }: ActivityFeedProps) => {
  const [commentTexts, setCommentTexts] = useState<{[key: string]: string}>({});
  const [likedActivities, setLikedActivities] = useState<Set<string>>(new Set());

  const handleCommentChange = (activityId: string, text: string) => {
    setCommentTexts(prev => ({ ...prev, [activityId]: text }));
  };

  const handleSubmitComment = (activityId: string) => {
    const text = commentTexts[activityId]?.trim();
    if (!text) return;
    
    // Handle comment submission logic here
    setCommentTexts(prev => ({ ...prev, [activityId]: '' }));
  };

  const toggleLike = (activityId: string) => {
    setLikedActivities(prev => {
      const newSet = new Set(prev);
      if (newSet.has(activityId)) {
        newSet.delete(activityId);
      } else {
        newSet.add(activityId);
      }
      return newSet;
    });
  };

  return (
    <div className="space-y-6 p-4">
      {isGenerating && (
        <div className="bg-gray-800 rounded-xl p-4 border border-gray-700">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full flex items-center justify-center">
              <div className="w-2 h-2 bg-white rounded-full animate-pulse"></div>
            </div>
            <div>
              <div className="text-gray-300">Pollen AI is generating new content...</div>
              <div className="text-sm text-gray-500">Learning and creating</div>
            </div>
          </div>
        </div>
      )}

      {activities.map((activity) => (
        <div key={activity.id} className="bg-gray-800 rounded-xl p-6 border border-gray-700">
          {/* Activity Header */}
          <div className="flex items-start justify-between mb-4">
            <div className="flex items-center space-x-3">
              <div className={`w-10 h-10 ${activity.user.avatar} rounded-full flex items-center justify-center text-white font-medium`}>
                {activity.user.initial}
              </div>
              <div>
                <div className="flex items-center space-x-2">
                  <span className="font-medium text-white">{activity.user.name}</span>
                  <span className="text-gray-400">{activity.action}</span>
                  <span className="text-blue-400 font-medium">{activity.target}</span>
                </div>
                <div className="text-sm text-gray-500">{activity.timestamp}</div>
              </div>
            </div>
            <button className="p-2 hover:bg-gray-700 rounded-lg transition-colors">
              <MoreHorizontal className="w-5 h-5 text-gray-400" />
            </button>
          </div>

          {/* Activity Content */}
          {activity.content && (
            <div className="mb-4">
              <p className="text-gray-200 leading-relaxed">{activity.content}</p>
            </div>
          )}

          {/* Activity Image */}
          {activity.image && (
            <div className="mb-4">
              <img
                src={activity.image}
                alt="Generated content"
                className="w-full rounded-lg max-h-96 object-cover"
              />
              {activity.actions && (
                <div className="flex space-x-3 mt-3">
                  {activity.actions.map((action) => (
                    <button
                      key={action}
                      className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                        action === 'Download'
                          ? 'bg-blue-600 hover:bg-blue-700 text-white'
                          : 'bg-gray-700 hover:bg-gray-600 text-gray-200'
                      }`}
                    >
                      {action === 'Download' && <Download className="w-4 h-4 mr-2" />}
                      {action}
                    </button>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Activity Replies */}
          {activity.replies && activity.replies.length > 0 && (
            <div className="mb-4 space-y-3">
              {activity.replies.map((reply, index) => (
                <div key={index} className="flex items-start space-x-3 ml-4 p-3 bg-gray-700/50 rounded-lg">
                  <div className={`w-8 h-8 ${reply.user.avatar} rounded-full flex items-center justify-center text-white text-sm font-medium`}>
                    {reply.user.initial}
                  </div>
                  <div className="flex-1">
                    <div className="flex items-center space-x-2 mb-1">
                      <span className="font-medium text-white text-sm">{reply.user.name}</span>
                      <span className="text-xs text-gray-500">{reply.timestamp}</span>
                    </div>
                    <p className="text-gray-200 text-sm">{reply.content}</p>
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Like Section */}
          {activity.hasNewReply && (
            <div className="mb-4">
              <div className="flex items-center space-x-2 mb-2">
                <div className="flex -space-x-2">
                  {['bg-red-500', 'bg-blue-500', 'bg-green-500', 'bg-purple-500', 'bg-orange-500', 'bg-pink-500', 'bg-yellow-500', 'bg-indigo-500'].slice(0, 8).map((color, i) => (
                    <div key={i} className={`w-6 h-6 ${color} rounded-full border-2 border-gray-800 flex items-center justify-center text-xs text-white font-medium`}>
                      {String.fromCharCode(65 + i)}
                    </div>
                  ))}
                </div>
              </div>
              <p className="text-sm text-gray-400">
                <span className="text-white font-medium">Benjamin</span> and <span className="text-white font-medium">8 others</span> liked your article
              </p>
              <p className="text-xs text-gray-500 mt-1">12h</p>
            </div>
          )}

          {/* Comment Input */}
          <div className="flex items-center space-x-3">
            <div className="w-8 h-8 bg-gray-600 rounded-full flex items-center justify-center text-white text-sm font-medium">
              U
            </div>
            <div className="flex-1 flex items-center space-x-2 bg-gray-700 rounded-lg px-3 py-2">
              <input
                type="text"
                placeholder="Write a comment"
                value={commentTexts[activity.id] || ''}
                onChange={(e) => handleCommentChange(activity.id, e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleSubmitComment(activity.id)}
                className="flex-1 bg-transparent text-white placeholder-gray-400 outline-none"
              />
              <button className="p-1 hover:bg-gray-600 rounded transition-colors">
                <Paperclip className="w-4 h-4 text-gray-400" />
              </button>
              <button
                onClick={() => handleSubmitComment(activity.id)}
                className="px-3 py-1 bg-blue-600 hover:bg-blue-700 text-white rounded text-sm font-medium transition-colors"
              >
                Send
              </button>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
};
