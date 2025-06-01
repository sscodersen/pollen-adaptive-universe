
import React from 'react';
import { Layout } from '../components/Layout';
import { Clock, MessageCircle, Heart, Share, Download, Sparkles } from 'lucide-react';

const activities = [
  {
    id: 1,
    user: "Pollen LLMX",
    action: "generated a new visual concept",
    time: "2m",
    content: {
      type: "image",
      description: "Minimalist product mockup with gradient overlays",
      image: "/lovable-uploads/f306d6f2-5915-45e7-b800-05894257a2c7.png"
    },
    likes: 12,
    comments: 3
  },
  {
    id: 2,
    user: "Text Engine",
    action: "created marketing copy",
    time: "5m",
    content: {
      type: "text",
      title: "Revolutionary AI Platform Launch",
      preview: "Introducing Pollen LLMX - the self-evolving AI that adapts to your unique needs..."
    },
    likes: 8,
    comments: 2
  },
  {
    id: 3,
    user: "Task Executor",
    action: "automated workflow optimization",
    time: "12m",
    content: {
      type: "task",
      title: "Content Calendar Updated",
      description: "Automatically scheduled 15 posts across platforms based on engagement patterns"
    },
    likes: 15,
    comments: 5
  }
];

const Activity = () => {
  return (
    <Layout>
      <div className="p-6 max-w-4xl mx-auto">
        <div className="mb-8">
          <h1 className="text-3xl font-bold mb-2">Activity Feed</h1>
          <p className="text-slate-400">Real-time updates from your Pollen LLMX ecosystem</p>
        </div>

        <div className="space-y-6">
          {activities.map((activity) => (
            <div key={activity.id} className="bg-slate-800/50 backdrop-blur-xl rounded-xl border border-slate-700/50 p-6 hover:border-slate-600/50 transition-all duration-300">
              <div className="flex items-start space-x-4">
                <div className="w-10 h-10 bg-gradient-to-r from-cyan-400 to-purple-400 rounded-full flex items-center justify-center flex-shrink-0">
                  <Sparkles className="w-5 h-5 text-white" />
                </div>
                
                <div className="flex-1">
                  <div className="flex items-center space-x-2 mb-3">
                    <span className="font-semibold text-cyan-300">{activity.user}</span>
                    <span className="text-slate-400">{activity.action}</span>
                    <span className="text-slate-500">•</span>
                    <span className="text-slate-500 text-sm">{activity.time}</span>
                  </div>
                  
                  {activity.content.type === 'image' && (
                    <div className="bg-slate-700/30 rounded-lg p-4 mb-4">
                      <img
                        src={activity.content.image}
                        alt={activity.content.description}
                        className="w-full h-48 object-cover rounded-lg mb-3"
                      />
                      <p className="text-slate-300">{activity.content.description}</p>
                      <div className="flex space-x-2 mt-3">
                        <button className="bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded-lg text-sm font-medium transition-colors flex items-center space-x-2">
                          <Download className="w-4 h-4" />
                          <span>Download</span>
                        </button>
                        <button className="bg-slate-700 hover:bg-slate-600 px-4 py-2 rounded-lg text-sm font-medium transition-colors">
                          Use Image
                        </button>
                      </div>
                    </div>
                  )}
                  
                  {activity.content.type === 'text' && (
                    <div className="bg-slate-700/30 rounded-lg p-4 mb-4">
                      <h3 className="font-semibold text-lg mb-2">{activity.content.title}</h3>
                      <p className="text-slate-300">{activity.content.preview}</p>
                    </div>
                  )}
                  
                  {activity.content.type === 'task' && (
                    <div className="bg-slate-700/30 rounded-lg p-4 mb-4">
                      <h3 className="font-semibold text-lg mb-2">{activity.content.title}</h3>
                      <p className="text-slate-300">{activity.content.description}</p>
                    </div>
                  )}
                  
                  <div className="flex items-center space-x-6 text-slate-400">
                    <button className="flex items-center space-x-2 hover:text-red-400 transition-colors">
                      <Heart className="w-4 h-4" />
                      <span>{activity.likes}</span>
                    </button>
                    <button className="flex items-center space-x-2 hover:text-blue-400 transition-colors">
                      <MessageCircle className="w-4 h-4" />
                      <span>{activity.comments}</span>
                    </button>
                    <button className="flex items-center space-x-2 hover:text-green-400 transition-colors">
                      <Share className="w-4 h-4" />
                      <span>Share</span>
                    </button>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
        
        <div className="mt-8 bg-gradient-to-r from-cyan-500/10 to-purple-500/10 rounded-xl border border-cyan-500/20 p-6">
          <div className="flex items-center space-x-3 mb-4">
            <div className="w-8 h-8 bg-gradient-to-r from-cyan-400 to-purple-400 rounded-lg flex items-center justify-center">
              <Sparkles className="w-5 h-5 text-white" />
            </div>
            <h3 className="text-xl font-semibold">Pollen Evolution Status</h3>
          </div>
          <p className="text-slate-300 mb-4">
            Your Pollen LLMX model has processed 1,247 interactions today and evolved 8 behavioral patterns. 
            The model is 23% more efficient at understanding your preferences than yesterday.
          </p>
          <div className="bg-slate-700/50 rounded-lg p-3">
            <div className="flex justify-between text-sm mb-2">
              <span>Learning Progress</span>
              <span>78%</span>
            </div>
            <div className="w-full bg-slate-600 rounded-full h-2">
              <div className="bg-gradient-to-r from-cyan-400 to-purple-400 h-2 rounded-full w-3/4"></div>
            </div>
          </div>
        </div>
      </div>
    </Layout>
  );
};

export default Activity;
