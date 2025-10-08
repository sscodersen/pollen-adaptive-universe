import { Clock, Sun, Search, TrendingUp, Flame } from "lucide-react";
import { BottomNav } from "./BottomNav";

interface FeedProps {
  onNavigate: (screen: 'feed' | 'explore' | 'shop') => void;
}

export function Feed({ onNavigate }: FeedProps) {
  return (
    <div className="relative min-h-screen pb-32">
      {/* Header */}
      <div className="p-4 sm:p-6 pt-6 sm:pt-8">
        <div className="flex items-start justify-between mb-6">
          <div className="flex items-center gap-2 sm:gap-3">
            <div className="w-10 h-10 sm:w-12 sm:h-12 rounded-full bg-gradient-to-br from-purple-400 to-pink-400 flex items-center justify-center text-white font-semibold text-base sm:text-lg">
              J
            </div>
            <div className="flex items-center gap-2 glass-card px-3 sm:px-4 py-1.5 sm:py-2">
              <Clock className="w-4 h-4 text-gray-600" />
              <span className="text-xl sm:text-2xl font-semibold">Hey Jane,</span>
            </div>
          </div>
          <div className="glass-card px-3 sm:px-4 py-1.5 sm:py-2 flex items-center gap-2">
            <Sun className="w-4 h-4 sm:w-5 sm:h-5 text-yellow-500" />
            <span className="text-base sm:text-lg font-medium">11°C</span>
          </div>
        </div>
        
        <h2 className="text-lg sm:text-xl font-medium text-gray-800 mb-4">Welcome back!</h2>
        
        {/* Search Bar */}
        <button 
          onClick={() => onNavigate('explore')}
          className="w-full glass-card p-4 rounded-2xl mb-6 text-left flex items-center gap-3"
        >
          <Search className="w-5 h-5 text-gray-400" />
          <span className="text-gray-500">Ask Pollen anything...</span>
        </button>

        {/* Feed Categories */}
        <div className="flex gap-2 mb-6 overflow-x-auto scrollbar-thin pb-2">
          <button className="px-6 py-2 bg-white/90 rounded-full text-sm font-medium whitespace-nowrap flex items-center gap-2">
            <Flame className="w-4 h-4" />
            All Posts
          </button>
          <button className="px-6 py-2 bg-white/50 rounded-full text-sm font-medium text-gray-600 whitespace-nowrap flex items-center gap-2">
            <TrendingUp className="w-4 h-4" />
            Trending
          </button>
          <button className="px-6 py-2 bg-white/50 rounded-full text-sm font-medium text-gray-600 whitespace-nowrap">
            High Impact
          </button>
        </div>

        {/* Feed Content */}
        <div className="space-y-4">
          {/* Sister's Birthday Card */}
          <div className="gradient-card-pink p-4 sm:p-6 shadow-lg">
            <div className="flex items-start justify-between mb-4">
              <div className="flex items-center gap-2">
                <div className="w-10 h-10 rounded-full bg-black text-white flex items-center justify-center text-sm font-medium">
                  12 min
                </div>
                <div>
                  <h3 className="font-semibold text-lg">Sister</h3>
                  <p className="text-sm text-gray-600">Birthday Party Check</p>
                </div>
              </div>
            </div>
            
            <div className="flex gap-3 mb-4">
              <div className="flex-1 bg-white/60 rounded-2xl p-3 text-center">
                <span className="text-sm font-medium">Gift</span>
                <p className="text-xs text-gray-600 mt-1">Not started</p>
              </div>
              <div className="flex-1 bg-white/60 rounded-2xl p-3 text-center">
                <span className="text-sm font-medium">Music</span>
                <p className="text-xs text-gray-600 mt-1">Not started</p>
              </div>
              <div className="flex-1 bg-white/60 rounded-2xl p-3 text-center">
                <span className="text-sm font-medium">Restaurant</span>
                <p className="text-xs text-gray-600 mt-1">Not started</p>
              </div>
            </div>
            
            <div className="glass-card p-3 text-center">
              <p className="text-sm text-gray-600">What do you want?</p>
            </div>
          </div>

          {/* Travel Card */}
          <div className="gradient-card-blue p-4 sm:p-6 shadow-lg">
            <div className="flex items-center justify-between mb-3">
              <h3 className="font-semibold text-lg">Travel</h3>
              <span className="bg-purple-500 text-white text-xs px-3 py-1 rounded-full font-medium">+2</span>
            </div>
            <div className="flex gap-2">
              <div className="w-24 h-24 rounded-2xl overflow-hidden">
                <img 
                  src="https://images.unsplash.com/photo-1501594907352-04cda38ebc29?w=200&h=200&fit=crop" 
                  alt="Travel destination" 
                  className="w-full h-full object-cover"
                />
              </div>
              <div className="w-24 h-24 rounded-2xl overflow-hidden">
                <img 
                  src="https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=200&h=200&fit=crop" 
                  alt="Mountain landscape" 
                  className="w-full h-full object-cover"
                />
              </div>
            </div>
          </div>

          {/* Recently Added Content */}
          <div className="gradient-card-purple p-4 sm:p-6 shadow-lg">
            <h3 className="font-semibold text-lg mb-3">Recently added Content</h3>
            <div className="space-y-3">
              <div className="flex gap-3 items-center">
                <div className="w-16 h-16 rounded-xl overflow-hidden">
                  <img 
                    src="https://images.unsplash.com/photo-1546069901-ba9599a7e63c?w=100&h=100&fit=crop" 
                    alt="Food" 
                    className="w-full h-full object-cover"
                  />
                </div>
                <div className="flex-1">
                  <p className="font-medium">1 min ago</p>
                </div>
              </div>
              <div className="flex gap-3 items-center">
                <div className="w-16 h-16 rounded-xl overflow-hidden">
                  <img 
                    src="https://images.unsplash.com/photo-1494976388531-d1058494cdd8?w=100&h=100&fit=crop" 
                    alt="Car" 
                    className="w-full h-full object-cover"
                  />
                </div>
                <div className="flex-1">
                  <p className="font-medium">Yesterday</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
      
      <BottomNav currentScreen="feed" onNavigate={onNavigate} />
    </div>
  );
}
