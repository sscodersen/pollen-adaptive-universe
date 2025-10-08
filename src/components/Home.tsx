import { Clock, Sun, Plus } from "lucide-react";

interface HomeProps {
  onNavigate: (screen: 'home' | 'search' | 'collections' | 'shopping') => void;
}

export function Home({ onNavigate }: HomeProps) {
  return (
    <div className="relative min-h-screen pb-20">
      {/* Header with greeting and user profile */}
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
          onClick={() => onNavigate('search')}
          className="w-full glass-card p-4 rounded-2xl mb-6 text-left flex items-center gap-3"
        >
          <svg className="w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
          </svg>
          <span className="text-gray-500">Ask Pollen anything...</span>
        </button>

        {/* Activity Cards Grid */}
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
        </div>
      </div>
      
      {/* Bottom Navigation */}
      <div className="fixed bottom-0 left-0 right-0 glass-card mx-4 mb-4 rounded-3xl p-4">
        <div className="flex items-center justify-around">
          <button className="flex flex-col items-center gap-1 text-purple-600">
            <div className="w-10 h-10 rounded-full bg-purple-100 flex items-center justify-center">
              <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                <path d="M10.707 2.293a1 1 0 00-1.414 0l-7 7a1 1 0 001.414 1.414L4 10.414V17a1 1 0 001 1h2a1 1 0 001-1v-2a1 1 0 011-1h2a1 1 0 011 1v2a1 1 0 001 1h2a1 1 0 001-1v-6.586l.293.293a1 1 0 001.414-1.414l-7-7z" />
              </svg>
            </div>
          </button>
          
          <button onClick={() => onNavigate('search')} className="flex flex-col items-center gap-1 text-gray-400">
            <div className="w-10 h-10 rounded-full flex items-center justify-center">
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
            </div>
          </button>
          
          <button onClick={() => onNavigate('collections')} className="flex flex-col items-center gap-1 text-gray-400">
            <div className="w-10 h-10 rounded-full flex items-center justify-center">
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 11V7a4 4 0 00-8 0v4M5 9h14l1 12H4L5 9z" />
              </svg>
            </div>
          </button>
        </div>
      </div>
    </div>
  );
}
