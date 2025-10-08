import { Home, Compass, ShoppingBag, Users } from "lucide-react";

interface BottomNavProps {
  currentScreen: 'feed' | 'explore' | 'shop' | 'community';
  onNavigate: (screen: 'feed' | 'explore' | 'shop' | 'community') => void;
}

export function BottomNav({ currentScreen, onNavigate }: BottomNavProps) {
  return (
    <div className="fixed bottom-0 left-0 right-0 z-50">
      <div className="glass-card mx-4 mb-4 rounded-3xl p-4 shadow-lg">
        <div className="flex items-center justify-around">
          <button 
            onClick={() => onNavigate('feed')} 
            className={`flex flex-col items-center gap-1 transition-colors ${
              currentScreen === 'feed' ? 'text-purple-600' : 'text-gray-400'
            }`}
          >
            <div className={`w-12 h-12 rounded-2xl flex items-center justify-center transition-all ${
              currentScreen === 'feed' ? 'bg-purple-100' : 'bg-transparent'
            }`}>
              <Home className="w-6 h-6" />
            </div>
            <span className="text-xs font-medium">Feed</span>
          </button>
          
          <button 
            onClick={() => onNavigate('explore')} 
            className={`flex flex-col items-center gap-1 transition-colors ${
              currentScreen === 'explore' ? 'text-purple-600' : 'text-gray-400'
            }`}
          >
            <div className={`w-12 h-12 rounded-2xl flex items-center justify-center transition-all ${
              currentScreen === 'explore' ? 'bg-purple-100' : 'bg-transparent'
            }`}>
              <Compass className="w-6 h-6" />
            </div>
            <span className="text-xs font-medium">Explore</span>
          </button>
          
          <button 
            onClick={() => onNavigate('community')} 
            className={`flex flex-col items-center gap-1 transition-colors ${
              currentScreen === 'community' ? 'text-purple-600' : 'text-gray-400'
            }`}
          >
            <div className={`w-12 h-12 rounded-2xl flex items-center justify-center transition-all ${
              currentScreen === 'community' ? 'bg-purple-100' : 'bg-transparent'
            }`}>
              <Users className="w-6 h-6" />
            </div>
            <span className="text-xs font-medium">Community</span>
          </button>
          
          <button 
            onClick={() => onNavigate('shop')} 
            className={`flex flex-col items-center gap-1 transition-colors ${
              currentScreen === 'shop' ? 'text-purple-600' : 'text-gray-400'
            }`}
          >
            <div className={`w-12 h-12 rounded-2xl flex items-center justify-center transition-all ${
              currentScreen === 'shop' ? 'bg-purple-100' : 'bg-transparent'
            }`}>
              <ShoppingBag className="w-6 h-6" />
            </div>
            <span className="text-xs font-medium">Smart Shop</span>
          </button>
        </div>
      </div>
    </div>
  );
}
