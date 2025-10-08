import { ArrowLeft, MapPin, Car, Bike, Footprints, Search as SearchIcon } from "lucide-react";
import { BottomNav } from "./BottomNav";

interface ExploreProps {
  onNavigate: (screen: 'feed' | 'explore' | 'shop') => void;
}

export function Explore({ onNavigate }: ExploreProps) {
  return (
    <div className="relative min-h-screen pb-32">
      {/* Header */}
      <div className="p-4 sm:p-6 pt-6 sm:pt-8">
        <div className="flex items-center gap-3 mb-6">
          <button onClick={() => onNavigate('feed')} className="w-10 h-10 rounded-full bg-white/80 backdrop-blur-sm flex items-center justify-center">
            <ArrowLeft className="w-5 h-5" />
          </button>
          <div className="flex-1 glass-card px-4 py-3 rounded-2xl flex items-center gap-3">
            <SearchIcon className="w-5 h-5 text-gray-400" />
            <input 
              type="text" 
              placeholder="What's interesting going on in SF?"
              className="flex-1 bg-transparent outline-none text-sm text-gray-600"
            />
          </div>
        </div>

        {/* Main Content Card */}
        <div className="glass-card p-6 shadow-lg mb-4">
          <h1 className="text-3xl font-semibold mb-2">What to try<br/>in San Francisco?</h1>
          <p className="text-gray-600 mb-4">San Francisco is known for its diverse and iconic food scene, offering a wide range of delicious dishes.</p>
          
          <div className="flex items-center gap-2 text-sm text-gray-500 mb-4">
            <span>W</span>
            <span>Wikipedia</span>
          </div>
          
          <div className="mb-4">
            <p className="text-sm text-gray-600 mb-2">Places to visit</p>
            <h3 className="font-semibold text-lg mb-3">Golden Gate Bridge</h3>
            <div className="flex gap-2">
              <div className="w-28 h-20 rounded-xl overflow-hidden">
                <img 
                  src="https://images.unsplash.com/photo-1501594907352-04cda38ebc29?w=200&h=150&fit=crop" 
                  alt="Golden Gate Bridge" 
                  className="w-full h-full object-cover"
                />
              </div>
              <div className="w-28 h-20 rounded-xl overflow-hidden">
                <img 
                  src="https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=200&h=150&fit=crop" 
                  alt="Bridge at night" 
                  className="w-full h-full object-cover"
                />
              </div>
              <div className="relative w-28 h-20 rounded-xl overflow-hidden">
                <img 
                  src="https://images.unsplash.com/photo-1449034446853-66c86144b0ad?w=200&h=150&fit=crop" 
                  alt="More images" 
                  className="w-full h-full object-cover"
                />
                <div className="absolute inset-0 bg-black/40 flex items-center justify-center">
                  <span className="text-white font-semibold text-lg">+16</span>
                </div>
              </div>
            </div>
          </div>
          
          <div className="flex items-center gap-2 text-sm text-gray-500">
            <span>G</span>
            <span>Google</span>
          </div>
        </div>

        {/* Transportation Options */}
        <div className="flex gap-3 mb-4">
          <button className="flex-1 glass-card p-3 rounded-2xl flex items-center justify-center gap-2">
            <Car className="w-5 h-5" />
            <span className="text-sm font-medium">Driving</span>
          </button>
          <button className="flex-1 glass-card p-3 rounded-2xl flex items-center justify-center gap-2">
            <Bike className="w-5 h-5" />
            <span className="text-sm font-medium">Biking</span>
          </button>
          <button className="flex-1 glass-card p-3 rounded-2xl flex items-center justify-center gap-2">
            <Footprints className="w-5 h-5" />
            <span className="text-sm font-medium">Walking</span>
          </button>
        </div>

        {/* Location Card */}
        <div className="glass-card p-4 rounded-2xl">
          <div className="flex items-start gap-3 mb-3">
            <MapPin className="w-5 h-5 text-blue-500" />
            <div className="flex-1">
              <p className="text-sm text-gray-600">Your location</p>
              <p className="font-medium">900 North Point St</p>
            </div>
          </div>
          <div className="ml-8">
            <p className="text-sm text-purple-600 font-medium mb-2">10 min</p>
            <div className="space-y-2">
              <div>
                <p className="font-medium">Golden Gate Bridge</p>
                <p className="text-sm text-gray-500">Golden Gate Brg</p>
              </div>
            </div>
          </div>
        </div>
      </div>
      
      <BottomNav currentScreen="explore" onNavigate={onNavigate} />
    </div>
  );
}
