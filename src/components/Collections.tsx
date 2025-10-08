import { ArrowLeft, Search } from "lucide-react";

interface CollectionsProps {
  onNavigate: (screen: 'home' | 'search' | 'collections' | 'shopping') => void;
}

export function Collections({ onNavigate }: CollectionsProps) {
  return (
    <div className="relative min-h-screen pb-20">
      {/* Header */}
      <div className="p-6 pt-8">
        <div className="flex items-center justify-between mb-6">
          <button onClick={() => onNavigate('home')} className="w-10 h-10 rounded-full bg-white/80 backdrop-blur-sm flex items-center justify-center">
            <ArrowLeft className="w-5 h-5" />
          </button>
          <h1 className="text-xl font-semibold">My Collections</h1>
          <button className="w-10 h-10 rounded-full bg-white/80 backdrop-blur-sm flex items-center justify-center">
            <Search className="w-5 h-5" />
          </button>
        </div>

        {/* Tabs */}
        <div className="flex gap-2 mb-6">
          <button className="px-6 py-2 bg-white/90 rounded-full text-sm font-medium">All collections</button>
          <button className="px-6 py-2 bg-white/50 rounded-full text-sm font-medium text-gray-600">Personal</button>
          <button className="px-6 py-2 bg-white/50 rounded-full text-sm font-medium text-gray-600">Suggested</button>
        </div>

        {/* Collections Grid */}
        <div className="grid grid-cols-2 gap-4">
          {/* Travel Collection */}
          <div className="gradient-card-blue p-4 shadow-md">
            <h3 className="font-semibold mb-1">Travel</h3>
            <p className="text-sm text-gray-600 mb-3">4 items</p>
            <div className="flex gap-2">
              <div className="w-16 h-16 rounded-xl overflow-hidden">
                <img 
                  src="https://images.unsplash.com/photo-1501594907352-04cda38ebc29?w=100&h=100&fit=crop" 
                  alt="Travel" 
                  className="w-full h-full object-cover"
                />
              </div>
              <div className="w-16 h-16 rounded-xl overflow-hidden">
                <img 
                  src="https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=100&h=100&fit=crop" 
                  alt="Travel" 
                  className="w-full h-full object-cover"
                />
              </div>
            </div>
          </div>

          {/* Food Collection */}
          <div className="gradient-card-pink p-4 shadow-md">
            <h3 className="font-semibold mb-1">Food</h3>
            <p className="text-sm text-gray-600 mb-3">6 items</p>
            <div className="flex gap-2">
              <div className="w-16 h-16 rounded-xl overflow-hidden">
                <img 
                  src="https://images.unsplash.com/photo-1546069901-ba9599a7e63c?w=100&h=100&fit=crop" 
                  alt="Food" 
                  className="w-full h-full object-cover"
                />
              </div>
              <div className="w-16 h-16 rounded-xl overflow-hidden">
                <img 
                  src="https://images.unsplash.com/photo-1565299624946-b28f40a0ae38?w=100&h=100&fit=crop" 
                  alt="Food" 
                  className="w-full h-full object-cover"
                />
              </div>
            </div>
          </div>

          {/* Goals Collection */}
          <div className="gradient-card-pink p-4 shadow-md">
            <h3 className="font-semibold mb-1">Goals</h3>
            <p className="text-sm text-gray-600 mb-3">3 left</p>
            <div className="flex gap-3 items-center">
              <div className="relative w-16 h-16">
                <svg className="transform -rotate-90" width="64" height="64">
                  <circle cx="32" cy="32" r="28" fill="none" stroke="#e5e7eb" strokeWidth="4"/>
                  <circle cx="32" cy="32" r="28" fill="none" stroke="#a855f7" strokeWidth="4" strokeDasharray="176" strokeDashoffset="44"/>
                </svg>
                <span className="absolute inset-0 flex items-center justify-center text-xs font-medium">75%</span>
              </div>
              <div className="relative w-16 h-16">
                <svg className="transform -rotate-90" width="64" height="64">
                  <circle cx="32" cy="32" r="28" fill="none" stroke="#e5e7eb" strokeWidth="4"/>
                  <circle cx="32" cy="32" r="28" fill="none" stroke="#3b82f6" strokeWidth="4" strokeDasharray="176" strokeDashoffset="70"/>
                </svg>
                <span className="absolute inset-0 flex items-center justify-center text-xs font-medium">60%</span>
              </div>
            </div>
          </div>

          {/* Events Collection */}
          <div className="gradient-card-blue p-4 shadow-md">
            <div className="flex items-start justify-between mb-2">
              <div>
                <h3 className="font-semibold mb-1">Events</h3>
                <p className="text-sm text-gray-600">2 to go</p>
              </div>
              <div className="bg-pink-500 text-white rounded-xl p-2 text-center">
                <div className="text-xs font-medium">Oct</div>
                <div className="text-2xl font-bold">10</div>
              </div>
            </div>
          </div>

          {/* Shopping Collection */}
          <div className="gradient-card-purple p-4 shadow-md col-span-2">
            <h3 className="font-semibold mb-1">Shopping</h3>
            <p className="text-sm text-gray-600 mb-3">16 items</p>
            <div className="flex gap-2">
              <div className="w-16 h-16 rounded-xl overflow-hidden">
                <img 
                  src="https://images.unsplash.com/photo-1505740420928-5e560c06d30e?w=100&h=100&fit=crop" 
                  alt="Shopping" 
                  className="w-full h-full object-cover"
                />
              </div>
              <div className="w-16 h-16 rounded-xl overflow-hidden">
                <img 
                  src="https://images.unsplash.com/photo-1523275335684-37898b6baf30?w=100&h=100&fit=crop" 
                  alt="Shopping" 
                  className="w-full h-full object-cover"
                />
              </div>
              <div className="w-16 h-16 rounded-xl overflow-hidden">
                <img 
                  src="https://images.unsplash.com/photo-1572635196237-14b3f281503f?w=100&h=100&fit=crop" 
                  alt="Shopping" 
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
          <button onClick={() => onNavigate('home')} className="flex flex-col items-center gap-1 text-gray-400">
            <div className="w-10 h-10 rounded-full flex items-center justify-center">
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
          
          <button onClick={() => onNavigate('shopping')} className="flex flex-col items-center gap-1 text-gray-400">
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
