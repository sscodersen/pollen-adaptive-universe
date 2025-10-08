import { ArrowLeft, Search, Plus, Star } from "lucide-react";

interface ShoppingProps {
  onNavigate: (screen: 'home' | 'search' | 'collections' | 'shopping') => void;
}

export function Shopping({ onNavigate }: ShoppingProps) {
  return (
    <div className="relative min-h-screen pb-20">
      {/* Header */}
      <div className="p-6 pt-8">
        <div className="flex items-center justify-between mb-6">
          <button onClick={() => onNavigate('collections')} className="w-10 h-10 rounded-full bg-white/80 backdrop-blur-sm flex items-center justify-center">
            <ArrowLeft className="w-5 h-5" />
          </button>
          <h1 className="text-xl font-semibold">Shopping</h1>
          <button className="w-10 h-10 rounded-full bg-white/80 backdrop-blur-sm flex items-center justify-center">
            <Search className="w-5 h-5" />
          </button>
        </div>

        {/* Tabs */}
        <div className="flex gap-2 mb-6">
          <button className="px-6 py-2 bg-white/90 rounded-full text-sm font-medium">All</button>
          <button className="px-6 py-2 bg-white/50 rounded-full text-sm font-medium text-gray-600">Info</button>
          <button className="px-6 py-2 bg-white/50 rounded-full text-sm font-medium text-gray-600">Images</button>
          <button className="px-6 py-2 bg-white/50 rounded-full text-sm font-medium text-gray-600">Products</button>
        </div>

        {/* Products Grid */}
        <div className="space-y-4">
          {/* Wireless Headphones Card */}
          <div className="gradient-card-blue p-4 shadow-md">
            <div className="flex gap-4 mb-3">
              <div className="w-32 h-32 rounded-2xl overflow-hidden bg-gray-100 flex items-center justify-center">
                <img 
                  src="https://images.unsplash.com/photo-1505740420928-5e560c06d30e?w=150&h=150&fit=crop" 
                  alt="Headphones" 
                  className="w-full h-full object-cover"
                />
              </div>
              <div className="flex-1">
                <h3 className="font-semibold mb-2">Buy Wireless Headphones</h3>
                <p className="text-sm text-gray-600 mb-3">A pair of high-quality Bluetooth headphones will allow your ears to enjoy the music without any wires getting in the way.</p>
                <div className="glass-card px-3 py-2 inline-flex items-center gap-2 rounded-xl">
                  <span className="text-xs">📦 Delivery to</span>
                  <span className="text-xs font-semibold">Home</span>
                </div>
              </div>
            </div>
            <div className="flex gap-2">
              <div className="flex-1 glass-card p-3 rounded-xl text-center">
                <img 
                  src="https://images.unsplash.com/photo-1505740420928-5e560c06d30e?w=80&h=80&fit=crop" 
                  alt="Product" 
                  className="w-full h-20 object-cover rounded-lg mb-2"
                />
                <p className="text-xs font-medium mb-1">APPLE HORN</p>
                <p className="text-xs text-gray-600 mb-1">Vice Record P...</p>
                <p className="text-xs font-semibold">$150.99</p>
              </div>
              <div className="flex-1 glass-card p-3 rounded-xl text-center">
                <img 
                  src="https://images.unsplash.com/photo-1484704849700-f032a568e944?w=80&h=80&fit=crop" 
                  alt="Product" 
                  className="w-full h-20 object-cover rounded-lg mb-2"
                />
                <p className="text-xs font-medium mb-1">JBL Tune 500T</p>
                <p className="text-xs text-gray-600 mb-1">Wireless On...</p>
                <p className="text-xs font-semibold">$34.99</p>
              </div>
            </div>
          </div>

          {/* Products List */}
          <div className="space-y-3">
            <div className="gradient-card-pink p-4 shadow-md flex gap-3">
              <div className="w-24 h-24 rounded-xl overflow-hidden">
                <img 
                  src="https://images.unsplash.com/photo-1572635196237-14b3f281503f?w=120&h=120&fit=crop" 
                  alt="Power Bank" 
                  className="w-full h-full object-cover"
                />
              </div>
              <div className="flex-1">
                <h3 className="font-medium mb-1">Charmast Power Bank 20000mAh, 20W Fas...</h3>
                <p className="text-lg font-semibold mb-1">$29.99</p>
                <div className="flex items-center gap-1">
                  {[1, 2, 3, 4].map((i) => (
                    <Star key={i} className="w-3 h-3 fill-yellow-400 text-yellow-400" />
                  ))}
                  <Star className="w-3 h-3 text-gray-300" />
                </div>
              </div>
              <button className="w-8 h-8 rounded-full bg-white/60 flex items-center justify-center">
                <Plus className="w-4 h-4" />
              </button>
            </div>

            <div className="gradient-card-purple p-4 shadow-md flex gap-3">
              <div className="w-24 h-24 rounded-xl overflow-hidden">
                <img 
                  src="https://images.unsplash.com/photo-1523275335684-37898b6baf30?w=120&h=120&fit=crop" 
                  alt="Product" 
                  className="w-full h-full object-cover"
                />
              </div>
              <div className="flex-1">
                <h3 className="font-medium mb-1">Charmast Power Bank 20000mAh, 20W Fas...</h3>
                <p className="text-lg font-semibold mb-1">$29.99</p>
                <div className="flex items-center gap-1">
                  {[1, 2, 3, 4].map((i) => (
                    <Star key={i} className="w-3 h-3 fill-yellow-400 text-yellow-400" />
                  ))}
                  <Star className="w-3 h-3 text-gray-300" />
                </div>
              </div>
              <button className="w-8 h-8 rounded-full bg-white/60 flex items-center justify-center">
                <Plus className="w-4 h-4" />
              </button>
            </div>

            <div className="gradient-card-blue p-4 shadow-md">
              <div className="flex gap-3 mb-3">
                <div className="w-32 h-32 rounded-2xl overflow-hidden">
                  <img 
                    src="https://images.unsplash.com/photo-1494976388531-d1058494cdd8?w=150&h=150&fit=crop" 
                    alt="Electric Vehicle" 
                    className="w-full h-full object-cover"
                  />
                </div>
                <div className="flex-1">
                  <h3 className="font-semibold mb-2">Revolutionary Electric Vehicle</h3>
                  <p className="text-sm text-gray-600">A cutting-edge electric vehicle featuring advanced technology and eco-friendly features that redefine the automotive landscape.</p>
                </div>
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
          
          <button className="flex flex-col items-center gap-1 text-purple-600">
            <div className="w-10 h-10 rounded-full bg-purple-100 flex items-center justify-center">
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
