import { ArrowLeft, Search, Plus, Star } from "lucide-react";
import { BottomNav } from "./BottomNav";

interface ShopProps {
  onNavigate: (screen: 'feed' | 'explore' | 'shop' | 'community') => void;
}

export function Shop({ onNavigate }: ShopProps) {
  return (
    <div className="relative min-h-screen pb-32">
      {/* Header */}
      <div className="p-4 sm:p-6 pt-6 sm:pt-8">
        <div className="flex items-center justify-between mb-6">
          <button onClick={() => onNavigate('feed')} className="w-10 h-10 rounded-full bg-white/80 backdrop-blur-sm flex items-center justify-center">
            <ArrowLeft className="w-5 h-5" />
          </button>
          <h1 className="text-xl font-semibold">Smart Shop</h1>
          <button className="w-10 h-10 rounded-full bg-white/80 backdrop-blur-sm flex items-center justify-center">
            <Search className="w-5 h-5" />
          </button>
        </div>

        {/* Tabs */}
        <div className="flex gap-2 mb-6 overflow-x-auto scrollbar-thin pb-2">
          <button className="px-6 py-2 bg-white/90 rounded-full text-sm font-medium whitespace-nowrap">All</button>
          <button className="px-6 py-2 bg-white/50 rounded-full text-sm font-medium text-gray-600 whitespace-nowrap">Info</button>
          <button className="px-6 py-2 bg-white/50 rounded-full text-sm font-medium text-gray-600 whitespace-nowrap">Images</button>
          <button className="px-6 py-2 bg-white/50 rounded-full text-sm font-medium text-gray-600 whitespace-nowrap">Products</button>
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
      
      <BottomNav currentScreen="shop" onNavigate={onNavigate} />
    </div>
  );
}
