
import React, { useState } from 'react';
import { Film, Play, Star, Clock, Eye, Headphones, Camera, Gamepad2 } from 'lucide-react';
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";

const entertainmentContent = [
  {
    id: 1,
    title: 'The Quantum Paradox',
    type: 'Movie',
    genre: 'Sci-Fi Thriller',
    duration: '2h 18m',
    rating: 4.8,
    views: '2.4M',
    thumbnail: 'bg-gradient-to-br from-purple-600 to-blue-600',
    description: 'A physicist discovers parallel universes are bleeding into reality through quantum experiments.',
    trending: true
  },
  {
    id: 2,
    title: 'Echoes of Tomorrow',
    type: 'Series',
    genre: 'Drama',
    duration: '8 episodes',
    rating: 4.6,
    views: '1.8M',
    thumbnail: 'bg-gradient-to-br from-emerald-500 to-cyan-500',
    description: 'A group of time travelers try to prevent an apocalyptic future.',
    trending: false
  },
  {
    id: 3,
    title: 'Digital Souls',
    type: 'Documentary',
    genre: 'Technology',
    duration: '1h 45m',
    rating: 4.9,
    views: '890K',
    thumbnail: 'bg-gradient-to-br from-orange-500 to-red-500',
    description: 'Exploring the intersection of consciousness and artificial intelligence.',
    trending: true
  },
  {
    id: 4,
    title: 'Neon Dreams',
    type: 'Music Video',
    genre: 'Electronic',
    duration: '4m 32s',
    rating: 4.7,
    views: '3.2M',
    thumbnail: 'bg-gradient-to-br from-pink-500 to-purple-500',
    description: 'A cyberpunk journey through sound and visual artistry.',
    trending: true
  }
];

const categories = [
  { name: 'Movies', icon: Film, count: 1234 },
  { name: 'Series', icon: Play, count: 567 },
  { name: 'Music', icon: Headphones, count: 2890 },
  { name: 'Documentaries', icon: Camera, count: 445 },
  { name: 'Gaming', icon: Gamepad2, count: 1678 }
];

export function EntertainmentPage() {
  const [selectedCategory, setSelectedCategory] = useState('all');

  return (
    <div className="flex-1 bg-gray-950 min-h-screen">
      {/* Header */}
      <div className="bg-gray-900/95 backdrop-blur-sm border-b border-gray-800/50 p-6">
        <div className="max-w-6xl mx-auto">
          <h1 className="text-3xl font-bold text-white mb-2 flex items-center gap-3">
            <Film className="w-8 h-8 text-purple-400" />
            Entertainment
          </h1>
          <p className="text-gray-400">Discover movies, series, music, and more curated content</p>
        </div>
      </div>

      <div className="max-w-6xl mx-auto p-6">
        {/* Categories */}
        <div className="mb-8">
          <h2 className="text-2xl font-bold text-white mb-4">Browse Categories</h2>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
            {categories.map((category, index) => (
              <button
                key={index}
                onClick={() => setSelectedCategory(category.name.toLowerCase())}
                className="bg-gray-900/50 border border-gray-800/50 rounded-lg p-4 hover:bg-gray-900/70 transition-colors group"
              >
                <div className="flex flex-col items-center gap-3">
                  <div className="p-3 bg-purple-500/20 rounded-lg group-hover:bg-purple-500/30 transition-colors">
                    <category.icon className="w-6 h-6 text-purple-300" />
                  </div>
                  <div>
                    <h3 className="font-semibold text-white">{category.name}</h3>
                    <p className="text-sm text-gray-400">{category.count} items</p>
                  </div>
                </div>
              </button>
            ))}
          </div>
        </div>

        {/* Featured Content */}
        <div className="mb-8">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-2xl font-bold text-white">Trending Now</h2>
            <Button variant="outline" className="border-gray-700 text-gray-300 hover:bg-gray-800">
              View All
            </Button>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {entertainmentContent.map((content) => (
              <div key={content.id} className="bg-gray-900/50 rounded-lg border border-gray-800/50 overflow-hidden hover:border-purple-500/50 transition-all group">
                {/* Thumbnail */}
                <div className={`${content.thumbnail} h-48 relative flex items-center justify-center group-hover:scale-105 transition-transform`}>
                  <Play className="w-12 h-12 text-white/80 group-hover:text-white transition-colors" />
                  {content.trending && (
                    <Badge className="absolute top-3 left-3 bg-red-500 hover:bg-red-500">
                      Trending
                    </Badge>
                  )}
                </div>
                
                {/* Content Info */}
                <div className="p-4">
                  <div className="flex items-center gap-2 mb-2">
                    <span className="text-xs font-medium text-purple-300 bg-purple-500/20 px-2 py-1 rounded">
                      {content.type}
                    </span>
                    <span className="text-xs text-gray-400">{content.genre}</span>
                  </div>
                  
                  <h3 className="font-semibold text-white mb-2 group-hover:text-purple-300 transition-colors">
                    {content.title}
                  </h3>
                  
                  <p className="text-sm text-gray-400 mb-3 line-clamp-2">
                    {content.description}
                  </p>
                  
                  <div className="flex items-center justify-between text-sm text-gray-400 mb-3">
                    <div className="flex items-center gap-1">
                      <Clock className="w-4 h-4" />
                      {content.duration}
                    </div>
                    <div className="flex items-center gap-1">
                      <Eye className="w-4 h-4" />
                      {content.views}
                    </div>
                  </div>
                  
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-1">
                      <Star className="w-4 h-4 text-yellow-400 fill-current" />
                      <span className="text-sm font-medium text-white">{content.rating}</span>
                    </div>
                    <Button size="sm" className="bg-purple-600 hover:bg-purple-700">
                      Watch Now
                    </Button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Recommended Playlists */}
        <div className="mb-8">
          <h2 className="text-2xl font-bold text-white mb-4">Curated Playlists</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {[
              { name: 'Future Sci-Fi', count: 24, color: 'from-blue-500 to-purple-500' },
              { name: 'Mind-Bending Docs', count: 18, color: 'from-emerald-500 to-cyan-500' },
              { name: 'Electronic Vibes', count: 156, color: 'from-pink-500 to-orange-500' }
            ].map((playlist, index) => (
              <div key={index} className="bg-gray-900/50 rounded-lg border border-gray-800/50 p-6 hover:bg-gray-900/70 transition-colors">
                <div className={`w-16 h-16 bg-gradient-to-br ${playlist.color} rounded-lg mb-4 flex items-center justify-center`}>
                  <Play className="w-8 h-8 text-white" />
                </div>
                <h3 className="text-xl font-semibold text-white mb-2">{playlist.name}</h3>
                <p className="text-gray-400 mb-4">{playlist.count} items</p>
                <Button variant="outline" size="sm" className="border-gray-700 text-gray-300 hover:bg-gray-800">
                  Explore Playlist
                </Button>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
