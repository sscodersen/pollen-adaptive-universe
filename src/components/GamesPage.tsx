
import React, { useState, useEffect } from 'react';
import { Gamepad2, Star, Users, Download, Trophy, Zap, Target, Sword } from 'lucide-react';
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { gamesAutomation } from '../services/gamesAutomation';
import { unifiedContentEngine } from '../services/unifiedContentEngine';

const staticGames = [
  {
    id: 1,
    title: 'Quantum Realms',
    genre: 'Action RPG',
    rating: 4.8,
    players: '2.4M',
    thumbnail: 'bg-gradient-to-br from-purple-600 to-blue-600',
    description: 'Explore parallel dimensions in this mind-bending adventure.',
    price: 'Free',
    featured: true
  },
  {
    id: 2,
    title: 'Neural Network',
    genre: 'Strategy',
    rating: 4.6,
    players: '890K',
    thumbnail: 'bg-gradient-to-br from-emerald-500 to-cyan-500',
    description: 'Build and optimize AI systems in this complex strategy game.',
    price: '$29.99',
    featured: false
  },
  {
    id: 3,
    title: 'Space Colony Alpha',
    genre: 'Simulation',
    rating: 4.9,
    players: '1.2M',
    thumbnail: 'bg-gradient-to-br from-orange-500 to-red-500',
    description: 'Manage resources and build the ultimate space settlement.',
    price: '$19.99',
    featured: true
  },
  {
    id: 4,
    title: 'Cyber Warrior',
    genre: 'FPS',
    rating: 4.7,
    players: '3.8M',
    thumbnail: 'bg-gradient-to-br from-pink-500 to-purple-500',
    description: 'Fast-paced cyberpunk shooter with advanced AI enemies.',
    price: 'Free',
    featured: false
  }
];

const gameCategories = [
  { name: 'Action', icon: Sword, count: 1456 },
  { name: 'Strategy', icon: Target, count: 892 },
  { name: 'RPG', icon: Trophy, count: 634 },
  { name: 'Simulation', icon: Zap, count: 445 },
];

const tournaments = [
  { name: 'Quantum Championship', prize: '$50,000', participants: 2048, endDate: '3 days' },
  { name: 'Neural Masters', prize: '$25,000', participants: 1024, endDate: '1 week' },
  { name: 'Space Builders Cup', prize: '$15,000', participants: 512, endDate: '2 weeks' }
];

export function GamesPage() {
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [games, setGames] = useState(staticGames);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    const generateGames = async () => {
      setIsLoading(true);
      try {
        // Generate trending games content
        const trendingGames = await unifiedContentEngine.generateContent('games', 6, {
          diversity: 0.8,
          freshness: 0.9,
          qualityThreshold: 7
        });

        const formattedGames = trendingGames.map((item: any, index: number) => ({
          id: index + 100,
          title: item.title,
          genre: item.genre,
          rating: item.rating,
          players: `${(Math.random() * 5 + 0.5).toFixed(1)}M`,
          thumbnail: item.thumbnail,
          description: item.description,
          price: Math.random() > 0.4 ? `$${(Math.random() * 50 + 10).toFixed(2)}` : 'Free',
          featured: item.featured
        }));

        setGames([...formattedGames, ...staticGames]);
      } catch (error) {
        console.error('Failed to generate games:', error);
        setGames(staticGames);
      } finally {
        setIsLoading(false);
      }
    };

    generateGames();
    const interval = setInterval(generateGames, 4 * 60 * 1000); // Refresh every 4 minutes
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="flex-1 bg-gray-950 min-h-screen">
      {/* Header */}
      <div className="bg-gray-900/95 backdrop-blur-sm border-b border-gray-800/50 p-6">
        <div className="max-w-6xl mx-auto">
          <h1 className="text-3xl font-bold text-white mb-2 flex items-center gap-3">
            <Gamepad2 className="w-8 h-8 text-green-400" />
            Games
          </h1>
          <p className="text-gray-400">Discover, play, and compete in cutting-edge games</p>
        </div>
      </div>

      <div className="max-w-6xl mx-auto p-6">
        {/* Game Categories */}
        <div className="mb-8">
          <h2 className="text-2xl font-bold text-white mb-4">Game Categories</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {gameCategories.map((category, index) => (
              <button
                key={index}
                onClick={() => setSelectedCategory(category.name.toLowerCase())}
                className="bg-gray-900/50 border border-gray-800/50 rounded-lg p-4 hover:bg-gray-900/70 transition-colors group"
              >
                <div className="flex flex-col items-center gap-3">
                  <div className="p-3 bg-green-500/20 rounded-lg group-hover:bg-green-500/30 transition-colors">
                    <category.icon className="w-6 h-6 text-green-300" />
                  </div>
                  <div>
                    <h3 className="font-semibold text-white">{category.name}</h3>
                    <p className="text-sm text-gray-400">{category.count} games</p>
                  </div>
                </div>
              </button>
            ))}
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Main Games Section */}
          <div className="lg:col-span-2">
            {/* Featured Games */}
            <div className="mb-8">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-2xl font-bold text-white">Featured Games</h2>
                <Button variant="outline" className="border-gray-700 text-gray-300 hover:bg-gray-800">
                  View All
                </Button>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {games.map((game) => (
                  <div key={game.id} className="bg-gray-900/50 rounded-lg border border-gray-800/50 overflow-hidden hover:border-green-500/50 transition-all group">
                    {/* Game Thumbnail */}
                    <div className={`${game.thumbnail} h-40 relative flex items-center justify-center group-hover:scale-105 transition-transform`}>
                      <Gamepad2 className="w-12 h-12 text-white/80 group-hover:text-white transition-colors" />
                      {game.featured && (
                        <Badge className="absolute top-3 left-3 bg-green-500 hover:bg-green-500">
                          Featured
                        </Badge>
                      )}
                      <span className="absolute top-3 right-3 bg-black/50 text-white px-2 py-1 rounded text-sm font-medium">
                        {game.price}
                      </span>
                    </div>
                    
                    {/* Game Info */}
                    <div className="p-4">
                      <div className="flex items-center gap-2 mb-2">
                        <span className="text-xs font-medium text-green-300 bg-green-500/20 px-2 py-1 rounded">
                          {game.genre}
                        </span>
                      </div>
                      
                      <h3 className="font-semibold text-white mb-2 group-hover:text-green-300 transition-colors">
                        {game.title}
                      </h3>
                      
                      <p className="text-sm text-gray-400 mb-3">
                        {game.description}
                      </p>
                      
                      <div className="flex items-center justify-between text-sm text-gray-400 mb-3">
                        <div className="flex items-center gap-1">
                          <Users className="w-4 h-4" />
                          {game.players}
                        </div>
                        <div className="flex items-center gap-1">
                          <Star className="w-4 h-4 text-yellow-400 fill-current" />
                          <span className="text-white">{game.rating}</span>
                        </div>
                      </div>
                      
                      <div className="flex gap-2">
                        <Button className="flex-1 bg-green-600 hover:bg-green-700">
                          <Download className="w-4 h-4 mr-2" />
                          {game.price === 'Free' ? 'Play Free' : 'Buy Now'}
                        </Button>
                        <Button variant="outline" size="icon" className="border-gray-700 hover:bg-gray-800">
                          <Star className="w-4 h-4" />
                        </Button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Sidebar */}
          <div className="space-y-8">
            {/* Live Tournaments */}
            <div>
              <h2 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
                <Trophy className="w-5 h-5 text-yellow-400" />
                Live Tournaments
              </h2>
              <div className="bg-gray-900/50 rounded-lg border border-gray-800/50 p-4">
                <div className="space-y-4">
                  {tournaments.map((tournament, index) => (
                    <div key={index} className="p-3 bg-gray-800/50 rounded-lg hover:bg-gray-800/70 transition-colors">
                      <h4 className="font-semibold text-white mb-1">{tournament.name}</h4>
                      <div className="text-sm text-gray-400 space-y-1">
                        <div className="flex justify-between">
                          <span>Prize Pool:</span>
                          <span className="text-green-400 font-medium">{tournament.prize}</span>
                        </div>
                        <div className="flex justify-between">
                          <span>Players:</span>
                          <span className="text-white">{tournament.participants.toLocaleString()}</span>
                        </div>
                        <div className="flex justify-between">
                          <span>Ends in:</span>
                          <span className="text-orange-400">{tournament.endDate}</span>
                        </div>
                      </div>
                      <Button size="sm" className="w-full mt-3 bg-yellow-600 hover:bg-yellow-700">
                        Join Tournament
                      </Button>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Gaming Stats */}
            <div>
              <h2 className="text-xl font-bold text-white mb-4">Gaming Community</h2>
              <div className="bg-gray-900/50 rounded-lg border border-gray-800/50 p-4 space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-gray-300">Active Players</span>
                  <span className="text-white font-semibold">892K</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-300">Games Played Today</span>
                  <span className="text-white font-semibold">2.4M</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-300">Total Prize Pools</span>
                  <span className="text-white font-semibold">$2.8M</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-300">New Games This Week</span>
                  <span className="text-white font-semibold">47</span>
                </div>
              </div>
            </div>

            {/* Quick Actions */}
            <div>
              <h2 className="text-xl font-bold text-white mb-4">Quick Actions</h2>
              <div className="space-y-3">
                <Button className="w-full justify-start bg-purple-600 hover:bg-purple-700">
                  <Trophy className="w-4 h-4 mr-2" />
                  Create Tournament
                </Button>
                <Button variant="outline" className="w-full justify-start border-gray-700 text-gray-300 hover:bg-gray-800">
                  <Users className="w-4 h-4 mr-2" />
                  Find Team
                </Button>
                <Button variant="outline" className="w-full justify-start border-gray-700 text-gray-300 hover:bg-gray-800">
                  <Gamepad2 className="w-4 h-4 mr-2" />
                  Game Library
                </Button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
