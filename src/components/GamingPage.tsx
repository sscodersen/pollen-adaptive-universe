import React, { useState, useEffect } from 'react';
import { 
  Gamepad2, 
  Plus, 
  ExternalLink, 
  Trash2, 
  Edit3,
  Globe,
  Play,
  Star,
  Clock,
  Users
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Badge } from '@/components/ui/badge';

interface CustomGame {
  id: string;
  title: string;
  url: string;
  description: string;
  thumbnail?: string;
  genre: string;
  dateAdded: string;
  rating?: number;
  playTime?: string;
  players?: string;
}

export function GamingPage() {
  const [games, setGames] = useState<CustomGame[]>([]);
  const [showAddForm, setShowAddForm] = useState(false);
  const [editingGame, setEditingGame] = useState<CustomGame | null>(null);
  const [formData, setFormData] = useState({
    title: '',
    url: '',
    description: '',
    thumbnail: '',
    genre: '',
    rating: '',
    playTime: '',
    players: ''
  });

  // Load games from localStorage on component mount
  useEffect(() => {
    const savedGames = localStorage.getItem('customGames');
    if (savedGames) {
      setGames(JSON.parse(savedGames));
    }
  }, []);

  // Save games to localStorage whenever games change
  useEffect(() => {
    localStorage.setItem('customGames', JSON.stringify(games));
  }, [games]);

  const resetForm = () => {
    setFormData({
      title: '',
      url: '',
      description: '',
      thumbnail: '',
      genre: '',
      rating: '',
      playTime: '',
      players: ''
    });
    setEditingGame(null);
  };

  const handleAddGame = () => {
    if (!formData.title || !formData.url) return;

    const newGame: CustomGame = {
      id: Date.now().toString(),
      title: formData.title,
      url: formData.url,
      description: formData.description,
      thumbnail: formData.thumbnail,
      genre: formData.genre || 'Casual',
      dateAdded: new Date().toISOString().split('T')[0],
      rating: formData.rating ? parseFloat(formData.rating) : undefined,
      playTime: formData.playTime,
      players: formData.players
    };

    setGames([newGame, ...games]);
    resetForm();
    setShowAddForm(false);
  };

  const handleEditGame = () => {
    if (!formData.title || !formData.url || !editingGame) return;

    const updatedGames = games.map(game =>
      game.id === editingGame.id
        ? {
            ...game,
            title: formData.title,
            url: formData.url,
            description: formData.description,
            thumbnail: formData.thumbnail,
            genre: formData.genre || game.genre,
            rating: formData.rating ? parseFloat(formData.rating) : game.rating,
            playTime: formData.playTime || game.playTime,
            players: formData.players || game.players
          }
        : game
    );

    setGames(updatedGames);
    resetForm();
    setShowAddForm(false);
  };

  const handleDeleteGame = (id: string) => {
    setGames(games.filter(game => game.id !== id));
  };

  const startEditingGame = (game: CustomGame) => {
    setFormData({
      title: game.title,
      url: game.url,
      description: game.description,
      thumbnail: game.thumbnail || '',
      genre: game.genre,
      rating: game.rating?.toString() || '',
      playTime: game.playTime || '',
      players: game.players || ''
    });
    setEditingGame(game);
    setShowAddForm(true);
  };

  const playGame = (url: string) => {
    window.open(url, '_blank', 'noopener,noreferrer');
  };

  const genres = [
    'Action', 'Adventure', 'Arcade', 'Puzzle', 'Strategy', 'RPG', 
    'Simulation', 'Sports', 'Racing', 'Platform', 'Shooter', 'Casual'
  ];

  return (
    <div className="flex-1 bg-gray-950 min-h-0 flex flex-col">
      {/* Header */}
      <div className="bg-gray-900/95 border-b border-gray-800/50 p-6">
        <div className="max-w-7xl mx-auto">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="p-3 bg-gradient-to-br from-green-500/20 to-blue-500/20 rounded-2xl border border-green-500/30">
                <Gamepad2 className="w-8 h-8 text-green-400" />
              </div>
              <div>
                <h1 className="text-3xl font-bold text-white mb-2">My Games</h1>
                <p className="text-gray-400">Add and organize your custom games collection</p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <Badge variant="outline" className="bg-gray-800/50 text-gray-300 border-gray-700/50">
                {games.length} {games.length === 1 ? 'Game' : 'Games'}
              </Badge>
              <Button
                onClick={() => {
                  resetForm();
                  setShowAddForm(true);
                }}
                className="bg-gradient-to-r from-green-500 to-blue-500 hover:from-green-600 hover:to-blue-600 text-white"
              >
                <Plus className="w-4 h-4 mr-2" />
                Add Game
              </Button>
            </div>
          </div>
        </div>
      </div>

      {/* Add/Edit Game Form */}
      {showAddForm && (
        <div className="bg-gray-900/90 border-b border-gray-800/50 p-6">
          <div className="max-w-4xl mx-auto">
            <h3 className="text-xl font-bold text-white mb-4">
              {editingGame ? 'Edit Game' : 'Add New Game'}
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-4">
                <div>
                  <label className="text-sm text-gray-400 mb-2 block">Game Title *</label>
                  <Input
                    value={formData.title}
                    onChange={(e) => setFormData({ ...formData, title: e.target.value })}
                    placeholder="Enter game title..."
                    className="bg-gray-800/50 border-gray-700 text-white"
                  />
                </div>
                <div>
                  <label className="text-sm text-gray-400 mb-2 block">Game URL *</label>
                  <Input
                    value={formData.url}
                    onChange={(e) => setFormData({ ...formData, url: e.target.value })}
                    placeholder="https://your-game-url.com"
                    className="bg-gray-800/50 border-gray-700 text-white"
                  />
                </div>
                <div>
                  <label className="text-sm text-gray-400 mb-2 block">Thumbnail URL</label>
                  <Input
                    value={formData.thumbnail}
                    onChange={(e) => setFormData({ ...formData, thumbnail: e.target.value })}
                    placeholder="https://thumbnail-url.com/image.jpg"
                    className="bg-gray-800/50 border-gray-700 text-white"
                  />
                </div>
                <div>
                  <label className="text-sm text-gray-400 mb-2 block">Genre</label>
                  <select
                    value={formData.genre}
                    onChange={(e) => setFormData({ ...formData, genre: e.target.value })}
                    className="w-full h-10 px-3 bg-gray-800/50 border border-gray-700 text-white rounded-md"
                  >
                    <option value="">Select genre...</option>
                    {genres.map(genre => (
                      <option key={genre} value={genre}>{genre}</option>
                    ))}
                  </select>
                </div>
              </div>
              
              <div className="space-y-4">
                <div>
                  <label className="text-sm text-gray-400 mb-2 block">Description</label>
                  <Textarea
                    value={formData.description}
                    onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                    placeholder="Describe your game..."
                    className="bg-gray-800/50 border-gray-700 text-white min-h-[80px]"
                    rows={3}
                  />
                </div>
                <div className="grid grid-cols-3 gap-3">
                  <div>
                    <label className="text-sm text-gray-400 mb-2 block">Rating</label>
                    <Input
                      type="number"
                      min="0"
                      max="5"
                      step="0.1"
                      value={formData.rating}
                      onChange={(e) => setFormData({ ...formData, rating: e.target.value })}
                      placeholder="4.5"
                      className="bg-gray-800/50 border-gray-700 text-white"
                    />
                  </div>
                  <div>
                    <label className="text-sm text-gray-400 mb-2 block">Play Time</label>
                    <Input
                      value={formData.playTime}
                      onChange={(e) => setFormData({ ...formData, playTime: e.target.value })}
                      placeholder="2h 30m"
                      className="bg-gray-800/50 border-gray-700 text-white"
                    />
                  </div>
                  <div>
                    <label className="text-sm text-gray-400 mb-2 block">Players</label>
                    <Input
                      value={formData.players}
                      onChange={(e) => setFormData({ ...formData, players: e.target.value })}
                      placeholder="1-4"
                      className="bg-gray-800/50 border-gray-700 text-white"
                    />
                  </div>
                </div>
              </div>
            </div>
            
            <div className="flex items-center justify-end space-x-3 mt-6">
              <Button
                onClick={() => {
                  setShowAddForm(false);
                  resetForm();
                }}
                variant="outline"
                className="border-gray-700 text-gray-300"
              >
                Cancel
              </Button>
              <Button
                onClick={editingGame ? handleEditGame : handleAddGame}
                className="bg-gradient-to-r from-green-500 to-blue-500 hover:from-green-600 hover:to-blue-600 text-white"
                disabled={!formData.title || !formData.url}
              >
                {editingGame ? 'Update Game' : 'Add Game'}
              </Button>
            </div>
          </div>
        </div>
      )}

      {/* Games Grid */}
      <div className="flex-1 overflow-auto p-6">
        <div className="max-w-7xl mx-auto">
          {games.length === 0 ? (
            <div className="text-center py-16">
              <div className="p-4 bg-gray-800/30 rounded-full w-20 h-20 mx-auto mb-6 flex items-center justify-center">
                <Gamepad2 className="w-10 h-10 text-gray-500" />
              </div>
              <h3 className="text-xl font-semibold text-white mb-2">No games added yet</h3>
              <p className="text-gray-400 mb-6">Start building your games collection by adding your first game!</p>
              <Button
                onClick={() => {
                  resetForm();
                  setShowAddForm(true);
                }}
                className="bg-gradient-to-r from-green-500 to-blue-500 hover:from-green-600 hover:to-blue-600 text-white"
              >
                <Plus className="w-4 h-4 mr-2" />
                Add Your First Game
              </Button>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
              {games.map((game) => (
                <div
                  key={game.id}
                  className="bg-gray-900/90 rounded-xl border border-gray-800/50 p-6 hover:bg-gray-900/95 transition-all group"
                >
                  {/* Game Thumbnail */}
                  <div className="relative mb-4">
                    {game.thumbnail ? (
                      <img
                        src={game.thumbnail}
                        alt={game.title}
                        className="w-full h-40 object-cover rounded-lg"
                        onError={(e) => {
                          e.currentTarget.style.display = 'none';
                          e.currentTarget.nextElementSibling?.classList.remove('hidden');
                        }}
                      />
                    ) : null}
                    <div className={`w-full h-40 bg-gradient-to-br from-green-500/20 to-blue-500/20 rounded-lg flex items-center justify-center border border-gray-700/30 ${game.thumbnail ? 'hidden' : ''}`}>
                      <Gamepad2 className="w-12 h-12 text-gray-500" />
                    </div>
                    
                    {/* Play Button Overlay */}
                    <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity rounded-lg flex items-center justify-center">
                      <Button
                        onClick={() => playGame(game.url)}
                        size="sm"
                        className="bg-green-500 hover:bg-green-600 text-white"
                      >
                        <Play className="w-4 h-4 mr-2" />
                        Play
                      </Button>
                    </div>
                  </div>

                  {/* Game Info */}
                  <div className="space-y-3">
                    <div className="flex items-start justify-between">
                      <h3 className="font-bold text-white text-lg leading-tight">{game.title}</h3>
                      <div className="flex space-x-1 opacity-0 group-hover:opacity-100 transition-opacity">
                        <Button
                          onClick={() => startEditingGame(game)}
                          size="sm"
                          variant="ghost"
                          className="p-2 h-8 w-8 text-gray-400 hover:text-white"
                        >
                          <Edit3 className="w-4 h-4" />
                        </Button>
                        <Button
                          onClick={() => handleDeleteGame(game.id)}
                          size="sm"
                          variant="ghost"
                          className="p-2 h-8 w-8 text-gray-400 hover:text-red-400"
                        >
                          <Trash2 className="w-4 h-4" />
                        </Button>
                      </div>
                    </div>

                    {game.description && (
                      <p className="text-gray-400 text-sm line-clamp-3">
                        {game.description}
                      </p>
                    )}

                    {/* Game Meta */}
                    <div className="flex flex-wrap gap-2 text-xs">
                      <Badge variant="outline" className="bg-gray-800/50 text-gray-300 border-gray-700/50">
                        {game.genre}
                      </Badge>
                      {game.rating && (
                        <div className="flex items-center space-x-1 text-yellow-400">
                          <Star className="w-3 h-3" />
                          <span>{game.rating}</span>
                        </div>
                      )}
                      {game.playTime && (
                        <div className="flex items-center space-x-1 text-gray-400">
                          <Clock className="w-3 h-3" />
                          <span>{game.playTime}</span>
                        </div>
                      )}
                      {game.players && (
                        <div className="flex items-center space-x-1 text-gray-400">
                          <Users className="w-3 h-3" />
                          <span>{game.players}</span>
                        </div>
                      )}
                    </div>

                    {/* Action Buttons */}
                    <div className="flex space-x-2 pt-2">
                      <Button
                        onClick={() => playGame(game.url)}
                        className="flex-1 bg-gradient-to-r from-green-500 to-blue-500 hover:from-green-600 hover:to-blue-600 text-white"
                        size="sm"
                      >
                        <Play className="w-4 h-4 mr-2" />
                        Play Now
                      </Button>
                      <Button
                        onClick={() => window.open(game.url, '_blank')}
                        variant="outline"
                        size="sm"
                        className="border-gray-700 text-gray-300"
                      >
                        <ExternalLink className="w-4 h-4" />
                      </Button>
                    </div>

                    <div className="text-xs text-gray-500 border-t border-gray-800 pt-2">
                      Added: {new Date(game.dateAdded).toLocaleDateString()}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}