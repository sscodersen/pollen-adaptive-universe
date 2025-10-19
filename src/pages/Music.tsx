import React, { useState, useEffect, useCallback } from 'react';
import { Music as MusicIcon, Play, Pause, Heart, ListMusic, Radio, Sparkles, TrendingUp } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { contentOrchestrator } from '@/services/contentOrchestrator';
import { personalizationEngine } from '@/services/personalizationEngine';

interface Track {
  id: string;
  title: string;
  artist: string;
  genre: string;
  duration: string;
  mood: string;
  thumbnail: string;
}

interface Playlist {
  id: string;
  name: string;
  description: string;
  trackCount: number;
  genre: string;
  mood: string;
  tracks: Track[];
}

const MUSIC_GENRES = [
  { id: 'all', label: 'For You', icon: Sparkles },
  { id: 'pop', label: 'Pop', icon: MusicIcon },
  { id: 'rock', label: 'Rock', icon: MusicIcon },
  { id: 'electronic', label: 'Electronic', icon: Radio },
  { id: 'jazz', label: 'Jazz', icon: MusicIcon },
  { id: 'classical', label: 'Classical', icon: MusicIcon }
];

const Music = () => {
  const [selectedGenre, setSelectedGenre] = useState('all');
  const [playlists, setPlaylists] = useState<Playlist[]>([]);
  const [loading, setLoading] = useState(true);
  const [playingTrack, setPlayingTrack] = useState<string | null>(null);
  const [likedTracks, setLikedTracks] = useState<Set<string>>(new Set());

  useEffect(() => {
    loadPlaylists();
  }, [selectedGenre]);

  const loadPlaylists = async () => {
    setLoading(true);
    try {
      const genre = selectedGenre === 'all' ? 'mixed' : selectedGenre;
      const query = `${genre} music playlists with different moods and vibes`;

      const response = await contentOrchestrator.generateContent({
        type: 'music',
        query,
        count: 50,
        strategy: {
          diversity: 0.95,
          freshness: 0.9,
          personalization: 0.85,
          qualityThreshold: 7.0,
          trendingBoost: 0.75
        }
      });

      const moods = ['energetic', 'relaxing', 'focus', 'party', 'chill'];
      const playlistsMap = new Map<string, any[]>();
      
      moods.forEach(mood => playlistsMap.set(mood, []));

      response.content.forEach((item: any, index: number) => {
        const mood = moods[index % moods.length];
        playlistsMap.get(mood)?.push({
          id: item.id || `track_${Date.now()}_${index}`,
          title: item.title || `${genre} Track ${index + 1}`,
          artist: item.artist || `AI Artist ${index + 1}`,
          genre,
          duration: item.duration || `${Math.floor(Math.random() * 3) + 2}:${String(Math.floor(Math.random() * 60)).padStart(2, '0')}`,
          mood,
          thumbnail: item.thumbnail || `bg-gradient-to-br from-${['purple', 'blue', 'pink', 'green', 'orange'][index % 5]}-400 to-${['pink', 'purple', 'orange', 'teal', 'red'][index % 5]}-400`
        });
      });

      const generatedPlaylists: Playlist[] = moods.map((mood, index) => ({
        id: `playlist_${Date.now()}_${index}`,
        name: `${mood.charAt(0).toUpperCase() + mood.slice(1)} ${genre.charAt(0).toUpperCase() + genre.slice(1)} Mix`,
        description: `AI-curated ${mood} ${genre} playlist powered by Pollen AI`,
        trackCount: playlistsMap.get(mood)?.length || 10,
        genre,
        mood,
        tracks: playlistsMap.get(mood) || []
      }));

      setPlaylists(generatedPlaylists);
    } catch (error) {
      console.error('Failed to load playlists:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleLikeTrack = useCallback((trackId: string) => {
    setLikedTracks(prev => {
      const newSet = new Set(prev);
      if (newSet.has(trackId)) {
        newSet.delete(trackId);
      } else {
        newSet.add(trackId);
      }
      return newSet;
    });

    personalizationEngine.trackBehavior({
      action: 'like',
      contentId: trackId,
      contentType: 'music',
      metadata: { genre: selectedGenre }
    });
  }, [selectedGenre]);

  const handlePlayPause = useCallback((trackId: string) => {
    setPlayingTrack(prev => prev === trackId ? null : trackId);
    
    personalizationEngine.trackBehavior({
      action: 'click',
      contentId: trackId,
      contentType: 'music',
      metadata: { action: 'play', genre: selectedGenre }
    });
  }, [selectedGenre]);

  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-3 bg-gradient-to-r from-indigo-500 to-purple-500 rounded-2xl">
              <MusicIcon className="w-8 h-8 text-white" />
            </div>
            <div>
              <h1 className="text-4xl font-bold bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent">
                AI Music Studio
              </h1>
              <p className="text-gray-600 dark:text-gray-400">
                Curated playlists powered by Pollen AI
              </p>
            </div>
          </div>
        </div>

        {/* Genre Tabs */}
        <div className="flex gap-2 mb-8 overflow-x-auto scrollbar-thin pb-2">
          {MUSIC_GENRES.map(genre => (
            <Button
              key={genre.id}
              variant={selectedGenre === genre.id ? 'default' : 'outline'}
              onClick={() => setSelectedGenre(genre.id)}
              className="whitespace-nowrap"
            >
              <genre.icon className="w-4 h-4 mr-2" />
              {genre.label}
            </Button>
          ))}
        </div>

        {/* Playlists */}
        {loading ? (
          <div className="space-y-6">
            {[...Array(3)].map((_, i) => (
              <div key={i} className="animate-pulse bg-white/60 dark:bg-gray-800/60 rounded-2xl p-6 h-64" />
            ))}
          </div>
        ) : (
          <div className="space-y-6">
            {playlists.map(playlist => (
              <div
                key={playlist.id}
                className="bg-white/60 dark:bg-gray-800/60 backdrop-blur-sm rounded-2xl p-6 border border-gray-200 dark:border-gray-700"
              >
                {/* Playlist Header */}
                <div className="flex items-start justify-between mb-6">
                  <div>
                    <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">
                      {playlist.name}
                    </h2>
                    <p className="text-gray-600 dark:text-gray-400 mb-2">
                      {playlist.description}
                    </p>
                    <div className="flex items-center gap-4 text-sm text-gray-500 dark:text-gray-400">
                      <div className="flex items-center gap-1">
                        <ListMusic className="w-4 h-4" />
                        {playlist.trackCount} tracks
                      </div>
                      <span className="px-2 py-1 bg-purple-500/20 text-purple-600 rounded-full text-xs font-medium capitalize">
                        {playlist.mood}
                      </span>
                    </div>
                  </div>
                  <Button className="bg-gradient-to-r from-indigo-500 to-purple-500 hover:from-indigo-600 hover:to-purple-600">
                    <Play className="w-4 h-4 mr-2" />
                    Play All
                  </Button>
                </div>

                {/* Tracks */}
                <div className="space-y-2">
                  {playlist.tracks.slice(0, 5).map((track, index) => (
                    <div
                      key={track.id}
                      className="flex items-center gap-4 p-3 rounded-xl hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors group"
                    >
                      <div className="text-gray-400 w-6 text-center text-sm">
                        {index + 1}
                      </div>
                      
                      {/* Thumbnail */}
                      <div className={`w-12 h-12 rounded-lg ${track.thumbnail} flex items-center justify-center`}>
                        <button
                          onClick={() => handlePlayPause(track.id)}
                          className="opacity-0 group-hover:opacity-100 transition-opacity"
                        >
                          {playingTrack === track.id ? (
                            <Pause className="w-6 h-6 text-white" />
                          ) : (
                            <Play className="w-6 h-6 text-white" />
                          )}
                        </button>
                      </div>

                      {/* Track Info */}
                      <div className="flex-1">
                        <h3 className="font-medium text-gray-900 dark:text-white">
                          {track.title}
                        </h3>
                        <p className="text-sm text-gray-500 dark:text-gray-400">
                          {track.artist}
                        </p>
                      </div>

                      {/* Duration */}
                      <span className="text-sm text-gray-500 dark:text-gray-400">
                        {track.duration}
                      </span>

                      {/* Like Button */}
                      <button
                        onClick={() => handleLikeTrack(track.id)}
                        className="p-2 hover:bg-gray-200 dark:hover:bg-gray-600 rounded-lg transition-colors"
                      >
                        <Heart
                          className={`w-5 h-5 ${
                            likedTracks.has(track.id)
                              ? 'fill-red-500 text-red-500'
                              : 'text-gray-400'
                          }`}
                        />
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default Music;
