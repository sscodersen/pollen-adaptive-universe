import React, { useState, useEffect } from 'react';
import { Music, Play, Heart, Share2, Clock, Headphones, Mic, Radio } from 'lucide-react';
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { MusicGenerator } from './music/MusicGenerator';
import { GeneratedTracks } from './music/GeneratedTracks';
import { GeneratedTrack } from '../services/musicGenerator';
import { musicSSEService } from '../services/musicSSE';
import { contentOrchestrator } from '../services/contentOrchestrator';


const staticTracks = [
  {
    id: 1,
    title: 'Digital Echoes',
    artist: 'Synthwave Collective',
    album: 'Future Nostalgia',
    duration: '3:45',
    plays: '2.4M',
    genre: 'Electronic',
    thumbnail: 'bg-gradient-to-br from-purple-600 to-pink-500',
    trending: true
  },
  {
    id: 2,
    title: 'Quantum Beats',
    artist: 'Neural Symphony',
    album: 'AI Compositions Vol. 1',
    duration: '4:12',
    plays: '1.8M',
    genre: 'Ambient',
    thumbnail: 'bg-gradient-to-br from-blue-500 to-cyan-500',
    trending: false
  },
  {
    id: 3,
    title: 'Cosmic Dance',
    artist: 'StarField',
    album: 'Interstellar',
    duration: '5:23',
    plays: '3.1M',
    genre: 'Space Rock',
    thumbnail: 'bg-gradient-to-br from-orange-500 to-red-500',
    trending: true
  },
  {
    id: 4,
    title: 'Binary Dreams',
    artist: 'Code Harmony',
    album: 'Digital Soul',
    duration: '2:58',
    plays: '890K',
    genre: 'Chiptune',
    thumbnail: 'bg-gradient-to-br from-emerald-500 to-teal-500',
    trending: false
  }
];

const playlists = [
  { name: 'Future Beats', tracks: 24, thumbnail: 'bg-gradient-to-br from-purple-500 to-blue-500' },
  { name: 'AI Compositions', tracks: 18, thumbnail: 'bg-gradient-to-br from-green-500 to-cyan-500' },
  { name: 'Space Vibes', tracks: 31, thumbnail: 'bg-gradient-to-br from-orange-500 to-pink-500' },
  { name: 'Digital Chill', tracks: 42, thumbnail: 'bg-gradient-to-br from-indigo-500 to-purple-500' }
];

const genres = [
  { name: 'Electronic', count: 1234, color: 'bg-purple-500/20 text-purple-300' },
  { name: 'Ambient', count: 567, color: 'bg-blue-500/20 text-blue-300' },
  { name: 'Synthwave', count: 890, color: 'bg-pink-500/20 text-pink-300' },
  { name: 'Chiptune', count: 445, color: 'bg-green-500/20 text-green-300' }
];

export function MusicPage() {
  const [currentlyPlaying, setCurrentlyPlaying] = useState<number | null>(null);
  const [generatedTracks, setGeneratedTracks] = useState<GeneratedTrack[]>([]);
  const [currentlyPlayingGenerated, setCurrentlyPlayingGenerated] = useState<string | null>(null);
  const [tracks, setTracks] = useState(staticTracks);
  const [isLoading, setIsLoading] = useState(false);
  const [streamPrompt, setStreamPrompt] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [streamProgress, setStreamProgress] = useState(0);
  const [streamStatus, setStreamStatus] = useState<'idle'|'generating'|'completed'|'error'>('idle');

  useEffect(() => {
    const generateTrendingMusic = async () => {
      setIsLoading(true);
      try {
        const { content: trendingMusic } = await contentOrchestrator.generateContent({
          type: 'music',
          count: 6,
          strategy: {
            diversity: 0.8,
            freshness: 0.9,
            qualityThreshold: 7,
            personalization: 0.8,
            trendingBoost: 1.2
          }
        });

        const formattedTracks = trendingMusic.map((item: any, index: number) => ({
          id: index + 100,
          title: item.title,
          artist: item.artist,
          album: item.album,
          duration: item.duration,
          plays: `${(Math.random() * 5 + 0.5).toFixed(1)}M`,
          genre: item.genre,
          thumbnail: item.thumbnail,
          trending: item.trending
        }));

        setTracks([...formattedTracks, ...staticTracks]);
      } catch (error) {
        console.error('Failed to generate trending music:', error);
        setTracks(staticTracks);
      } finally {
        setIsLoading(false);
      }
    };

    generateTrendingMusic();
    const interval = setInterval(generateTrendingMusic, 3 * 60 * 1000); // Refresh every 3 minutes
    return () => clearInterval(interval);
  }, []);

  const handleTrackGenerated = (track: GeneratedTrack) => {
    setGeneratedTracks(prev => [track, ...prev]);
  };

  const handlePlayGenerated = (trackId: string) => {
    setCurrentlyPlayingGenerated(currentlyPlayingGenerated === trackId ? null : trackId);
    setCurrentlyPlaying(null); // Stop regular tracks
  };

  const handlePlayRegular = (trackId: number) => {
    setCurrentlyPlaying(currentlyPlaying === trackId ? null : trackId);
    setCurrentlyPlayingGenerated(null); // Stop generated tracks
  };

  return (
    <div className="flex-1 bg-gray-950 min-h-screen">
      {/* Header */}
      <div className="bg-gray-900/95 backdrop-blur-sm border-b border-gray-800/50 p-6">
        <div className="max-w-6xl mx-auto">
          <h1 className="text-3xl font-bold text-white mb-2 flex items-center gap-3">
            <Music className="w-8 h-8 text-pink-400" />
            Music
          </h1>
          <p className="text-gray-400">Discover AI-generated music, futuristic sounds, and digital symphonies</p>
        </div>
      </div>

      <div className="max-w-6xl mx-auto p-6">
        {/* AI Music Generator */}
        <div className="mb-8">
          <MusicGenerator onTrackGenerated={handleTrackGenerated} />
        </div>

        {/* Generated Tracks */}
        {generatedTracks.length > 0 && (
          <div className="mb-8">
            <GeneratedTracks 
              tracks={generatedTracks}
              currentlyPlaying={currentlyPlayingGenerated}
              onPlay={handlePlayGenerated}
            />
          </div>
        )}

        {/* Streamed Music Generator (ACE-Step style, no API keys) */}
        <div className="mb-8">
          <div className="bg-gray-900/50 rounded-lg border border-gray-800/50 p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-white">Streamed Music Generator</h3>
              {isStreaming && (
                <span className="text-xs text-pink-400">Generating… {streamProgress}%</span>
              )}
            </div>
            <div className="flex gap-3">
              <input
                className="flex-1 bg-gray-800/50 border border-gray-700/50 rounded px-3 py-2 text-sm text-gray-200"
                placeholder="Describe the track (e.g., upbeat electronic dance)"
                value={streamPrompt}
                onChange={(e) => setStreamPrompt(e.target.value)}
                disabled={isStreaming}
              />
              <Button
                className="bg-pink-600 hover:bg-pink-700 disabled:bg-gray-700"
                disabled={!streamPrompt.trim() || isStreaming}
                onClick={async () => {
                  setIsStreaming(true);
                  setStreamProgress(0);
                  setStreamStatus('generating');
                  try {
                    const gen = await musicSSEService.generateMusic({ prompt: streamPrompt, style: 'electronic', duration: 120 });
                    for await (const ev of gen) {
                      if (typeof ev.progress === 'number') setStreamProgress(ev.progress);
                      if (ev.status === 'completed') {
                        const track: GeneratedTrack = {
                          id: ev.id,
                          title: ev.title,
                          artist: ev.artist,
                          duration: '3:24',
                          audioUrl: ev.audioUrl || '',
                          genre: 'Electronic',
                          mood: 'Upbeat',
                          thumbnail: 'bg-gradient-to-br from-purple-600 to-pink-500',
                          isGenerating: false,
                        };
                        setGeneratedTracks(prev => [track, ...prev]);
                        setStreamStatus('completed');
                      }
                      if (ev.status === 'error') {
                        setStreamStatus('error');
                      }
                    }
                  } catch (e) {
                    setStreamStatus('error');
                  } finally {
                    setIsStreaming(false);
                    setStreamPrompt('');
                  }
                }}
              >
                {isStreaming ? 'Generating…' : 'Generate'}
              </Button>
            </div>
            {isStreaming && (
              <div className="mt-4">
                <div className="w-full bg-gray-700 rounded-full h-1">
                  <div className="bg-pink-500 h-1 rounded-full" style={{ width: `${streamProgress}%` }}></div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Music Genres */}
        <div className="mb-8">
          <h2 className="text-2xl font-bold text-white mb-4">Explore Genres</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {genres.map((genre, index) => (
              <div key={index} className="bg-gray-900/50 border border-gray-800/50 rounded-lg p-4 hover:bg-gray-900/70 transition-colors cursor-pointer">
                <div className="flex flex-col items-center gap-3">
                  <div className={`p-3 rounded-lg ${genre.color}`}>
                    <Music className="w-6 h-6" />
                  </div>
                  <div className="text-center">
                    <h3 className="font-semibold text-white">{genre.name}</h3>
                    <p className="text-sm text-gray-400">{genre.count} tracks</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Main Content */}
          <div className="lg:col-span-2">
            {/* Trending Tracks */}
            <div className="mb-8">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-2xl font-bold text-white">Trending Tracks</h2>
                <Button variant="outline" className="border-gray-700 text-gray-300 hover:bg-gray-800">
                  View All
                </Button>
              </div>
              
              <div className="space-y-4">
                {tracks.map((track) => (
                  <div key={track.id} className="bg-gray-900/50 rounded-lg border border-gray-800/50 p-4 hover:bg-gray-900/70 transition-colors group">
                    <div className="flex items-center gap-4">
                      {/* Track Thumbnail */}
                      <div className={`${track.thumbnail} w-16 h-16 rounded-lg flex items-center justify-center relative`}>
                        <button
                          onClick={() => handlePlayRegular(track.id)}
                          className="hover:scale-110 transition-transform"
                        >
                          <Play className="w-6 h-6 text-white" />
                        </button>
                        {track.trending && (
                          <Badge className="absolute -top-2 -right-2 bg-pink-500 hover:bg-pink-500 text-xs">
                            Hot
                          </Badge>
                        )}
                      </div>
                      
                      {/* Track Info */}
                      <div className="flex-1 min-w-0">
                        <h3 className="font-semibold text-white truncate group-hover:text-pink-300 transition-colors">
                          {track.title}
                        </h3>
                        <p className="text-gray-400 text-sm truncate">{track.artist}</p>
                        <p className="text-gray-500 text-xs truncate">{track.album}</p>
                      </div>
                      
                      {/* Track Stats */}
                      <div className="hidden md:flex items-center gap-6 text-sm text-gray-400">
                        <span className={`px-2 py-1 rounded text-xs ${
                          track.genre === 'Electronic' ? 'bg-purple-500/20 text-purple-300' :
                          track.genre === 'Ambient' ? 'bg-blue-500/20 text-blue-300' :
                          track.genre === 'Space Rock' ? 'bg-orange-500/20 text-orange-300' :
                          'bg-green-500/20 text-green-300'
                        }`}>
                          {track.genre}
                        </span>
                        <div className="flex items-center gap-1">
                          <Headphones className="w-4 h-4" />
                          {track.plays}
                        </div>
                        <div className="flex items-center gap-1">
                          <Clock className="w-4 h-4" />
                          {track.duration}
                        </div>
                      </div>
                      
                      {/* Action Buttons */}
                      <div className="flex items-center gap-2">
                        <Button size="sm" variant="ghost" className="text-gray-400 hover:text-red-400">
                          <Heart className="w-4 h-4" />
                        </Button>
                        <Button size="sm" variant="ghost" className="text-gray-400 hover:text-blue-400">
                          <Share2 className="w-4 h-4" />
                        </Button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Featured Playlists */}
            <div>
              <h2 className="text-2xl font-bold text-white mb-4">Curated Playlists</h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {playlists.map((playlist, index) => (
                  <div key={index} className="bg-gray-900/50 rounded-lg border border-gray-800/50 p-6 hover:bg-gray-900/70 transition-colors group cursor-pointer">
                    <div className="flex items-center gap-4">
                      <div className={`${playlist.thumbnail} w-20 h-20 rounded-lg flex items-center justify-center group-hover:scale-105 transition-transform`}>
                        <Play className="w-8 h-8 text-white" />
                      </div>
                      <div className="flex-1">
                        <h3 className="text-lg font-semibold text-white mb-1 group-hover:text-pink-300 transition-colors">
                          {playlist.name}
                        </h3>
                        <p className="text-gray-400">{playlist.tracks} tracks</p>
                        <Button size="sm" className="mt-2 bg-pink-600 hover:bg-pink-700">
                          Play Playlist
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
            {/* Now Playing */}
            {(currentlyPlaying || currentlyPlayingGenerated) && (
              <div>
                <h2 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
                  <Radio className="w-5 h-5 text-pink-400" />
                  Now Playing
                </h2>
                <div className="bg-gray-900/50 rounded-lg border border-gray-800/50 p-4">
                  {currentlyPlayingGenerated && (
                    // Show generated track that's playing
                    generatedTracks.filter(t => t.id === currentlyPlayingGenerated).map(track => (
                      <div key={track.id}>
                        <div className={`${track.thumbnail} w-full h-32 rounded-lg mb-4 flex items-center justify-center relative`}>
                          <Play className="w-12 h-12 text-white" />
                          <Badge className="absolute top-2 right-2 bg-pink-500 hover:bg-pink-500 text-xs">
                            AI Generated
                          </Badge>
                        </div>
                        <h3 className="font-semibold text-white mb-1">{track.title}</h3>
                        <p className="text-gray-400 text-sm mb-2">{track.artist}</p>
                        <div className="w-full bg-gray-700 rounded-full h-1 mb-2">
                          <div className="bg-pink-500 h-1 rounded-full w-1/3"></div>
                        </div>
                        <div className="flex justify-between text-xs text-gray-400">
                          <span>1:23</span>
                          <span>{track.duration}</span>
                        </div>
                      </div>
                    ))
                  )}
                  {currentlyPlaying && !currentlyPlayingGenerated && (
                    // Show regular track that's playing
                    tracks.filter(t => t.id === currentlyPlaying).map(track => (
                      <div>
                        <div className={`${track.thumbnail} w-full h-32 rounded-lg mb-4 flex items-center justify-center`}>
                          <Play className="w-12 h-12 text-white" />
                        </div>
                        <h3 className="font-semibold text-white mb-1">{track.title}</h3>
                        <p className="text-gray-400 text-sm mb-2">{track.artist}</p>
                        <div className="w-full bg-gray-700 rounded-full h-1 mb-2">
                          <div className="bg-pink-500 h-1 rounded-full w-1/3"></div>
                        </div>
                        <div className="flex justify-between text-xs text-gray-400">
                          <span>1:23</span>
                          <span>{track.duration}</span>
                        </div>
                      </div>
                    ))
                  )}
                </div>
              </div>
            )}

            {/* Music Stats */}
            <div>
              <h2 className="text-xl font-bold text-white mb-4">Platform Stats</h2>
              <div className="bg-gray-900/50 rounded-lg border border-gray-800/50 p-4 space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-gray-300">Total Tracks</span>
                  <span className="text-white font-semibold">4.2K</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-300">Artists</span>
                  <span className="text-white font-semibold">892</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-300">Daily Streams</span>
                  <span className="text-white font-semibold">156K</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-gray-300">Playlists</span>
                  <span className="text-white font-semibold">1.8K</span>
                </div>
              </div>
            </div>

            {/* Quick Actions */}
            <div>
              <h2 className="text-xl font-bold text-white mb-4">Create</h2>
              <div className="space-y-3">
                <Button className="w-full justify-start bg-pink-600 hover:bg-pink-700">
                  <Mic className="w-4 h-4 mr-2" />
                  Record Track
                </Button>
                <Button variant="outline" className="w-full justify-start border-gray-700 text-gray-300 hover:bg-gray-800">
                  <Music className="w-4 h-4 mr-2" />
                  Create Playlist
                </Button>
                <Button variant="outline" className="w-full justify-start border-gray-700 text-gray-300 hover:bg-gray-800">
                  <Radio className="w-4 h-4 mr-2" />
                  Start Radio
                </Button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
