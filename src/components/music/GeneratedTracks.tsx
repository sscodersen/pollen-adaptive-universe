
import React from 'react';
import { Play, Pause, Download, Heart, Share2, Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { GeneratedTrack } from '../../services/musicGenerator';

interface GeneratedTracksProps {
  tracks: GeneratedTrack[];
  currentlyPlaying: string | null;
  onPlay: (trackId: string) => void;
}

export const GeneratedTracks: React.FC<GeneratedTracksProps> = ({ 
  tracks, 
  currentlyPlaying, 
  onPlay 
}) => {
  if (tracks.length === 0) {
    return (
      <div className="text-center py-8 text-gray-500">
        <p>No generated tracks yet. Use the generator above to create your first AI music!</p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <h3 className="text-xl font-bold text-white flex items-center gap-2">
        <span className="text-pink-400">✨</span>
        Your Generated Tracks
      </h3>
      
      {tracks.map((track) => (
        <div 
          key={track.id} 
          className="bg-gray-900/50 rounded-lg border border-gray-800/50 p-4 hover:bg-gray-900/70 transition-colors group"
        >
          <div className="flex items-center gap-4">
            {/* Track Thumbnail */}
            <div className={`${track.thumbnail} w-16 h-16 rounded-lg flex items-center justify-center relative`}>
              {track.isGenerating ? (
                <Loader2 className="w-6 h-6 text-white animate-spin" />
              ) : (
                <button
                  onClick={() => onPlay(track.id)}
                  className="hover:scale-110 transition-transform"
                  disabled={track.isGenerating}
                >
                  {currentlyPlaying === track.id ? (
                    <Pause className="w-6 h-6 text-white" />
                  ) : (
                    <Play className="w-6 h-6 text-white" />
                  )}
                </button>
              )}
              <Badge className="absolute -top-2 -right-2 bg-pink-500 hover:bg-pink-500 text-xs">
                AI
              </Badge>
            </div>
            
            {/* Track Info */}
            <div className="flex-1 min-w-0">
              <h4 className="font-semibold text-white truncate group-hover:text-pink-300 transition-colors">
                {track.title}
                {track.isGenerating && <span className="text-yellow-400 ml-2">(Generating...)</span>}
              </h4>
              <p className="text-gray-400 text-sm truncate">{track.artist}</p>
              <div className="flex items-center gap-3 mt-1">
                <Badge variant="outline" className="text-xs border-gray-600 text-gray-300">
                  {track.genre}
                </Badge>
                <Badge variant="outline" className="text-xs border-gray-600 text-gray-300">
                  {track.mood}
                </Badge>
                <span className="text-xs text-gray-500">{track.duration}</span>
              </div>
            </div>
            
            {/* Action Buttons */}
            <div className="flex items-center gap-2">
              <Button 
                size="sm" 
                variant="ghost" 
                className="text-gray-400 hover:text-red-400"
                disabled={track.isGenerating}
              >
                <Heart className="w-4 h-4" />
              </Button>
              <Button 
                size="sm" 
                variant="ghost" 
                className="text-gray-400 hover:text-green-400"
                disabled={track.isGenerating}
              >
                <Download className="w-4 h-4" />
              </Button>
              <Button 
                size="sm" 
                variant="ghost" 
                className="text-gray-400 hover:text-blue-400"
                disabled={track.isGenerating}
              >
                <Share2 className="w-4 h-4" />
              </Button>
            </div>
          </div>
          
          {/* Progress Bar for Currently Playing */}
          {currentlyPlaying === track.id && !track.isGenerating && (
            <div className="mt-3">
              <div className="w-full bg-gray-700 rounded-full h-1">
                <div className="bg-pink-500 h-1 rounded-full w-1/3 transition-all duration-300"></div>
              </div>
            </div>
          )}
        </div>
      ))}
    </div>
  );
};
