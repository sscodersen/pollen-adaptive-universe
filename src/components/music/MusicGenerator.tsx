
import React, { useState } from 'react';
import { Music, Wand2, Play, Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { musicGenerator, GeneratedTrack } from '../../services/musicGenerator';

interface MusicGeneratorProps {
  onTrackGenerated: (track: GeneratedTrack) => void;
}

export const MusicGenerator: React.FC<MusicGeneratorProps> = ({ onTrackGenerated }) => {
  const [prompt, setPrompt] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [selectedGenre, setSelectedGenre] = useState<string>('');
  const [selectedMood, setSelectedMood] = useState<string>('');

  const genres = ['Electronic', 'Ambient', 'Rock', 'Pop', 'Jazz', 'Classical', 'Hip-Hop', 'Chillout'];
  const moods = ['Upbeat', 'Relaxing', 'Energetic', 'Melancholic', 'Mysterious', 'Joyful', 'Intense', 'Peaceful'];

  const handleGenerate = async () => {
    if (!prompt.trim()) return;
    
    setIsGenerating(true);
    try {
      const track = await musicGenerator.generateMusic({
        prompt: prompt.trim(),
        genre: selectedGenre,
        mood: selectedMood,
        duration: 180 // 3 minutes default
      });
      
      onTrackGenerated(track);
      setPrompt('');
    } catch (error) {
      console.error('Music generation failed:', error);
    } finally {
      setIsGenerating(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !isGenerating) {
      handleGenerate();
    }
  };

  return (
    <div className="bg-gray-900/50 rounded-lg border border-gray-800/50 p-6">
      <div className="flex items-center gap-3 mb-6">
        <div className="p-2 bg-pink-500/20 rounded-lg">
          <Wand2 className="w-5 h-5 text-pink-400" />
        </div>
        <div>
          <h3 className="text-lg font-semibold text-white">AI Music Generator</h3>
          <p className="text-sm text-gray-400">Powered by Pollen AI + ACE-Step</p>
        </div>
      </div>

      {/* Genre Selection */}
      <div className="mb-4">
        <label className="text-sm font-medium text-gray-300 mb-2 block">Genre (Optional)</label>
        <div className="flex flex-wrap gap-2">
          {genres.map((genre) => (
            <Badge
              key={genre}
              variant={selectedGenre === genre ? "default" : "outline"}
              className={`cursor-pointer transition-colors ${
                selectedGenre === genre 
                  ? 'bg-pink-500 hover:bg-pink-600' 
                  : 'border-gray-600 hover:bg-gray-800'
              }`}
              onClick={() => setSelectedGenre(selectedGenre === genre ? '' : genre)}
            >
              {genre}
            </Badge>
          ))}
        </div>
      </div>

      {/* Mood Selection */}
      <div className="mb-4">
        <label className="text-sm font-medium text-gray-300 mb-2 block">Mood (Optional)</label>
        <div className="flex flex-wrap gap-2">
          {moods.map((mood) => (
            <Badge
              key={mood}
              variant={selectedMood === mood ? "default" : "outline"}
              className={`cursor-pointer transition-colors ${
                selectedMood === mood 
                  ? 'bg-cyan-500 hover:bg-cyan-600' 
                  : 'border-gray-600 hover:bg-gray-800'
              }`}
              onClick={() => setSelectedMood(selectedMood === mood ? '' : mood)}
            >
              {mood}
            </Badge>
          ))}
        </div>
      </div>

      {/* Prompt Input */}
      <div className="flex gap-3">
        <Input
          type="text"
          placeholder="Describe the music you want to generate..."
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          onKeyPress={handleKeyPress}
          disabled={isGenerating}
          className="flex-1 bg-gray-800/50 border-gray-700/50 text-white placeholder-gray-400 focus:border-pink-500/50"
        />
        <Button
          onClick={handleGenerate}
          disabled={!prompt.trim() || isGenerating}
          className="bg-pink-600 hover:bg-pink-700 disabled:bg-gray-600"
        >
          {isGenerating ? (
            <>
              <Loader2 className="w-4 h-4 mr-2 animate-spin" />
              Generating...
            </>
          ) : (
            <>
              <Music className="w-4 h-4 mr-2" />
              Generate
            </>
          )}
        </Button>
      </div>

      {/* Quick Prompts */}
      <div className="mt-4">
        <p className="text-xs text-gray-500 mb-2">Quick ideas:</p>
        <div className="flex flex-wrap gap-2">
          {[
            'lofi hip hop for studying',
            'upbeat electronic dance',
            'relaxing piano melody',
            'epic orchestral soundtrack'
          ].map((idea) => (
            <button
              key={idea}
              onClick={() => setPrompt(idea)}
              disabled={isGenerating}
              className="text-xs px-2 py-1 bg-gray-800/50 hover:bg-gray-700/50 text-gray-300 rounded border border-gray-700/50 transition-colors"
            >
              {idea}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};
