import { pollenAI } from './pollenAI';
import { pythonScriptIntegration } from './pythonScriptIntegration';

export interface MusicGenerationRequest {
  prompt: string;
  duration?: number;
  genre?: string;
  mood?: string;
}

export interface GeneratedTrack {
  id: string;
  title: string;
  artist: string;
  duration: string;
  audioUrl: string;
  genre: string;
  mood: string;
  thumbnail: string;
  isGenerating: boolean;
}

class MusicGenerator {
  async generateMusic(request: MusicGenerationRequest): Promise<GeneratedTrack> {
    console.log('🎵 Starting AI music generation pipeline...');
    
    // First, use Pollen AI to enhance and filter the music request
    const pollenResponse = await pollenAI.generate(
      `Transform this music request into a detailed, creative music generation prompt: "${request.prompt}". 
       Consider genre: ${request.genre || 'any'}, mood: ${request.mood || 'any'}. 
       Make it specific and inspiring for AI music generation.`,
      'entertainment'
    );
    
    const enhancedPrompt = pollenResponse.content;
    console.log('🌸 Pollen AI enhanced prompt:', enhancedPrompt);
    
    // Generate a unique track ID
    const trackId = `generated-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    
    // Try to call Python script for actual generation
    const pythonResponse = await pythonScriptIntegration.generateMusic({
      prompt: enhancedPrompt,
      genre: request.genre,
      mood: request.mood,
      duration: request.duration
    });
    
    let mockTrack: GeneratedTrack;
    
    if (pythonResponse.success && pythonResponse.data) {
      // Use Python script response
      mockTrack = {
        id: trackId,
        title: pythonResponse.data.title || this.generateTrackTitle(request.prompt, request.genre),
        artist: pythonResponse.data.artist || 'AI Composer',
        duration: pythonResponse.data.duration || this.formatDuration(request.duration || 180),
        audioUrl: pythonResponse.data.audioUrl || this.generateMockAudioUrl(trackId),
        genre: request.genre || 'Electronic',
        mood: request.mood || 'Upbeat',
        thumbnail: this.generateThumbnail(request.genre || 'Electronic'),
        isGenerating: false
      };
    } else {
      // Fallback to simulation
      console.log('📝 Python script not available, using simulation mode');
      mockTrack = {
        id: trackId,
        title: this.generateTrackTitle(request.prompt, request.genre),
        artist: 'AI Composer',
        duration: this.formatDuration(request.duration || 180),
        audioUrl: this.generateMockAudioUrl(trackId),
        genre: request.genre || 'Electronic',
        mood: request.mood || 'Upbeat',
        thumbnail: this.generateThumbnail(request.genre || 'Electronic'),
        isGenerating: true
      };
      
      // Simulate generation time
      setTimeout(() => {
        mockTrack.isGenerating = false;
        console.log('🎵 Music generation completed:', mockTrack.title);
      }, 3000);
    }
    
    return mockTrack;
  }
  
  private formatDuration(seconds: number): string {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
  }
  
  private generateTrackTitle(prompt: string, genre?: string): string {
    const titles = [
      `${prompt.split(' ').slice(0, 2).join(' ')} Dreams`,
      `AI ${genre || 'Sound'} Experience`,
      `Generated ${prompt.split(' ')[0]} Vibes`,
      `Digital ${prompt.split(' ').slice(-1)[0]}`,
      `Neural ${genre || 'Music'} Composition`
    ];
    return titles[Math.floor(Math.random() * titles.length)];
  }
  
  private generateMockAudioUrl(trackId: string): string {
    // In real implementation, this would be the actual generated audio URL
    return `https://example.com/generated-audio/${trackId}.mp3`;
  }
  
  private generateThumbnail(genre: string): string {
    const thumbnails = {
      'Electronic': 'bg-gradient-to-br from-purple-600 to-pink-500',
      'Ambient': 'bg-gradient-to-br from-blue-500 to-cyan-500',
      'Rock': 'bg-gradient-to-br from-orange-500 to-red-500',
      'Pop': 'bg-gradient-to-br from-pink-500 to-purple-500',
      'Jazz': 'bg-gradient-to-br from-yellow-500 to-orange-500',
      'Classical': 'bg-gradient-to-br from-indigo-500 to-purple-500'
    };
    return thumbnails[genre as keyof typeof thumbnails] || 'bg-gradient-to-br from-gray-500 to-gray-700';
  }
}

export const musicGenerator = new MusicGenerator();
