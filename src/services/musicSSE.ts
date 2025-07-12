interface MusicGenerationRequest {
  prompt: string;
  style?: string;
  duration?: number;
}

interface MusicGenerationResponse {
  id: string;
  title: string;
  artist: string;
  audioUrl?: string;
  status: 'generating' | 'completed' | 'error';
  progress?: number;
  error?: string;
}

class MusicSSEService {
  private baseUrl = 'https://ace-step-ace-step.hf.space';
  private eventSource: EventSource | null = null;
  private pollenIntegration = true; // Ready for Pollen integration

  async generateMusic(request: MusicGenerationRequest): Promise<AsyncGenerator<MusicGenerationResponse>> {
    const controller = new AbortController();
    
    return this.createAsyncGenerator(request, controller);
  }

  private async *createAsyncGenerator(
    request: MusicGenerationRequest, 
    controller: AbortController
  ): AsyncGenerator<MusicGenerationResponse> {
    const generationId = `music-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    
    try {
      // Initial response
      yield {
        id: generationId,
        title: this.generateTitle(request.prompt),
        artist: this.generateArtist(request.style || 'electronic'),
        status: 'generating',
        progress: 0
      };

      // Simulate progressive generation with realistic timing
      const steps = [
        { progress: 15, message: 'Analyzing musical prompt...' },
        { progress: 30, message: 'Generating base melody...' },
        { progress: 50, message: 'Adding harmonies and rhythm...' },
        { progress: 70, message: 'Applying style and effects...' },
        { progress: 90, message: 'Finalizing audio quality...' },
        { progress: 100, message: 'Generation complete!' }
      ];

      for (const step of steps) {
        await new Promise(resolve => setTimeout(resolve, 2000 + Math.random() * 1000));
        
        if (controller.signal.aborted) {
          throw new Error('Generation cancelled');
        }

        yield {
          id: generationId,
          title: this.generateTitle(request.prompt),
          artist: this.generateArtist(request.style || 'electronic'),
          status: step.progress === 100 ? 'completed' : 'generating',
          progress: step.progress,
          ...(step.progress === 100 && { 
            audioUrl: this.generateAudioUrl(generationId)
          })
        };
      }

    } catch (error) {
      yield {
        id: generationId,
        title: this.generateTitle(request.prompt),
        artist: this.generateArtist(request.style || 'electronic'),
        status: 'error',
        error: error instanceof Error ? error.message : 'Generation failed'
      };
    }
  }

  private generateTitle(prompt: string): string {
    const words = prompt.toLowerCase().split(' ');
    const titleWords = words.filter(word => 
      !['music', 'song', 'track', 'create', 'generate', 'make'].includes(word)
    );
    
    if (titleWords.length > 0) {
      const key = titleWords[Math.floor(Math.random() * titleWords.length)];
      const suffixes = ['Dreams', 'Waves', 'Echo', 'Pulse', 'Rhythm', 'Harmony', 'Flow', 'Beat'];
      const suffix = suffixes[Math.floor(Math.random() * suffixes.length)];
      return `${key.charAt(0).toUpperCase() + key.slice(1)} ${suffix}`;
    }
    
    const genericTitles = [
      'Digital Dreams', 'Cosmic Flow', 'Neural Beats', 'Quantum Rhythm',
      'Electric Pulse', 'Synthetic Waves', 'Future Harmony', 'AI Symphony'
    ];
    
    return genericTitles[Math.floor(Math.random() * genericTitles.length)];
  }

  private generateArtist(style: string): string {
    const artists = {
      electronic: ['Synthwave AI', 'Digital Echo', 'Neural Beat', 'Code Harmony'],
      ambient: ['Ethereal Sounds', 'Cosmic Drift', 'Deep Space AI', 'Mindful Circuits'],
      classical: ['AI Orchestra', 'Digital Symphony', 'Neural Philharmonic', 'Quantum Strings'],
      rock: ['Electric AI', 'Digital Storm', 'Neural Rock', 'Quantum Band'],
      jazz: ['AI Jazz Collective', 'Digital Blue Note', 'Neural Swing', 'Quantum Fusion'],
      default: ['AI Composer', 'Digital Artist', 'Neural Music', 'Quantum Sound']
    };

    const styleArtists = artists[style as keyof typeof artists] || artists.default;
    return styleArtists[Math.floor(Math.random() * styleArtists.length)];
  }

  private generateAudioUrl(id: string): string {
    // In a real implementation, this would be the actual audio file URL
    // For now, we'll generate a placeholder URL
    return `https://example.com/generated-music/${id}.mp3`;
  }

  cancelGeneration(): void {
    if (this.eventSource) {
      this.eventSource.close();
      this.eventSource = null;
    }
  }

  // Helper method to get available styles
  getAvailableStyles(): string[] {
    return [
      'electronic',
      'ambient',
      'classical',
      'rock',
      'jazz',
      'hip-hop',
      'pop',
      'folk',
      'reggae',
      'blues'
    ];
  }
}

export const musicSSEService = new MusicSSEService();