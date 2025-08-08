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
    // Generate a short 2s tone as a data URL so it can actually play
    const freqs = [329, 392, 440, 493, 523];
    const f = freqs[Math.abs(this.hashString(id)) % freqs.length];
    return this.generateToneWav(f, 2.0);
  }

  private hashString(s: string): number {
    let h = 0;
    for (let i = 0; i < s.length; i++) {
      h = ((h << 5) - h) + s.charCodeAt(i);
      h |= 0;
    }
    return h;
  }

  private generateToneWav(frequency: number, durationSec: number): string {
    const sampleRate = 22050;
    const numSamples = Math.floor(durationSec * sampleRate);
    const headerSize = 44;
    const buffer = new ArrayBuffer(headerSize + numSamples * 2);
    const view = new DataView(buffer);

    // RIFF header
    this.writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + numSamples * 2, true);
    this.writeString(view, 8, 'WAVE');

    // fmt chunk
    this.writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true); // PCM
    view.setUint16(20, 1, true);  // Audio format (1 = PCM)
    view.setUint16(22, 1, true);  // Channels
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true); // Byte rate
    view.setUint16(32, 2, true); // Block align
    view.setUint16(34, 16, true); // Bits per sample

    // data chunk
    this.writeString(view, 36, 'data');
    view.setUint32(40, numSamples * 2, true);

    // Samples
    const amplitude = 0.3 * 0x7fff;
    for (let i = 0; i < numSamples; i++) {
      const t = i / sampleRate;
      const sample = Math.sin(2 * Math.PI * frequency * t);
      view.setInt16(44 + i * 2, Math.max(-1, Math.min(1, sample)) * amplitude, true);
    }

    // Convert to base64
    const bytes = new Uint8Array(buffer);
    let binary = '';
    for (let i = 0; i < bytes.length; i++) {
      binary += String.fromCharCode(bytes[i]);
    }
    const base64 = typeof btoa === 'function' ? btoa(binary) : Buffer.from(binary, 'binary').toString('base64');
    return `data:audio/wav;base64,${base64}`;
  }

  private writeString(view: DataView, offset: number, str: string) {
    for (let i = 0; i < str.length; i++) {
      view.setUint8(offset + i, str.charCodeAt(i));
    }
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