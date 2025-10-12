import { personalizationEngine } from '../personalizationEngine';
import pollenAIUnified from '../pollenAIUnified';

export interface MoodPlaylist {
  id: string;
  name: string;
  mood: string;
  description: string;
  tracks: MusicTrack[];
  coverGradient: string;
  createdBy: string;
  isPublic: boolean;
  likes: number;
  plays: number;
  createdAt: string;
}

export interface MusicTrack {
  id: string;
  title: string;
  artist: string;
  album: string;
  duration: string;
  genre: string;
  mood: string;
  bpm?: number;
  energy?: number;
  preview?: string;
}

export interface LivePerformance {
  id: string;
  artist: string;
  title: string;
  venue: string;
  location: string;
  date: string;
  time: string;
  price: number;
  ticketLink: string;
  type: 'concert' | 'festival' | 'virtual';
  genre: string;
  image: string;
  available: boolean;
}

export interface MusicDiscovery {
  track: MusicTrack;
  reason: string;
  similarTo?: string[];
  discoveryScore: number;
}

class MusicMoodBoardService {
  private readonly PLAYLISTS_KEY = 'mood_playlists';
  private readonly PERFORMANCES_KEY = 'live_performances';

  async createMoodPlaylist(
    name: string,
    mood: string,
    userId: string,
    isPublic: boolean = true,
    trackCount: number = 15
  ): Promise<MoodPlaylist> {
    const tracks = await this.generateMoodTracks(mood, trackCount);
    
    const playlist: MoodPlaylist = {
      id: `playlist_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      name,
      mood,
      description: `A ${mood} playlist curated by AI`,
      tracks,
      coverGradient: this.getMoodGradient(mood),
      createdBy: userId,
      isPublic,
      likes: 0,
      plays: 0,
      createdAt: new Date().toISOString()
    };

    const playlists = this.getAllPlaylists();
    playlists.push(playlist);
    if (typeof window !== 'undefined' && window.localStorage) {
      localStorage.setItem(this.PLAYLISTS_KEY, JSON.stringify(playlists));
    }

    personalizationEngine.trackBehavior({
      action: 'generate',
      contentId: playlist.id,
      contentType: 'music',
      metadata: { mood, trackCount }
    });

    return playlist;
  }

  private async generateMoodTracks(mood: string, count: number): Promise<MusicTrack[]> {
    const genres = ['pop', 'rock', 'electronic', 'indie', 'jazz', 'classical'];
    
    const promises = Array.from({ length: count }, (_, i) => {
      const genre = genres[i % genres.length];
      return pollenAIUnified.generate({
        prompt: `Suggest a ${mood} ${genre} song with artist name. Format: "Song Title" by Artist Name`,
        mode: 'music',
        type: 'track'
      }).then(response => {
        const [title, artist] = this.parseSongResponse(response.content);
        return {
          id: `track_${Date.now()}_${i}`,
          title,
          artist,
          album: `${mood.charAt(0).toUpperCase() + mood.slice(1)} Collection`,
          duration: `${Math.floor(Math.random() * 3) + 2}:${String(Math.floor(Math.random() * 60)).padStart(2, '0')}`,
          genre,
          mood,
          bpm: Math.floor(Math.random() * 100) + 60,
          energy: Math.random()
        };
      }).catch(() => this.generateFallbackTrack(mood, genre, i));
    });

    return Promise.all(promises);
  }

  private parseSongResponse(content: string): [string, string] {
    const match = content.match(/"([^"]+)"\s+by\s+(.+)/i);
    if (match) {
      return [match[1], match[2].trim()];
    }
    return [content.substring(0, 30), 'AI Generated Artist'];
  }

  private generateFallbackTrack(mood: string, genre: string, index: number): MusicTrack {
    return {
      id: `track_${Date.now()}_${index}`,
      title: `${mood.charAt(0).toUpperCase() + mood.slice(1)} ${genre} ${index + 1}`,
      artist: `AI Artist ${index + 1}`,
      album: `${mood} Collection`,
      duration: `${Math.floor(Math.random() * 3) + 2}:${String(Math.floor(Math.random() * 60)).padStart(2, '0')}`,
      genre,
      mood,
      bpm: Math.floor(Math.random() * 100) + 60,
      energy: Math.random()
    };
  }

  private getMoodGradient(mood: string): string {
    const gradients: Record<string, string> = {
      energetic: 'from-red-500 to-orange-500',
      relaxing: 'from-blue-500 to-purple-500',
      focus: 'from-green-500 to-teal-500',
      party: 'from-pink-500 to-purple-500',
      chill: 'from-cyan-500 to-blue-500',
      happy: 'from-yellow-500 to-orange-500',
      melancholic: 'from-gray-600 to-blue-800',
      romantic: 'from-pink-400 to-red-500'
    };

    return gradients[mood.toLowerCase()] || 'from-purple-500 to-blue-500';
  }

  getAllPlaylists(): MoodPlaylist[] {
    if (typeof window === 'undefined' || !window.localStorage) {
      return [];
    }
    try {
      const stored = localStorage.getItem(this.PLAYLISTS_KEY);
      return stored ? JSON.parse(stored) : [];
    } catch {
      return [];
    }
  }

  getPublicPlaylists(): MoodPlaylist[] {
    return this.getAllPlaylists().filter(p => p.isPublic);
  }

  getUserPlaylists(userId: string): MoodPlaylist[] {
    return this.getAllPlaylists().filter(p => p.createdBy === userId);
  }

  likePlaylist(playlistId: string): void {
    const playlists = this.getAllPlaylists();
    const playlist = playlists.find(p => p.id === playlistId);

    if (!playlist) return;

    playlist.likes++;
    if (typeof window !== 'undefined' && window.localStorage) {
      localStorage.setItem(this.PLAYLISTS_KEY, JSON.stringify(playlists));
    }

    personalizationEngine.trackBehavior({
      action: 'like',
      contentId: playlistId,
      contentType: 'music',
      metadata: { mood: playlist.mood }
    });
  }

  playPlaylist(playlistId: string): void {
    const playlists = this.getAllPlaylists();
    const playlist = playlists.find(p => p.id === playlistId);

    if (!playlist) return;

    playlist.plays++;
    if (typeof window !== 'undefined' && window.localStorage) {
      localStorage.setItem(this.PLAYLISTS_KEY, JSON.stringify(playlists));
    }

    personalizationEngine.trackBehavior({
      action: 'view',
      contentId: playlistId,
      contentType: 'music',
      metadata: { mood: playlist.mood, trackCount: playlist.tracks.length }
    });
  }

  async discoverNewMusic(preferences: string[], count: number = 10): Promise<MusicDiscovery[]> {
    const genres = ['indie', 'electronic', 'alternative', 'experimental', 'world music'];
    
    const promises = Array.from({ length: count }, (_, i) => {
      const genre = genres[i % genres.length];
      return pollenAIUnified.generate({
        prompt: `Discover a lesser-known ${genre} artist and song that music lovers should know about`,
        mode: 'music',
        type: 'discovery'
      }).then(response => {
        const [title, artist] = this.parseSongResponse(response.content);
        const track: MusicTrack = {
          id: `discovery_${Date.now()}_${i}`,
          title,
          artist,
          album: 'Discovery',
          duration: `${Math.floor(Math.random() * 3) + 2}:${String(Math.floor(Math.random() * 60)).padStart(2, '0')}`,
          genre,
          mood: 'discovery',
          energy: Math.random()
        };
        return {
          track,
          reason: `Hidden gem in ${genre}`,
          discoveryScore: 0.7 + Math.random() * 0.3
        };
      }).catch(error => {
        console.error('Failed to discover music:', error);
        return null;
      });
    });

    const results = await Promise.all(promises);
    return results.filter((d): d is MusicDiscovery => d !== null)
      .sort((a, b) => b.discoveryScore - a.discoveryScore);
  }

  async generateLivePerformances(count: number = 10): Promise<LivePerformance[]> {
    const performances: LivePerformance[] = [];
    const cities = ['New York', 'Los Angeles', 'Chicago', 'Austin', 'Nashville', 'Seattle'];
    const venues = ['Arena', 'Theater', 'Club', 'Park', 'Hall', 'Stadium'];
    const types: Array<'concert' | 'festival' | 'virtual'> = ['concert', 'festival', 'virtual'];

    for (let i = 0; i < count; i++) {
      const city = cities[i % cities.length];
      const venue = venues[i % venues.length];
      const type = types[i % types.length];

      const date = new Date();
      date.setDate(date.getDate() + Math.floor(Math.random() * 90));

      performances.push({
        id: `performance_${Date.now()}_${i}`,
        artist: `Artist ${i + 1}`,
        title: `${type === 'festival' ? 'Festival' : 'Live'} Performance`,
        venue: `${city} ${venue}`,
        location: city,
        date: date.toISOString().split('T')[0],
        time: `${Math.floor(Math.random() * 12) + 6}:00 PM`,
        price: Math.floor(Math.random() * 100) + 25,
        ticketLink: '#',
        type,
        genre: ['rock', 'pop', 'jazz', 'electronic', 'indie'][i % 5],
        image: `bg-gradient-to-br from-purple-500 to-pink-500`,
        available: Math.random() > 0.2
      });
    }

    return performances.sort((a, b) => 
      new Date(a.date).getTime() - new Date(b.date).getTime()
    );
  }

  getLivePerformances(): LivePerformance[] {
    if (typeof window === 'undefined' || !window.localStorage) {
      return [];
    }
    try {
      const stored = localStorage.getItem(this.PERFORMANCES_KEY);
      return stored ? JSON.parse(stored) : [];
    } catch {
      return [];
    }
  }

  saveLivePerformances(performances: LivePerformance[]): void {
    if (typeof window !== 'undefined' && window.localStorage) {
      localStorage.setItem(this.PERFORMANCES_KEY, JSON.stringify(performances));
    }
  }
}

export const musicMoodBoardService = new MusicMoodBoardService();
