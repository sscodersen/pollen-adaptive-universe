export type SearchIntent = 
  | 'content' 
  | 'news' 
  | 'trends' 
  | 'wellness' 
  | 'shopping' 
  | 'music' 
  | 'media' 
  | 'entertainment' 
  | 'education'
  | 'tools' 
  | 'assistant' 
  | 'smart_home'
  | 'robot'
  | 'general';

export interface SearchMetadata {
  query: string;
  intent: SearchIntent[];
  timestamp: number;
  sessionId?: string;
}

export interface SearchResult {
  id: string;
  type: SearchIntent;
  title: string;
  description: string;
  content: any;
  metadata?: Record<string, any>;
  score: number;
}

export interface SearchResponse {
  results: SearchResult[];
  metadata: SearchMetadata;
  totalResults: number;
}
