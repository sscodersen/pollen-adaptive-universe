
import { useState, useCallback } from 'react';
import { pollenAI } from '../services/pollenAI';
import { significanceAlgorithm } from '../services/significanceAlgorithm';

export interface GeneratedContent {
  id: string;
  content: string;
  confidence: number;
  significance: number;
  timestamp: number;
}

export const useContentGeneration = () => {
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const generateContent = useCallback(async (
    prompt: string, 
    mode: string = 'social'
  ): Promise<GeneratedContent | null> => {
    if (!prompt.trim()) return null;

    setIsGenerating(true);
    setError(null);

    try {
      const response = await pollenAI.generate(prompt, mode, true);
      const scored = significanceAlgorithm.scoreContent(response.content, mode as any, 'User Generated');

      return {
        id: Date.now().toString(),
        content: response.content,
        confidence: response.confidence,
        significance: scored.significanceScore,
        timestamp: Date.now()
      };
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Generation failed';
      setError(errorMessage);
      console.error('Content generation error:', err);
      return null;
    } finally {
      setIsGenerating(false);
    }
  }, []);

  const clearError = useCallback(() => setError(null), []);

  return {
    generateContent,
    isGenerating,
    error,
    clearError
  };
};
