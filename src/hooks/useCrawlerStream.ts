import { useEffect, useRef, useState } from 'react';
import { crawlService } from '@/services/crawlService';

export function useCrawlerStream(url: string | null, enabled: boolean) {
  const stopRef = useRef<null | (() => void)>(null);
  const [chunks, setChunks] = useState<string[]>([]);
  const [active, setActive] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!enabled || !url) return;

    setChunks([]);
    setError(null);
    setActive(true);

    stopRef.current = crawlService.stream(url, {
      throttleMs: 350,
      onChunk: (text) => {
        setChunks((prev) => [...prev, text].slice(-40));
      },
      onError: (err) => {
        setError(err?.message || 'Stream error');
        setActive(false);
      },
      onDone: () => setActive(false),
    });

    return () => {
      if (stopRef.current) stopRef.current();
      stopRef.current = null;
    };
  }, [url, enabled]);

  return { chunks, active, error };
}
