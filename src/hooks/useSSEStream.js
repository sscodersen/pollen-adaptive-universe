import { useState, useCallback, useRef } from 'react';

export const useSSEStream = () => {
  const [data, setData] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState(null);
  const eventSourceRef = useRef(null);

  const startStream = useCallback((url, requestBody) => {
    setIsStreaming(true);
    setError(null);
    setData('');

    const urlParams = new URLSearchParams(requestBody).toString();
    const fullUrl = `${url}?${urlParams}`;

    const eventSource = new EventSource(fullUrl);
    eventSourceRef.current = eventSource;

    const handleMessage = (event) => {
      try {
        const parsed = JSON.parse(event.data);
        if (parsed.text) {
          setData(prev => prev + parsed.text);
        } else if (parsed.error) {
          setError(parsed.error);
          eventSource.close();
          setIsStreaming(false);
        } else if (parsed.content) {
          setData(prev => prev + parsed.content);
        }
      } catch (e) {
        console.warn('Failed to parse SSE data:', e);
        setData(prev => prev + event.data);
      }
    };

    eventSource.onmessage = handleMessage;
    eventSource.addEventListener('ai_response', handleMessage);
    eventSource.addEventListener('scraped_content', handleMessage);

    eventSource.addEventListener('error', (event) => {
      if (event.data) {
        try {
          const parsed = JSON.parse(event.data);
          if (parsed.error) {
            setError(parsed.error);
          }
        } catch (e) {
          setError('Connection error occurred');
        }
      }
    });

    eventSource.onerror = (err) => {
      if (eventSource.readyState === EventSource.CLOSED) {
        setIsStreaming(false);
      } else {
        setError('Connection error occurred');
        eventSource.close();
        setIsStreaming(false);
      }
    };

    eventSource.addEventListener('done', () => {
      eventSource.close();
      setIsStreaming(false);
    });

  }, []);

  const stopStream = useCallback(() => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }
    setIsStreaming(false);
  }, []);

  const clearData = useCallback(() => {
    setData('');
    setError(null);
  }, []);

  return {
    data,
    isStreaming,
    error,
    startStream,
    stopStream,
    clearData,
  };
};
