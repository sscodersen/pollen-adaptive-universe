import { useState, useEffect, useRef } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  Box,
  Heading,
  VStack,
  Text,
  HStack,
  Icon,
  Badge,
  Spinner,
  useToast,
  Button,
  IconButton
} from '@chakra-ui/react';
import { TrendingUp, ArrowLeft, RefreshCw } from 'lucide-react';
import { API_BASE_URL } from '@utils/constants';
import PostCard from '@components/common/PostCard';

export default function TrendDetail() {
  const { tag } = useParams();
  const navigate = useNavigate();
  const [content, setContent] = useState([]);
  const [trendInfo, setTrendInfo] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [status, setStatus] = useState('');
  const toast = useToast();
  const eventSourceRef = useRef(null);

  useEffect(() => {
    if (tag) {
      const cleanup = loadTrendContent();
      return cleanup;
    }
  }, [tag]);

  const loadTrendContent = () => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }

    setIsLoading(true);
    setContent([]);
    setStatus(`Loading content for ${tag}...`);

    const params = new URLSearchParams();
    params.append('topic', tag.replace('#', ''));
    params.append('max_results', '20');

    const eventSource = new EventSource(`${API_BASE_URL}/trends/market?${params}`);
    eventSourceRef.current = eventSource;

    eventSource.onmessage = (event) => {
      try {
        const parsed = JSON.parse(event.data);
        
        if (parsed.type === 'status') {
          setStatus(parsed.message);
        } else if (parsed.type === 'trend_info') {
          setTrendInfo(parsed.data);
        } else if (parsed.type === 'content') {
          setContent(prev => [...prev, parsed.data]);
        } else if (parsed.type === 'complete') {
          setStatus('');
          setIsLoading(false);
          eventSource.close();
          eventSourceRef.current = null;
          toast({
            title: parsed.message || 'Content loaded successfully',
            status: 'success',
            duration: 2000,
          });
        } else if (parsed.type === 'error') {
          setStatus('');
          setIsLoading(false);
          eventSource.close();
          eventSourceRef.current = null;
          toast({
            title: 'Error loading trend content',
            description: parsed.error,
            status: 'error',
            duration: 4000,
          });
        }
      } catch (e) {
        console.error('Parse error:', e);
      }
    };

    eventSource.onerror = () => {
      eventSource.close();
      eventSourceRef.current = null;
      setIsLoading(false);
      setStatus('');
      
      if (content.length === 0) {
        toast({
          title: 'Unable to load trend content',
          description: 'Please try again later',
          status: 'warning',
          duration: 3000,
        });
      }
    };

    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
        eventSourceRef.current = null;
      }
    };
  };

  const topicName = tag?.replace('#', '') || 'Trending Topic';

  return (
    <Box px={4} py={6}>
      <VStack align="start" spacing={5}>
        <HStack w="100%" spacing={3}>
          <IconButton
            icon={<ArrowLeft />}
            onClick={() => navigate(-1)}
            variant="ghost"
            colorScheme="purple"
            aria-label="Go back"
          />
        </HStack>

        <Box
          w="100%"
          p={6}
          borderRadius="2xl"
          bgGradient="linear(135deg, purple.600, pink.600)"
          color="white"
          position="relative"
          overflow="hidden"
        >
          <Box position="absolute" right="-20px" top="-20px" opacity={0.2}>
            <TrendingUp size={120} />
          </Box>
          <VStack align="start" spacing={2} position="relative">
            <HStack>
              <Icon as={TrendingUp} boxSize={8} />
              <Heading size="lg">{tag || topicName}</Heading>
            </HStack>
            <Text fontSize="sm" opacity={0.9}>
              Discover trending content, news, and discussions
            </Text>
            {trendInfo && (
              <HStack spacing={3} pt={2}>
                {trendInfo.posts && (
                  <Badge colorScheme="purple" fontSize="sm" px={3} py={1}>
                    {trendInfo.posts} posts
                  </Badge>
                )}
                {trendInfo.trend && (
                  <Badge colorScheme="green" fontSize="sm" px={3} py={1}>
                    {trendInfo.trend} growth
                  </Badge>
                )}
              </HStack>
            )}
          </VStack>
        </Box>

        <HStack w="100%" spacing={3}>
          <Button
            onClick={loadTrendContent}
            isLoading={isLoading}
            colorScheme="purple"
            leftIcon={<RefreshCw size={16} />}
            size="md"
          >
            Refresh
          </Button>
        </HStack>

        {status && (
          <HStack w="100%" justify="center" p={4}>
            <Spinner size="sm" color="purple.400" />
            <Text fontSize="sm" color="gray.400">{status}</Text>
          </HStack>
        )}

        {content.length > 0 && (
          <VStack spacing={4} w="100%">
            {content.map((item, idx) => (
              <PostCard 
                key={idx} 
                post={{
                  ...item,
                  tags: [tag, ...(item.tags || [])],
                  trending: true
                }} 
                showImage={true} 
              />
            ))}
          </VStack>
        )}

        {!isLoading && content.length === 0 && (
          <Box
            w="100%"
            p={8}
            textAlign="center"
            bg="whiteAlpha.50"
            borderRadius="xl"
            border="1px dashed"
            borderColor="whiteAlpha.300"
          >
            <TrendingUp size={48} style={{ margin: '0 auto 16px', opacity: 0.3 }} />
            <Text color="gray.500">
              No content found for this topic yet. Check back later!
            </Text>
          </Box>
        )}
      </VStack>
    </Box>
  );
}
