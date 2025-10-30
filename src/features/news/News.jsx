import { useState, useEffect } from 'react';
import {
  Box,
  Heading,
  VStack,
  Text,
  HStack,
  Icon,
  Select,
  Spinner,
  useToast,
  Button
} from '@chakra-ui/react';
import { Newspaper, RefreshCw } from 'lucide-react';
import { API_BASE_URL } from '@utils/constants';
import PostCard from '@components/common/PostCard';

export default function News() {
  const [articles, setArticles] = useState([]);
  const [categories, setCategories] = useState([]);
  const [selectedCategory, setSelectedCategory] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [status, setStatus] = useState('');
  const toast = useToast();

  useEffect(() => {
    fetch(`${API_BASE_URL}/api/news/categories`)
      .then(res => res.json())
      .then(data => setCategories(data.categories || []))
      .catch(() => {});
    
    loadNews();
  }, []);

  const loadNews = () => {
    setIsLoading(true);
    setArticles([]);
    setStatus('Fetching latest news...');

    const params = new URLSearchParams();
    if (selectedCategory) params.append('category', selectedCategory.toLowerCase());
    params.append('min_score', '50');
    params.append('max_results', '20');

    const eventSource = new EventSource(`${API_BASE_URL}/api/news/fetch?${params}`);

    eventSource.onmessage = (event) => {
      try {
        const parsed = JSON.parse(event.data);
        
        if (parsed.type === 'status') {
          setStatus(parsed.message);
        } else if (parsed.type === 'content') {
          setArticles(prev => [...prev, parsed.data]);
        } else if (parsed.type === 'complete') {
          setStatus('');
          setIsLoading(false);
          toast({
            title: parsed.message,
            status: 'success',
            duration: 2000,
          });
        } else if (parsed.type === 'error') {
          setStatus('');
          setIsLoading(false);
          toast({
            title: 'Error loading news',
            description: parsed.error,
            status: 'error',
            duration: 4000,
          });
        }
      } catch (e) {}
    };

    eventSource.onerror = () => {
      eventSource.close();
      setIsLoading(false);
      setStatus('');
    };

    return () => eventSource.close();
  };

  return (
    <Box px={4} py={6}>
      <VStack align="start" spacing={5}>
        <Box
          w="100%"
          p={6}
          borderRadius="2xl"
          bgGradient="linear(135deg, pink.600, red.600)"
          color="white"
          position="relative"
          overflow="hidden"
        >
          <Box position="absolute" right="-20px" top="-20px" opacity={0.2}>
            <Newspaper size={120} />
          </Box>
          <VStack align="start" spacing={2} position="relative">
            <HStack>
              <Icon as={Newspaper} boxSize={8} />
              <Heading size="lg">Unbiased News</Heading>
            </HStack>
            <Text fontSize="sm" opacity={0.9}>
              Curated from BBC, TechCrunch, Hacker News & more
            </Text>
          </VStack>
        </Box>

        <HStack w="100%" spacing={3}>
          <Select
            value={selectedCategory}
            onChange={(e) => setSelectedCategory(e.target.value)}
            bg="whiteAlpha.100"
            backdropFilter="blur(10px)"
            border="1px solid"
            borderColor="whiteAlpha.300"
            color="white"
            borderRadius="xl"
            _focus={{
              bg: 'whiteAlpha.150',
              borderColor: 'pink.400',
            }}
          >
            <option value="" style={{ background: '#1a202c' }}>All Categories</option>
            {categories.map(cat => (
              <option key={cat} value={cat} style={{ background: '#1a202c' }}>
                {cat}
              </option>
            ))}
          </Select>
          <Button
            onClick={loadNews}
            isLoading={isLoading}
            colorScheme="pink"
            leftIcon={<RefreshCw size={16} />}
            size="md"
          >
            Refresh
          </Button>
        </HStack>

        {status && (
          <HStack w="100%" justify="center" p={4}>
            <Spinner size="sm" color="pink.400" />
            <Text fontSize="sm" color="gray.400">{status}</Text>
          </HStack>
        )}

        {articles.length > 0 && (
          <VStack spacing={4} w="100%">
            {articles.map((article, idx) => (
              <PostCard key={idx} post={article} showImage={true} />
            ))}
          </VStack>
        )}

        {!isLoading && articles.length === 0 && (
          <Box
            w="100%"
            p={8}
            textAlign="center"
            bg="whiteAlpha.50"
            borderRadius="xl"
            border="1px dashed"
            borderColor="whiteAlpha.300"
          >
            <Newspaper size={48} style={{ margin: '0 auto 16px', opacity: 0.3 }} />
            <Text color="gray.500">No news found. Try a different category.</Text>
          </Box>
        )}
      </VStack>
    </Box>
  );
}
