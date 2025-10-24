import { useState } from 'react';
import {
  Box,
  Heading,
  VStack,
  Text,
  Button,
  Textarea,
  useToast
} from '@chakra-ui/react';
import { Newspaper } from 'lucide-react';
import { useSSEStream } from '@hooks/useSSEStream';

export default function News() {
  const [query, setQuery] = useState('');
  const { data, isStreaming, error, startStream, clearData } = useSSEStream();
  const toast = useToast();

  const handleSearch = () => {
    if (!query.trim()) {
      toast({
        title: 'Please enter a topic',
        status: 'warning',
        duration: 2000,
      });
      return;
    }

    clearData();
    startStream('/api/news/fetch', { query });
  };

  return (
    <Box px={4} py={6}>
      <VStack align="start" spacing={4}>
        <Box
          p={4}
          borderRadius="xl"
          bgGradient="linear-gradient(135deg, #ec4899 0%, #ef4444 100%)"
          color="white"
          w="100%"
        >
          <Newspaper size={32} />
          <Heading size="lg" mt={2}>Unbiased Updates</Heading>
          <Text fontSize="sm" mt={1} opacity={0.9}>
            Curated news from diverse, credible sources
          </Text>
        </Box>

        <Textarea
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="What news are you interested in? (e.g., latest AI developments, climate change)"
          bg="white"
          borderRadius="lg"
          minH="120px"
        />

        <Button
          onClick={handleSearch}
          isLoading={isStreaming}
          loadingText="Fetching..."
          colorScheme="pink"
          w="100%"
          size="lg"
          isDisabled={!query.trim()}
        >
          Get News
        </Button>

        {error && (
          <Box
            w="100%"
            p={4}
            bg="red.50"
            borderRadius="lg"
            border="1px solid"
            borderColor="red.200"
          >
            <Text fontSize="sm" color="red.700">
              Error: {error}
            </Text>
          </Box>
        )}

        {data && (
          <Box
            w="100%"
            p={4}
            bg="whiteAlpha.800"
            backdropFilter="blur(10px)"
            borderRadius="lg"
            border="1px solid"
            borderColor="whiteAlpha.400"
          >
            <Text fontSize="sm" fontWeight="medium" color="gray.800" mb={2}>
              News Summary (Live Streaming):
            </Text>
            <Text fontSize="sm" color="gray.700" whiteSpace="pre-wrap">
              {data}
            </Text>
          </Box>
        )}
      </VStack>
    </Box>
  );
}