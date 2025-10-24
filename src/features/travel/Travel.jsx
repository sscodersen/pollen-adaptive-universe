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
import { Plane } from 'lucide-react';
import { useSSEStream } from '@hooks/useSSEStream';

export default function Travel() {
  const [query, setQuery] = useState('');
  const { data, isStreaming, error, startStream, clearData } = useSSEStream();
  const toast = useToast();

  const handleSearch = () => {
    if (!query.trim()) {
      toast({
        title: 'Please enter your travel plans',
        status: 'warning',
        duration: 2000,
      });
      return;
    }

    clearData();
    startStream('/api/travel/plan', { query });
  };

  return (
    <Box px={4} py={6}>
      <VStack align="start" spacing={4}>
        <Box
          p={4}
          borderRadius="xl"
          bgGradient="linear-gradient(135deg, #06b6d4 0%, #3b82f6 100%)"
          color="white"
          w="100%"
        >
          <Plane size={32} />
          <Heading size="lg" mt={2}>Plan Your Trip</Heading>
          <Text fontSize="sm" mt={1} opacity={0.9}>
            Personalized travel recommendations and itineraries
          </Text>
        </Box>

        <Textarea
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Where would you like to go? Tell me about your travel plans..."
          bg="white"
          borderRadius="lg"
          minH="120px"
        />

        <Button
          onClick={handleSearch}
          isLoading={isStreaming}
          loadingText="Planning..."
          colorScheme="blue"
          w="100%"
          size="lg"
          isDisabled={!query.trim()}
        >
          Plan My Trip
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
              Your Travel Plan (Live Streaming):
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