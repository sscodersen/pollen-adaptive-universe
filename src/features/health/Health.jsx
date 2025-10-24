import { useState } from 'react';
import {
  Box,
  Heading,
  VStack,
  Text,
  Button,
  Textarea,
  Alert,
  AlertIcon,
  useToast
} from '@chakra-ui/react';
import { Heart } from 'lucide-react';
import { useSSEStream } from '@hooks/useSSEStream';

export default function Health() {
  const [query, setQuery] = useState('');
  const { data, isStreaming, error, startStream, clearData } = useSSEStream();
  const toast = useToast();

  const handleSearch = () => {
    if (!query.trim()) {
      toast({
        title: 'Please enter your health question',
        status: 'warning',
        duration: 2000,
      });
      return;
    }

    clearData();
    startStream('/api/health/advice', { query });
  };

  return (
    <Box px={4} py={6}>
      <VStack align="start" spacing={4}>
        <Box
          p={4}
          borderRadius="xl"
          bgGradient="linear-gradient(135deg, #dc2626 0%, #f43f5e 100%)"
          color="white"
          w="100%"
        >
          <Heart size={32} />
          <Heading size="lg" mt={2}>Health & Wellness</Heading>
          <Text fontSize="sm" mt={1} opacity={0.9}>
            Evidence-based health guidance
          </Text>
        </Box>

        <Alert status="info" borderRadius="lg">
          <AlertIcon />
          <Text fontSize="sm">
            This is general wellness information, not medical advice. Consult healthcare professionals for medical concerns.
          </Text>
        </Alert>

        <Textarea
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="What health topic would you like to explore? (e.g., benefits of meditation)"
          bg="white"
          borderRadius="lg"
          minH="120px"
        />

        <Button
          onClick={handleSearch}
          isLoading={isStreaming}
          loadingText="Gathering info..."
          colorScheme="red"
          w="100%"
          size="lg"
          isDisabled={!query.trim()}
        >
          Get Health Advice
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
              Health Guidance (Live Streaming):
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