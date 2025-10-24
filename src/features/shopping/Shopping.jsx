import { Box, Heading, Text, VStack, Button, Textarea } from '@chakra-ui/react';
import { useState } from 'react';
import { ShoppingBag } from 'lucide-react';
import { useSSEStream } from '@hooks/useSSEStream';
import { API } from '@services/api';

const Shopping = () => {
  const [query, setQuery] = useState('');
  const { data, isStreaming, error, startStream, clearData } = useSSEStream();

  const handleSearch = async () => {
    if (!query.trim()) return;
    
    clearData();
    await startStream(API.shopping.search, {
      query: query,
      user_id: 'user_123',
    });
  };

  return (
    <Box px={4} py={6}>
      <VStack align="start" spacing={4}>
        <Box
          p={4}
          borderRadius="xl"
          bgGradient="linear-gradient(135deg, #667eea 0%, #764ba2 100%)"
          color="white"
          w="100%"
        >
          <ShoppingBag size={32} />
          <Heading size="lg" mt={2}>Shopping Assistant</Heading>
          <Text fontSize="sm" mt={1} opacity={0.9}>
            Find quality products with AI recommendations
          </Text>
        </Box>

        <Textarea
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="What are you looking for? (e.g., best wireless headphones under $200)"
          bg="white"
          borderRadius="lg"
          minH="120px"
        />

        <Button
          onClick={handleSearch}
          isLoading={isStreaming}
          loadingText="Searching..."
          colorScheme="purple"
          w="100%"
          size="lg"
          isDisabled={!query.trim()}
        >
          Search Products
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

        {data.length > 0 && (
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
              AI Recommendations (Streaming):
            </Text>
            <VStack align="start" spacing={2}>
              {data.map((chunk, index) => (
                <Text key={index} fontSize="sm" color="gray.700">
                  {typeof chunk === 'string' ? chunk : JSON.stringify(chunk)}
                </Text>
              ))}
            </VStack>
          </Box>
        )}
      </VStack>
    </Box>
  );
};

export default Shopping;
