import { Box, Heading, Text, VStack, Button, Textarea, HStack, Icon } from '@chakra-ui/react';
import { useState, useEffect } from 'react';
import { ShoppingBag, Sparkles } from 'lucide-react';
import { useSSEStream } from '@hooks/useSSEStream';
import { API } from '@services/api';
import { useLocation } from 'react-router-dom';
import ResponseActions from '@components/common/ResponseActions';
import TypingIndicator from '@components/common/TypingIndicator';
import { useConversationHistory, useFavorites } from '@hooks/useLocalStorage';

const Shopping = () => {
  const location = useLocation();
  const [query, setQuery] = useState(location.state?.query || '');
  const { data, isStreaming, error, startStream, clearData } = useSSEStream();
  const { addConversation } = useConversationHistory();
  const { addFavorite, isFavorite } = useFavorites();

  useEffect(() => {
    if (location.state?.query) {
      handleSearch();
    }
  }, []);

  useEffect(() => {
    if (data && !isStreaming && query) {
      addConversation(query, data, 'Shopping');
    }
  }, [data, isStreaming]);

  const handleSearch = () => {
    if (!query.trim()) return;
    
    clearData();
    startStream(API.shopping.search, {
      query: query,
      user_id: 'user_123',
    });
  };

  const handleFavorite = (shouldAdd) => {
    if (shouldAdd) {
      addFavorite(query, data, 'Shopping');
    }
  };

  return (
    <Box px={4} py={6}>
      <VStack align="start" spacing={5}>
        <Box
          w="100%"
          p={6}
          borderRadius="2xl"
          bgGradient="linear(135deg, purple.600, purple.800)"
          color="white"
          position="relative"
          overflow="hidden"
        >
          <Box position="absolute" right="-20px" top="-20px" opacity={0.2}>
            <ShoppingBag size={120} />
          </Box>
          <VStack align="start" spacing={2} position="relative">
            <HStack>
              <Icon as={ShoppingBag} boxSize={8} />
              <Heading size="lg">Shopping Assistant</Heading>
            </HStack>
            <Text fontSize="sm" opacity={0.9}>
              Find quality products with AI-powered recommendations
            </Text>
          </VStack>
        </Box>

        <VStack w="100%" spacing={4}>
          <Textarea
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="What are you looking for? (e.g., best wireless headphones under $200)"
            bg="whiteAlpha.100"
            backdropFilter="blur(10px)"
            border="1px solid"
            borderColor="whiteAlpha.300"
            color="white"
            borderRadius="xl"
            minH="120px"
            _placeholder={{ color: 'gray.500' }}
            _focus={{
              bg: 'whiteAlpha.150',
              borderColor: 'purple.400',
              boxShadow: '0 0 0 1px var(--chakra-colors-purple-400)',
            }}
          />

          <Button
            onClick={handleSearch}
            isLoading={isStreaming}
            loadingText="Searching..."
            colorScheme="purple"
            w="100%"
            size="lg"
            isDisabled={!query.trim()}
            leftIcon={<Sparkles size={20} />}
            bgGradient="linear(to-r, purple.500, purple.600)"
            _hover={{ bgGradient: 'linear(to-r, purple.600, purple.700)' }}
          >
            Search Products
          </Button>
        </VStack>

        {error && (
          <Box
            w="100%"
            p={4}
            bg="red.900"
            bgAlpha="0.3"
            borderRadius="xl"
            border="1px solid"
            borderColor="red.700"
          >
            <Text fontSize="sm" color="red.200">
              Error: {error}
            </Text>
          </Box>
        )}

        {(data || isStreaming) && (
          <Box
            w="100%"
            p={5}
            bg="whiteAlpha.100"
            backdropFilter="blur(10px)"
            borderRadius="xl"
            border="1px solid"
            borderColor="whiteAlpha.200"
          >
            <HStack mb={3} spacing={2} justify="space-between">
              <HStack spacing={2}>
                <Sparkles size={18} color="var(--chakra-colors-purple-400)" />
                <Text fontSize="sm" fontWeight="semibold" color="purple.300">
                  AI Recommendations
                </Text>
              </HStack>
              {data && !isStreaming && (
                <ResponseActions 
                  content={data} 
                  onFavorite={handleFavorite}
                  isFavorited={isFavorite(query)}
                />
              )}
            </HStack>
            {isStreaming && !data && (
              <HStack spacing={3} py={3}>
                <TypingIndicator color="purple.400" />
                <Text fontSize="sm" color="gray.400">AI is thinking...</Text>
              </HStack>
            )}
            {data && (
              <Text fontSize="sm" color="gray.200" whiteSpace="pre-wrap" lineHeight="tall">
                {data}
              </Text>
            )}
          </Box>
        )}
      </VStack>
    </Box>
  );
};

export default Shopping;
