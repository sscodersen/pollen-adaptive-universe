import { useState } from 'react';
import {
  Box,
  Heading,
  VStack,
  Text,
  Button,
  Textarea,
  Select,
  useToast
} from '@chakra-ui/react';
import { Sparkles } from 'lucide-react';
import { useSSEStream } from '@hooks/useSSEStream';

export default function ContentGeneration() {
  const [query, setQuery] = useState('');
  const [contentType, setContentType] = useState('article');
  const [tone, setTone] = useState('professional');
  const { data, isStreaming, error, startStream, clearData } = useSSEStream();
  const toast = useToast();

  const handleSearch = () => {
    if (!query.trim()) {
      toast({
        title: 'Please describe what you want to create',
        status: 'warning',
        duration: 2000,
      });
      return;
    }

    clearData();
    startStream('/api/content/generate', { 
      query, 
      content_type: contentType, 
      tone, 
      length: 'medium' 
    });
  };

  return (
    <Box px={4} py={6}>
      <VStack align="start" spacing={4}>
        <Box
          p={4}
          borderRadius="xl"
          bgGradient="linear-gradient(135deg, #f97316 0%, #fb923c 100%)"
          color="white"
          w="100%"
        >
          <Sparkles size={32} />
          <Heading size="lg" mt={2}>Content Generation</Heading>
          <Text fontSize="sm" mt={1} opacity={0.9}>
            Generate articles, posts, and creative content
          </Text>
        </Box>

        <Select 
          value={contentType} 
          onChange={(e) => setContentType(e.target.value)}
          bg="white"
          borderRadius="lg"
        >
          <option value="article">Article</option>
          <option value="blog">Blog Post</option>
          <option value="social">Social Media</option>
          <option value="email">Email</option>
          <option value="marketing">Marketing Copy</option>
        </Select>
        
        <Select 
          value={tone} 
          onChange={(e) => setTone(e.target.value)}
          bg="white"
          borderRadius="lg"
        >
          <option value="professional">Professional</option>
          <option value="casual">Casual</option>
          <option value="friendly">Friendly</option>
          <option value="formal">Formal</option>
          <option value="creative">Creative</option>
        </Select>

        <Textarea
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="What would you like to create? (e.g., article about sustainable living)"
          bg="white"
          borderRadius="lg"
          minH="120px"
        />

        <Button
          onClick={handleSearch}
          isLoading={isStreaming}
          loadingText="Generating..."
          colorScheme="orange"
          w="100%"
          size="lg"
          isDisabled={!query.trim()}
        >
          Generate Content
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
              Generated Content (Live Streaming):
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