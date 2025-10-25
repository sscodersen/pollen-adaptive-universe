import { useState, useEffect } from 'react';
import {
  Box,
  Heading,
  VStack,
  Text,
  Button,
  Textarea,
  Select,
  HStack,
  Icon,
  useToast
} from '@chakra-ui/react';
import { Sparkles } from 'lucide-react';
import { useSSEStream } from '@hooks/useSSEStream';
import { useLocation } from 'react-router-dom';

export default function ContentGeneration() {
  const location = useLocation();
  const [query, setQuery] = useState(location.state?.query || '');
  const [contentType, setContentType] = useState('article');
  const [tone, setTone] = useState('professional');
  const { data, isStreaming, error, startStream, clearData } = useSSEStream();
  const toast = useToast();

  useEffect(() => {
    if (location.state?.query) {
      handleSearch();
    }
  }, []);

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
      <VStack align="start" spacing={5}>
        <Box
          w="100%"
          p={6}
          borderRadius="2xl"
          bgGradient="linear(135deg, orange.600, orange.700)"
          color="white"
          position="relative"
          overflow="hidden"
        >
          <Box position="absolute" right="-20px" top="-20px" opacity={0.2}>
            <Sparkles size={120} />
          </Box>
          <VStack align="start" spacing={2} position="relative">
            <HStack>
              <Icon as={Sparkles} boxSize={8} />
              <Heading size="lg">Content Generation</Heading>
            </HStack>
            <Text fontSize="sm" opacity={0.9}>
              Generate articles, posts, and creative content
            </Text>
          </VStack>
        </Box>

        <VStack w="100%" spacing={4}>
          <HStack w="100%" spacing={3}>
            <Select 
              value={contentType} 
              onChange={(e) => setContentType(e.target.value)}
              bg="whiteAlpha.100"
              color="white"
              borderColor="whiteAlpha.300"
              borderRadius="xl"
              _hover={{ borderColor: 'orange.400' }}
              _focus={{ borderColor: 'orange.400', boxShadow: '0 0 0 1px var(--chakra-colors-orange-400)' }}
            >
              <option value="article" style={{ background: '#1a1a2e', color: 'white' }}>Article</option>
              <option value="blog" style={{ background: '#1a1a2e', color: 'white' }}>Blog Post</option>
              <option value="social" style={{ background: '#1a1a2e', color: 'white' }}>Social Media</option>
              <option value="email" style={{ background: '#1a1a2e', color: 'white' }}>Email</option>
              <option value="marketing" style={{ background: '#1a1a2e', color: 'white' }}>Marketing Copy</option>
            </Select>
            
            <Select 
              value={tone} 
              onChange={(e) => setTone(e.target.value)}
              bg="whiteAlpha.100"
              color="white"
              borderColor="whiteAlpha.300"
              borderRadius="xl"
              _hover={{ borderColor: 'orange.400' }}
              _focus={{ borderColor: 'orange.400', boxShadow: '0 0 0 1px var(--chakra-colors-orange-400)' }}
            >
              <option value="professional" style={{ background: '#1a1a2e', color: 'white' }}>Professional</option>
              <option value="casual" style={{ background: '#1a1a2e', color: 'white' }}>Casual</option>
              <option value="friendly" style={{ background: '#1a1a2e', color: 'white' }}>Friendly</option>
              <option value="formal" style={{ background: '#1a1a2e', color: 'white' }}>Formal</option>
              <option value="creative" style={{ background: '#1a1a2e', color: 'white' }}>Creative</option>
            </Select>
          </HStack>

          <Textarea
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="What would you like to create? (e.g., article about sustainable living)"
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
              borderColor: 'orange.400',
              boxShadow: '0 0 0 1px var(--chakra-colors-orange-400)',
            }}
          />

          <Button
            onClick={handleSearch}
            isLoading={isStreaming}
            loadingText="Generating..."
            colorScheme="orange"
            w="100%"
            size="lg"
            isDisabled={!query.trim()}
            leftIcon={<Sparkles size={20} />}
            bgGradient="linear(to-r, orange.500, orange.600)"
            _hover={{ bgGradient: 'linear(to-r, orange.600, orange.700)' }}
          >
            Generate Content
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

        {data && (
          <Box
            w="100%"
            p={5}
            bg="whiteAlpha.100"
            backdropFilter="blur(10px)"
            borderRadius="xl"
            border="1px solid"
            borderColor="whiteAlpha.200"
          >
            <HStack mb={3} spacing={2}>
              <Sparkles size={18} color="var(--chakra-colors-orange-400)" />
              <Text fontSize="sm" fontWeight="semibold" color="orange.300">
                Generated Content
              </Text>
            </HStack>
            <Text fontSize="sm" color="gray.200" whiteSpace="pre-wrap" lineHeight="tall">
              {data}
            </Text>
          </Box>
        )}
      </VStack>
    </Box>
  );
}
