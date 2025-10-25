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
import { GraduationCap, Sparkles } from 'lucide-react';
import { useSSEStream } from '@hooks/useSSEStream';
import { useLocation } from 'react-router-dom';

export default function Education() {
  const location = useLocation();
  const [query, setQuery] = useState(location.state?.query || '');
  const [subject, setSubject] = useState('general');
  const [level, setLevel] = useState('intermediate');
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
        title: 'Please enter what you want to learn',
        status: 'warning',
        duration: 2000,
      });
      return;
    }

    clearData();
    startStream('/api/education/learn', { query, subject, level });
  };

  return (
    <Box px={4} py={6}>
      <VStack align="start" spacing={5}>
        <Box
          w="100%"
          p={6}
          borderRadius="2xl"
          bgGradient="linear(135deg, purple.700, violet.700)"
          color="white"
          position="relative"
          overflow="hidden"
        >
          <Box position="absolute" right="-20px" top="-20px" opacity={0.2}>
            <GraduationCap size={120} />
          </Box>
          <VStack align="start" spacing={2} position="relative">
            <HStack>
              <Icon as={GraduationCap} boxSize={8} />
              <Heading size="lg">Learning Assistant</Heading>
            </HStack>
            <Text fontSize="sm" opacity={0.9}>
              Personalized educational support
            </Text>
          </VStack>
        </Box>

        <VStack w="100%" spacing={4}>
          <HStack w="100%" spacing={3}>
            <Select 
              value={subject} 
              onChange={(e) => setSubject(e.target.value)}
              bg="whiteAlpha.100"
              color="white"
              borderColor="whiteAlpha.300"
              borderRadius="xl"
              _hover={{ borderColor: 'purple.400' }}
              _focus={{ borderColor: 'purple.400', boxShadow: '0 0 0 1px var(--chakra-colors-purple-400)' }}
            >
              <option value="general" style={{ background: '#1a1a2e', color: 'white' }}>General</option>
              <option value="mathematics" style={{ background: '#1a1a2e', color: 'white' }}>Mathematics</option>
              <option value="science" style={{ background: '#1a1a2e', color: 'white' }}>Science</option>
              <option value="programming" style={{ background: '#1a1a2e', color: 'white' }}>Programming</option>
              <option value="languages" style={{ background: '#1a1a2e', color: 'white' }}>Languages</option>
              <option value="history" style={{ background: '#1a1a2e', color: 'white' }}>History</option>
              <option value="arts" style={{ background: '#1a1a2e', color: 'white' }}>Arts</option>
            </Select>
            
            <Select 
              value={level} 
              onChange={(e) => setLevel(e.target.value)}
              bg="whiteAlpha.100"
              color="white"
              borderColor="whiteAlpha.300"
              borderRadius="xl"
              _hover={{ borderColor: 'purple.400' }}
              _focus={{ borderColor: 'purple.400', boxShadow: '0 0 0 1px var(--chakra-colors-purple-400)' }}
            >
              <option value="beginner" style={{ background: '#1a1a2e', color: 'white' }}>Beginner</option>
              <option value="intermediate" style={{ background: '#1a1a2e', color: 'white' }}>Intermediate</option>
              <option value="advanced" style={{ background: '#1a1a2e', color: 'white' }}>Advanced</option>
            </Select>
          </HStack>

          <Textarea
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="What would you like to learn? (e.g., explain quantum physics basics)"
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
            loadingText="Preparing lesson..."
            colorScheme="purple"
            w="100%"
            size="lg"
            isDisabled={!query.trim()}
            leftIcon={<Sparkles size={20} />}
            bgGradient="linear(to-r, purple.500, violet.600)"
            _hover={{ bgGradient: 'linear(to-r, purple.600, violet.700)' }}
          >
            Start Learning
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
              <Sparkles size={18} color="var(--chakra-colors-purple-400)" />
              <Text fontSize="sm" fontWeight="semibold" color="purple.300">
                Learning Content
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
