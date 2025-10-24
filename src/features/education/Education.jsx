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
import { GraduationCap } from 'lucide-react';
import { useSSEStream } from '@hooks/useSSEStream';

export default function Education() {
  const [query, setQuery] = useState('');
  const [subject, setSubject] = useState('general');
  const [level, setLevel] = useState('intermediate');
  const { data, isStreaming, error, startStream, clearData } = useSSEStream();
  const toast = useToast();

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
      <VStack align="start" spacing={4}>
        <Box
          p={4}
          borderRadius="xl"
          bgGradient="linear-gradient(135deg, #7c3aed 0%, #a855f7 100%)"
          color="white"
          w="100%"
        >
          <GraduationCap size={32} />
          <Heading size="lg" mt={2}>Learning Assistant</Heading>
          <Text fontSize="sm" mt={1} opacity={0.9}>
            Personalized educational support
          </Text>
        </Box>

        <Select 
          value={subject} 
          onChange={(e) => setSubject(e.target.value)}
          bg="white"
          borderRadius="lg"
        >
          <option value="general">General</option>
          <option value="mathematics">Mathematics</option>
          <option value="science">Science</option>
          <option value="programming">Programming</option>
          <option value="languages">Languages</option>
          <option value="history">History</option>
          <option value="arts">Arts</option>
        </Select>
        
        <Select 
          value={level} 
          onChange={(e) => setLevel(e.target.value)}
          bg="white"
          borderRadius="lg"
        >
          <option value="beginner">Beginner</option>
          <option value="intermediate">Intermediate</option>
          <option value="advanced">Advanced</option>
        </Select>

        <Textarea
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="What would you like to learn? (e.g., explain quantum physics basics)"
          bg="white"
          borderRadius="lg"
          minH="120px"
        />

        <Button
          onClick={handleSearch}
          isLoading={isStreaming}
          loadingText="Preparing lesson..."
          colorScheme="purple"
          w="100%"
          size="lg"
          isDisabled={!query.trim()}
        >
          Start Learning
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
              Learning Content (Live Streaming):
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