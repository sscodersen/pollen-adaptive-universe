import { useState } from 'react';
import {
  Box,
  Container,
  Heading,
  VStack,
  Card,
  CardBody,
  Select,
  useToast
} from '@chakra-ui/react';
import { GraduationCap } from 'lucide-react';
import SearchBar from '@components/common/SearchBar';
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
    <Container maxW="container.md" py={8}>
      <VStack spacing={6} align="stretch">
        <Box textAlign="center">
          <GraduationCap size={48} style={{ margin: '0 auto 16px' }} />
          <Heading size="lg" mb={2}>Learning Assistant</Heading>
          <Box color="gray.600">Personalized educational support</Box>
        </Box>

        <Box>
          <Select value={subject} onChange={(e) => setSubject(e.target.value)} mb={3}>
            <option value="general">General</option>
            <option value="mathematics">Mathematics</option>
            <option value="science">Science</option>
            <option value="programming">Programming</option>
            <option value="languages">Languages</option>
            <option value="history">History</option>
            <option value="arts">Arts</option>
          </Select>
          
          <Select value={level} onChange={(e) => setLevel(e.target.value)} mb={3}>
            <option value="beginner">Beginner</option>
            <option value="intermediate">Intermediate</option>
            <option value="advanced">Advanced</option>
          </Select>
        </Box>

        <SearchBar
          value={query}
          onChange={setQuery}
          onSearch={handleSearch}
          placeholder="What would you like to learn?"
          isLoading={isStreaming}
        />

        {error && (
          <Card bg="red.50" borderColor="red.200">
            <CardBody>
              <Box color="red.700">Error: {error}</Box>
            </CardBody>
          </Card>
        )}

        {data && (
          <Card>
            <CardBody>
              <Box whiteSpace="pre-wrap" fontSize="md" lineHeight="tall">
                {data}
              </Box>
            </CardBody>
          </Card>
        )}
      </VStack>
    </Container>
  );
}
