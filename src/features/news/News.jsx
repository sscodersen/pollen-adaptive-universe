import { useState } from 'react';
import {
  Box,
  Container,
  Heading,
  VStack,
  Card,
  CardBody,
  useToast
} from '@chakra-ui/react';
import { Newspaper } from 'lucide-react';
import SearchBar from '@components/common/SearchBar';
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
    <Container maxW="container.md" py={8}>
      <VStack spacing={6} align="stretch">
        <Box textAlign="center">
          <Newspaper size={48} style={{ margin: '0 auto 16px' }} />
          <Heading size="lg" mb={2}>Unbiased Updates</Heading>
          <Box color="gray.600">Curated news from diverse sources</Box>
        </Box>

        <SearchBar
          value={query}
          onChange={setQuery}
          onSearch={handleSearch}
          placeholder="What news are you interested in?"
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
