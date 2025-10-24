import { useState } from 'react';
import {
  Box,
  Container,
  Heading,
  VStack,
  Card,
  CardBody,
  Alert,
  AlertIcon,
  useToast
} from '@chakra-ui/react';
import { Heart } from 'lucide-react';
import SearchBar from '@components/common/SearchBar';
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
    <Container maxW="container.md" py={8}>
      <VStack spacing={6} align="stretch">
        <Box textAlign="center">
          <Heart size={48} style={{ margin: '0 auto 16px' }} />
          <Heading size="lg" mb={2}>Health & Wellness</Heading>
          <Box color="gray.600">Evidence-based health guidance</Box>
        </Box>

        <Alert status="info" borderRadius="md">
          <AlertIcon />
          This is general wellness information, not medical advice. Consult healthcare professionals for medical concerns.
        </Alert>

        <SearchBar
          value={query}
          onChange={setQuery}
          onSearch={handleSearch}
          placeholder="What health topic would you like to explore?"
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
