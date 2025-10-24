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
import { Sparkles } from 'lucide-react';
import SearchBar from '@components/common/SearchBar';
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
    <Container maxW="container.md" py={8}>
      <VStack spacing={6} align="stretch">
        <Box textAlign="center">
          <Sparkles size={48} style={{ margin: '0 auto 16px' }} />
          <Heading size="lg" mb={2}>Content Generation</Heading>
          <Box color="gray.600">Generate articles, posts, and more</Box>
        </Box>

        <Box>
          <Select value={contentType} onChange={(e) => setContentType(e.target.value)} mb={3}>
            <option value="article">Article</option>
            <option value="blog">Blog Post</option>
            <option value="social">Social Media</option>
            <option value="email">Email</option>
            <option value="marketing">Marketing Copy</option>
          </Select>
          
          <Select value={tone} onChange={(e) => setTone(e.target.value)} mb={3}>
            <option value="professional">Professional</option>
            <option value="casual">Casual</option>
            <option value="friendly">Friendly</option>
            <option value="formal">Formal</option>
            <option value="creative">Creative</option>
          </Select>
        </Box>

        <SearchBar
          value={query}
          onChange={setQuery}
          onSearch={handleSearch}
          placeholder="What would you like to create?"
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
