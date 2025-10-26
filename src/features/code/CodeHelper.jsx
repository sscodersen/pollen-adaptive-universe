import { useState } from 'react';
import { Box, VStack, Text, Textarea, Button, HStack, Icon } from '@chakra-ui/react';
import { Code, Zap } from 'lucide-react';
import { useSSEStream } from '@hooks/useSSEStream';
import { API } from '@services/api';
import TypingIndicator from '@components/common/TypingIndicator';
import ErrorState from '@components/common/ErrorState';

const CodeHelper = () => {
  const [query, setQuery] = useState('');
  const { response, isLoading, error, startStream, resetStream } = useSSEStream();

  const handleSubmit = (e) => {
    e.preventDefault();
    if (query.trim()) {
      startStream(API.code.help, { query });
    }
  };

  return (
    <Box px={4} py={4}>
      <VStack align="stretch" spacing={6}>
        <VStack align="start" spacing={2}>
          <HStack spacing={3}>
            <Icon as={Code} boxSize={8} color="orange.400" />
            <Text fontSize="3xl" fontWeight="bold" color="white">
              Code Helper
            </Text>
          </HStack>
          <Text fontSize="sm" color="gray.400">
            Get code review, debugging help, and development tips
          </Text>
        </VStack>

        <Box
          as="form"
          onSubmit={handleSubmit}
          p={4}
          bg="black"
          borderRadius="xl"
          border="1px solid"
          borderColor="whiteAlpha.200"
        >
          <VStack spacing={3}>
            <Textarea
              placeholder="Paste your code or describe your coding problem..."
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              size="lg"
              minH="150px"
              bg="whiteAlpha.100"
              border="1px solid"
              borderColor="whiteAlpha.200"
              color="white"
              fontFamily="monospace"
              _placeholder={{ color: 'gray.500' }}
              _focus={{ borderColor: 'orange.400', bg: 'whiteAlpha.150' }}
            />
            <Button
              type="submit"
              colorScheme="orange"
              size="lg"
              w="100%"
              isLoading={isLoading}
              leftIcon={<Icon as={Zap} />}
            >
              Get Coding Help
            </Button>
          </VStack>
        </Box>

        {error && <ErrorState error={error} onRetry={resetStream} />}

        {isLoading && <TypingIndicator />}

        {response && (
          <Box
            p={6}
            bg="black"
            borderRadius="xl"
            border="1px solid"
            borderColor="whiteAlpha.200"
          >
            <Text color="white" whiteSpace="pre-wrap" lineHeight="tall" fontFamily="monospace" fontSize="sm">
              {response}
            </Text>
          </Box>
        )}
      </VStack>
    </Box>
  );
};

export default CodeHelper;
