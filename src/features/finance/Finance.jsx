import { useState } from 'react';
import { Box, VStack, Text, Input, Button, HStack, Icon } from '@chakra-ui/react';
import { DollarSign, TrendingUp } from 'lucide-react';
import { useSSEStream } from '@hooks/useSSEStream';
import { API } from '@services/api';
import TypingIndicator from '@components/common/TypingIndicator';
import ErrorState from '@components/common/ErrorState';

const Finance = () => {
  const [query, setQuery] = useState('');
  const { response, isLoading, error, startStream, resetStream } = useSSEStream();

  const handleSubmit = (e) => {
    e.preventDefault();
    if (query.trim()) {
      startStream(API.finance.advice, { query });
    }
  };

  return (
    <Box px={4} py={4}>
      <VStack align="stretch" spacing={6}>
        <VStack align="start" spacing={2}>
          <HStack spacing={3}>
            <Icon as={DollarSign} boxSize={8} color="green.400" />
            <Text fontSize="3xl" fontWeight="bold" color="white">
              Finance Assistant
            </Text>
          </HStack>
          <Text fontSize="sm" color="gray.400">
            Get budget planning, investment insights, and financial advice
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
            <Input
              placeholder="Ask about budgeting, investments, savings..."
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              size="lg"
              bg="whiteAlpha.100"
              border="1px solid"
              borderColor="whiteAlpha.200"
              color="white"
              _placeholder={{ color: 'gray.500' }}
              _focus={{ borderColor: 'green.400', bg: 'whiteAlpha.150' }}
            />
            <Button
              type="submit"
              colorScheme="green"
              size="lg"
              w="100%"
              isLoading={isLoading}
              leftIcon={<Icon as={TrendingUp} />}
            >
              Get Financial Advice
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
            <Text color="white" whiteSpace="pre-wrap" lineHeight="tall">
              {response}
            </Text>
          </Box>
        )}
      </VStack>
    </Box>
  );
};

export default Finance;
