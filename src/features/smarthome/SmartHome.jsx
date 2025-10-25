import { useState, useEffect } from 'react';
import {
  Box,
  Heading,
  VStack,
  Text,
  Button,
  Textarea,
  HStack,
  Icon,
  useToast
} from '@chakra-ui/react';
import { Home, Sparkles } from 'lucide-react';
import { useSSEStream } from '@hooks/useSSEStream';
import { useLocation } from 'react-router-dom';

export default function SmartHome() {
  const location = useLocation();
  const [query, setQuery] = useState(location.state?.query || '');
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
        title: 'Please describe what you need help with',
        status: 'warning',
        duration: 2000,
      });
      return;
    }

    clearData();
    startStream('/api/smarthome/control', { query });
  };

  return (
    <Box px={4} py={6}>
      <VStack align="start" spacing={5}>
        <Box
          w="100%"
          p={6}
          borderRadius="2xl"
          bgGradient="linear(135deg, green.600, teal.600)"
          color="white"
          position="relative"
          overflow="hidden"
        >
          <Box position="absolute" right="-20px" top="-20px" opacity={0.2}>
            <Home size={120} />
          </Box>
          <VStack align="start" spacing={2} position="relative">
            <HStack>
              <Icon as={Home} boxSize={8} />
              <Heading size="lg">Smart Home Assistant</Heading>
            </HStack>
            <Text fontSize="sm" opacity={0.9}>
              Control and optimize your smart home
            </Text>
          </VStack>
        </Box>

        <VStack w="100%" spacing={4}>
          <Textarea
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="How can I help with your smart home? (e.g., turn off living room lights)"
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
              borderColor: 'teal.400',
              boxShadow: '0 0 0 1px var(--chakra-colors-teal-400)',
            }}
          />

          <Button
            onClick={handleSearch}
            isLoading={isStreaming}
            loadingText="Processing..."
            colorScheme="teal"
            w="100%"
            size="lg"
            isDisabled={!query.trim()}
            leftIcon={<Sparkles size={20} />}
            bgGradient="linear(to-r, green.500, teal.600)"
            _hover={{ bgGradient: 'linear(to-r, green.600, teal.700)' }}
          >
            Control Devices
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
              <Sparkles size={18} color="var(--chakra-colors-teal-400)" />
              <Text fontSize="sm" fontWeight="semibold" color="teal.300">
                Smart Home Assistance
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
