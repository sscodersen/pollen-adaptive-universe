import { useState, useRef, useEffect } from 'react';
import {
  Box,
  InputGroup,
  Input,
  InputLeftElement,
  InputRightElement,
  IconButton,
  VStack,
  Text,
  useDisclosure,
  Collapse,
  Spinner,
  useToast,
} from '@chakra-ui/react';
import { Search, X, Sparkles } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

const FloatingSearchBar = () => {
  const [query, setQuery] = useState('');
  const [isExpanded, setIsExpanded] = useState(false);
  const inputRef = useRef(null);
  const navigate = useNavigate();
  const toast = useToast();

  const suggestions = [
    { text: 'Find best laptops under $1000', route: '/shopping', icon: '🛍️' },
    { text: 'Plan a trip to Tokyo', route: '/travel', icon: '✈️' },
    { text: 'Latest tech news', route: '/news', icon: '📰' },
    { text: 'Generate blog post ideas', route: '/content', icon: '✍️' },
    { text: 'Control smart lights', route: '/smarthome', icon: '💡' },
    { text: 'Healthy breakfast ideas', route: '/health', icon: '🥗' },
    { text: 'Learn Python basics', route: '/education', icon: '📚' },
  ];

  const handleSearch = (searchQuery = query, route = null) => {
    if (!searchQuery.trim()) {
      toast({
        title: 'Enter a search query',
        status: 'warning',
        duration: 2000,
        isClosable: true,
      });
      return;
    }

    if (route) {
      navigate(route, { state: { query: searchQuery } });
    } else {
      const lowerQuery = searchQuery.toLowerCase();
      
      if (lowerQuery.includes('shop') || lowerQuery.includes('buy') || lowerQuery.includes('product')) {
        navigate('/shopping', { state: { query: searchQuery } });
      } else if (lowerQuery.includes('travel') || lowerQuery.includes('trip') || lowerQuery.includes('vacation')) {
        navigate('/travel', { state: { query: searchQuery } });
      } else if (lowerQuery.includes('news') || lowerQuery.includes('article')) {
        navigate('/news', { state: { query: searchQuery } });
      } else if (lowerQuery.includes('write') || lowerQuery.includes('create') || lowerQuery.includes('generate')) {
        navigate('/content', { state: { query: searchQuery } });
      } else if (lowerQuery.includes('smart home') || lowerQuery.includes('iot') || lowerQuery.includes('control')) {
        navigate('/smarthome', { state: { query: searchQuery } });
      } else if (lowerQuery.includes('health') || lowerQuery.includes('fitness') || lowerQuery.includes('wellness')) {
        navigate('/health', { state: { query: searchQuery } });
      } else if (lowerQuery.includes('learn') || lowerQuery.includes('study') || lowerQuery.includes('education')) {
        navigate('/education', { state: { query: searchQuery } });
      } else {
        navigate('/dashboard', { state: { query: searchQuery, useAI: true } });
      }
    }
    
    setQuery('');
    setIsExpanded(false);
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      handleSearch();
    }
  };

  useEffect(() => {
    const handleGlobalKeyPress = (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        setIsExpanded(true);
        setTimeout(() => inputRef.current?.focus(), 100);
      }
      if (e.key === 'Escape') {
        setIsExpanded(false);
        setQuery('');
      }
    };

    window.addEventListener('keydown', handleGlobalKeyPress);
    return () => window.removeEventListener('keydown', handleGlobalKeyPress);
  }, []);

  return (
    <>
      {isExpanded && (
        <Box
          position="fixed"
          top={0}
          left={0}
          right={0}
          bottom={0}
          bg="blackAlpha.600"
          backdropFilter="blur(4px)"
          zIndex={999}
          onClick={() => setIsExpanded(false)}
        />
      )}

      <Box
        position="fixed"
        bottom={isExpanded ? '50%' : '100px'}
        left="50%"
        transform={isExpanded ? 'translate(-50%, 50%)' : 'translate(-50%, 0)'}
        zIndex={1000}
        transition="all 0.3s ease-in-out"
        w={isExpanded ? '90%' : 'auto'}
        maxW="480px"
      >
        <VStack spacing={2} align="stretch">
          <InputGroup size={isExpanded ? 'lg' : 'md'}>
            <InputLeftElement pointerEvents="none" color="purple.500">
              <Sparkles size={20} />
            </InputLeftElement>
            <Input
              ref={inputRef}
              placeholder="Ask Pollen AI anything... (⌘K)"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyPress={handleKeyPress}
              onFocus={() => setIsExpanded(true)}
              bg="white"
              border="2px solid"
              borderColor="purple.400"
              borderRadius="full"
              boxShadow="0 8px 32px rgba(128, 90, 213, 0.3)"
              _focus={{
                borderColor: 'purple.500',
                boxShadow: '0 8px 48px rgba(128, 90, 213, 0.5)',
              }}
              _placeholder={{ color: 'gray.400' }}
              pr={query ? '100px' : '50px'}
            />
            <InputRightElement w="auto" pr={2}>
              {query && (
                <IconButton
                  icon={<X size={18} />}
                  size="sm"
                  variant="ghost"
                  colorScheme="purple"
                  onClick={() => setQuery('')}
                  mr={1}
                  aria-label="Clear search"
                />
              )}
              <IconButton
                icon={<Search size={18} />}
                size="sm"
                colorScheme="purple"
                onClick={() => handleSearch()}
                aria-label="Search"
              />
            </InputRightElement>
          </InputGroup>

          <Collapse in={isExpanded && !query} animateOpacity>
            <VStack
              spacing={1}
              align="stretch"
              bg="white"
              borderRadius="xl"
              p={3}
              boxShadow="0 8px 32px rgba(0, 0, 0, 0.1)"
              maxH="400px"
              overflowY="auto"
            >
              <Text fontSize="xs" fontWeight="600" color="gray.500" px={2} mb={1}>
                SUGGESTED SEARCHES
              </Text>
              {suggestions.map((suggestion, index) => (
                <Box
                  key={index}
                  px={3}
                  py={2}
                  borderRadius="md"
                  cursor="pointer"
                  _hover={{ bg: 'purple.50' }}
                  transition="all 0.2s"
                  onClick={() => handleSearch(suggestion.text, suggestion.route)}
                >
                  <Text fontSize="sm" color="gray.700">
                    {suggestion.icon} {suggestion.text}
                  </Text>
                </Box>
              ))}
            </VStack>
          </Collapse>
        </VStack>
      </Box>
    </>
  );
};

export default FloatingSearchBar;