import { useState, useRef, useEffect } from 'react';
import { 
  Box, 
  Input, 
  InputGroup, 
  InputLeftElement,
  InputRightElement,
  IconButton,
  VStack,
  Text,
  HStack,
  useToast,
  Portal,
  keyframes
} from '@chakra-ui/react';
import { Search, Sparkles, X, Command } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

const gradientAnimation = keyframes`
  0% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
`;

const UnifiedSearchBar = ({ 
  placeholder = 'Ask Pollen AI anything...',
  onSearch,
  autoFocus = false,
  variant = 'floating',
  showSuggestions = true
}) => {
  const [query, setQuery] = useState('');
  const [isExpanded, setIsExpanded] = useState(false);
  const [isFocused, setIsFocused] = useState(false);
  const inputRef = useRef(null);
  const navigate = useNavigate();
  const toast = useToast();

  const suggestions = [
    { text: 'Find best laptops under $1000', route: '/shopping', icon: '🛍️', gradient: 'linear(to-r, purple.400, pink.400)' },
    { text: 'Plan a trip to Tokyo', route: '/travel', icon: '✈️', gradient: 'linear(to-r, cyan.400, blue.400)' },
    { text: 'Latest tech news', route: '/news', icon: '📰', gradient: 'linear(to-r, pink.400, red.400)' },
    { text: 'Generate blog post ideas', route: '/content', icon: '✍️', gradient: 'linear(to-r, orange.400, yellow.400)' },
    { text: 'Control smart lights', route: '/smarthome', icon: '💡', gradient: 'linear(to-r, green.400, teal.400)' },
    { text: 'Healthy breakfast ideas', route: '/health', icon: '🥗', gradient: 'linear(to-r, red.400, pink.400)' },
    { text: 'Learn Python basics', route: '/education', icon: '📚', gradient: 'linear(to-r, purple.400, violet.400)' },
  ];

  useEffect(() => {
    const handleKeyDown = (e) => {
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

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

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

    if (onSearch) {
      onSearch(searchQuery);
      setIsExpanded(false);
      setQuery('');
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

  const handleSuggestionClick = (suggestion) => {
    handleSearch(suggestion.text, suggestion.route);
  };

  if (variant === 'floating') {
    return (
      <>
        {isExpanded && (
          <Portal>
            <Box
              position="fixed"
              top="0"
              left="0"
              right="0"
              bottom="0"
              bg="blackAlpha.700"
              backdropFilter="blur(8px)"
              zIndex="9999"
              onClick={() => setIsExpanded(false)}
            >
              <Box
                position="absolute"
                top="20%"
                left="50%"
                transform="translateX(-50%)"
                w={{ base: '90%', md: '600px' }}
                onClick={(e) => e.stopPropagation()}
              >
                <VStack spacing={4} align="stretch">
                  <Box
                    position="relative"
                    p="2px"
                    borderRadius="full"
                    bgGradient="linear(to-r, #667eea, #764ba2, #f093fb, #4facfe, #00f2fe, #43e97b)"
                    backgroundSize="300% 300%"
                    animation={`${gradientAnimation} 6s ease infinite`}
                  >
                    <InputGroup size="lg">
                      <InputLeftElement pointerEvents="none" color="gray.400" h="full">
                        <Sparkles size={20} />
                      </InputLeftElement>
                      <Input
                        ref={inputRef}
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
                        placeholder={placeholder}
                        bg="gray.900"
                        color="white"
                        border="none"
                        borderRadius="full"
                        fontSize="md"
                        h="56px"
                        _placeholder={{ color: 'gray.400' }}
                        _focus={{ outline: 'none' }}
                        pl="48px"
                        pr="48px"
                      />
                      <InputRightElement h="full">
                        <IconButton
                          icon={<X size={18} />}
                          onClick={() => {
                            setIsExpanded(false);
                            setQuery('');
                          }}
                          size="sm"
                          variant="ghost"
                          color="gray.400"
                          _hover={{ color: 'white', bg: 'whiteAlpha.200' }}
                          borderRadius="full"
                        />
                      </InputRightElement>
                    </InputGroup>
                  </Box>

                  {showSuggestions && query.length === 0 && (
                    <VStack
                      spacing={2}
                      align="stretch"
                      bg="gray.900"
                      p={4}
                      borderRadius="xl"
                      border="1px solid"
                      borderColor="whiteAlpha.200"
                    >
                      <Text fontSize="xs" color="gray.400" fontWeight="medium" px={2}>
                        SUGGESTED SEARCHES
                      </Text>
                      {suggestions.map((suggestion, index) => (
                        <HStack
                          key={index}
                          p={3}
                          cursor="pointer"
                          borderRadius="lg"
                          transition="all 0.2s"
                          _hover={{ 
                            bg: 'whiteAlpha.100',
                            transform: 'translateX(4px)'
                          }}
                          onClick={() => handleSuggestionClick(suggestion)}
                        >
                          <Box fontSize="20px">{suggestion.icon}</Box>
                          <Text fontSize="sm" color="white" flex="1">
                            {suggestion.text}
                          </Text>
                        </HStack>
                      ))}
                    </VStack>
                  )}
                </VStack>
              </Box>
            </Box>
          </Portal>
        )}

        <Box
          position="fixed"
          bottom="24px"
          left="50%"
          transform="translateX(-50%)"
          zIndex="1000"
          cursor="pointer"
          onClick={() => setIsExpanded(true)}
        >
          <Box
            position="relative"
            p="2px"
            borderRadius="full"
            bgGradient="linear(to-r, #667eea, #764ba2, #f093fb, #4facfe)"
            backgroundSize="200% 200%"
            animation={`${gradientAnimation} 4s ease infinite`}
            transition="all 0.3s"
            _hover={{ transform: 'scale(1.05)' }}
          >
            <HStack
              bg="gray.900"
              color="white"
              px={6}
              py={3}
              borderRadius="full"
              spacing={2}
            >
              <Sparkles size={20} />
              <Text fontSize="sm" fontWeight="medium">
                Ask AI
              </Text>
              <HStack spacing={1} opacity={0.7} display={{ base: 'none', md: 'flex' }}>
                <Box
                  as="kbd"
                  px={1.5}
                  py={0.5}
                  bg="whiteAlpha.200"
                  borderRadius="md"
                  fontSize="xs"
                >
                  <Command size={12} style={{ display: 'inline' }} />K
                </Box>
              </HStack>
            </HStack>
          </Box>
        </Box>
      </>
    );
  }

  return (
    <Box
      position="relative"
      p="2px"
      borderRadius="full"
      bgGradient={
        isFocused
          ? "linear(to-r, #667eea, #764ba2, #f093fb, #4facfe)"
          : "linear(to-r, gray.300, gray.400)"
      }
      backgroundSize="200% 200%"
      animation={isFocused ? `${gradientAnimation} 4s ease infinite` : 'none'}
      transition="all 0.3s"
      w="100%"
    >
      <InputGroup size="lg">
        <InputLeftElement pointerEvents="none" color="gray.500" h="full">
          <Search size={20} />
        </InputLeftElement>
        <Input
          ref={inputRef}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
          onFocus={() => setIsFocused(true)}
          onBlur={() => setIsFocused(false)}
          placeholder={placeholder}
          bg="white"
          color="gray.900"
          border="none"
          borderRadius="full"
          fontSize="md"
          h="52px"
          autoFocus={autoFocus}
          _placeholder={{ color: 'gray.500' }}
          _focus={{ outline: 'none' }}
          pl="48px"
          pr={query ? '48px' : '16px'}
        />
        {query && (
          <InputRightElement h="full">
            <IconButton
              icon={<X size={18} />}
              onClick={() => setQuery('')}
              size="sm"
              variant="ghost"
              color="gray.500"
              _hover={{ color: 'gray.700', bg: 'gray.100' }}
              borderRadius="full"
            />
          </InputRightElement>
        )}
      </InputGroup>
    </Box>
  );
};

export default UnifiedSearchBar;
