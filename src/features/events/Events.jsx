import { useState, useEffect } from 'react';
import {
  Box,
  Heading,
  VStack,
  Text,
  HStack,
  Icon,
  Badge,
  Select,
  SimpleGrid,
  Spinner,
  useToast,
  Link,
  Button
} from '@chakra-ui/react';
import { Calendar, MapPin, ExternalLink, TrendingUp, RefreshCw } from 'lucide-react';
import { API_BASE_URL } from '@utils/constants';

export default function Events() {
  const [events, setEvents] = useState([]);
  const [categories, setCategories] = useState([]);
  const [selectedCategory, setSelectedCategory] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [status, setStatus] = useState('');
  const toast = useToast();

  useEffect(() => {
    fetch(`${API_BASE_URL}/events/categories`)
      .then(res => res.json())
      .then(data => setCategories(data.categories || []))
      .catch(() => {});
    
    loadEvents();
  }, []);

  const loadEvents = () => {
    setIsLoading(true);
    setEvents([]);
    setStatus('Loading events...');

    const params = new URLSearchParams();
    if (selectedCategory) params.append('category', selectedCategory);
    params.append('max_results', '20');

    const eventSource = new EventSource(`${API_BASE_URL}/events/upcoming?${params}`);

    eventSource.onmessage = (event) => {
      try {
        const parsed = JSON.parse(event.data);
        
        if (parsed.type === 'status') {
          setStatus(parsed.message);
        } else if (parsed.type === 'event') {
          setEvents(prev => [...prev, parsed.data]);
        } else if (parsed.type === 'complete') {
          setStatus('');
          setIsLoading(false);
          toast({
            title: parsed.message,
            status: 'success',
            duration: 2000,
          });
        } else if (parsed.type === 'error') {
          setStatus('');
          setIsLoading(false);
          toast({
            title: 'Error loading events',
            description: parsed.error,
            status: 'error',
            duration: 4000,
          });
        }
      } catch (e) {}
    };

    eventSource.onerror = () => {
      eventSource.close();
      setIsLoading(false);
      setStatus('');
    };

    return () => eventSource.close();
  };

  const formatDate = (dateString) => {
    try {
      const date = new Date(dateString);
      return date.toLocaleDateString('en-US', { 
        month: 'short', 
        day: 'numeric', 
        year: 'numeric' 
      });
    } catch {
      return dateString;
    }
  };

  return (
    <Box px={4} py={6}>
      <VStack align="start" spacing={5}>
        <Box
          w="100%"
          p={6}
          borderRadius="2xl"
          bgGradient="linear(135deg, blue.600, purple.600)"
          color="white"
          position="relative"
          overflow="hidden"
        >
          <Box position="absolute" right="-20px" top="-20px" opacity={0.2}>
            <Calendar size={120} />
          </Box>
          <VStack align="start" spacing={2} position="relative">
            <HStack>
              <Icon as={Calendar} boxSize={8} />
              <Heading size="lg">Upcoming Events</Heading>
            </HStack>
            <Text fontSize="sm" opacity={0.9}>
              Discover trending conferences, meetups, and industry events
            </Text>
          </VStack>
        </Box>

        <HStack w="100%" spacing={3}>
          <Select
            value={selectedCategory}
            onChange={(e) => setSelectedCategory(e.target.value)}
            bg="whiteAlpha.100"
            backdropFilter="blur(10px)"
            border="1px solid"
            borderColor="whiteAlpha.300"
            color="white"
            borderRadius="xl"
            _focus={{
              bg: 'whiteAlpha.150',
              borderColor: 'blue.400',
            }}
          >
            <option value="" style={{ background: '#1a202c' }}>All Categories</option>
            {categories.map(cat => (
              <option key={cat} value={cat.toLowerCase()} style={{ background: '#1a202c' }}>
                {cat}
              </option>
            ))}
          </Select>
          <Button
            onClick={loadEvents}
            isLoading={isLoading}
            colorScheme="blue"
            leftIcon={<RefreshCw size={16} />}
            size="md"
          >
            Refresh
          </Button>
        </HStack>

        {status && (
          <HStack w="100%" justify="center" p={4}>
            <Spinner size="sm" color="blue.400" />
            <Text fontSize="sm" color="gray.400">{status}</Text>
          </HStack>
        )}

        {events.length > 0 && (
          <SimpleGrid columns={{ base: 1, md: 2, lg: 3 }} spacing={4} w="100%">
            {events.map((event, idx) => (
              <Box
                key={idx}
                p={5}
                bg="whiteAlpha.100"
                backdropFilter="blur(10px)"
                borderRadius="xl"
                border="1px solid"
                borderColor="whiteAlpha.200"
                transition="all 0.2s"
                _hover={{
                  transform: 'translateY(-4px)',
                  borderColor: 'blue.400',
                  boxShadow: '0 8px 24px rgba(66, 153, 225, 0.15)',
                }}
              >
                <VStack align="start" spacing={3}>
                  <HStack justify="space-between" w="100%">
                    <Badge
                      colorScheme="blue"
                      fontSize="xs"
                      px={2}
                      py={1}
                      borderRadius="md"
                    >
                      {event.category || 'General'}
                    </Badge>
                    {event.adaptive_score && (
                      <HStack spacing={1}>
                        <TrendingUp size={14} color="var(--chakra-colors-blue-400)" />
                        <Text fontSize="xs" color="blue.400" fontWeight="bold">
                          {Math.round(event.adaptive_score.overall)}
                        </Text>
                      </HStack>
                    )}
                  </HStack>

                  <Heading size="sm" color="white" lineHeight="1.3">
                    {event.title}
                  </Heading>

                  <Text fontSize="sm" color="gray.400" noOfLines={2}>
                    {event.description}
                  </Text>

                  <VStack align="start" spacing={1} w="100%">
                    {event.date && (
                      <HStack spacing={2}>
                        <Icon as={Calendar} boxSize={4} color="gray.500" />
                        <Text fontSize="xs" color="gray.500">
                          {formatDate(event.date)}
                        </Text>
                      </HStack>
                    )}
                    {event.location && (
                      <HStack spacing={2}>
                        <Icon as={MapPin} boxSize={4} color="gray.500" />
                        <Text fontSize="xs" color="gray.500">
                          {event.location}
                        </Text>
                      </HStack>
                    )}
                  </VStack>

                  {event.url && (
                    <Link href={event.url} isExternal w="100%">
                      <Button
                        size="sm"
                        w="100%"
                        variant="outline"
                        colorScheme="blue"
                        rightIcon={<ExternalLink size={14} />}
                      >
                        Learn More
                      </Button>
                    </Link>
                  )}

                  <HStack justify="space-between" w="100%" pt={2} borderTop="1px solid" borderColor="whiteAlpha.200">
                    <Text fontSize="xs" color="gray.600">
                      {event.source || 'Event Source'}
                    </Text>
                  </HStack>
                </VStack>
              </Box>
            ))}
          </SimpleGrid>
        )}

        {!isLoading && events.length === 0 && (
          <Box
            w="100%"
            p={8}
            textAlign="center"
            bg="whiteAlpha.50"
            borderRadius="xl"
            border="1px dashed"
            borderColor="whiteAlpha.300"
          >
            <Calendar size={48} style={{ margin: '0 auto 16px', opacity: 0.3 }} />
            <Text color="gray.500">No events found. Try a different category.</Text>
          </Box>
        )}
      </VStack>
    </Box>
  );
}
