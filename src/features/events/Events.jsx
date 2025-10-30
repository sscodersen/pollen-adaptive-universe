import { useState, useEffect, useRef } from 'react';
import {
  Box,
  Heading,
  VStack,
  Text,
  HStack,
  Icon,
  Select,
  Spinner,
  useToast,
  Button
} from '@chakra-ui/react';
import { Calendar, RefreshCw } from 'lucide-react';
import { API_BASE_URL } from '@utils/constants';
import PostCard from '@components/common/PostCard';

export default function Events() {
  const [events, setEvents] = useState([]);
  const [categories, setCategories] = useState([]);
  const [selectedCategory, setSelectedCategory] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [status, setStatus] = useState('');
  const toast = useToast();
  const eventSourceRef = useRef(null);

  useEffect(() => {
    fetch(`${API_BASE_URL}/api/events/categories`)
      .then(res => res.json())
      .then(data => setCategories(data.categories || []))
      .catch(() => {});
    
    const cleanup = loadEvents();
    return cleanup;
  }, [selectedCategory]);

  const loadEvents = () => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }

    setIsLoading(true);
    setEvents([]);
    setStatus('Loading events...');

    const params = new URLSearchParams();
    if (selectedCategory) params.append('category', selectedCategory);
    params.append('max_results', '20');

    const eventSource = new EventSource(`${API_BASE_URL}/api/events/upcoming?${params}`);
    eventSourceRef.current = eventSource;

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
          eventSource.close();
          eventSourceRef.current = null;
          toast({
            title: parsed.message,
            status: 'success',
            duration: 2000,
          });
        } else if (parsed.type === 'error') {
          setStatus('');
          setIsLoading(false);
          eventSource.close();
          eventSourceRef.current = null;
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
      eventSourceRef.current = null;
      setIsLoading(false);
      setStatus('');
    };

    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
        eventSourceRef.current = null;
      }
    };
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
          <VStack spacing={4} w="100%">
            {events.map((event, idx) => (
              <PostCard 
                key={idx} 
                post={{
                  ...event,
                  content: event.title,
                  tags: [
                    event.category,
                    event.date && `📅 ${formatDate(event.date)}`,
                    event.location && `📍 ${event.location}`
                  ].filter(Boolean)
                }} 
                showImage={true} 
              />
            ))}
          </VStack>
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
