import { useState, useEffect } from 'react';
import { Box, VStack, HStack, Text, Icon, Badge, Avatar, Button, Image, Skeleton } from '@chakra-ui/react';
import { TrendingUp, Calendar, Users, ExternalLink, Sparkles, MapPin, Clock } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

const RightSidebar = () => {
  const navigate = useNavigate();
  const [trending, setTrending] = useState([]);
  const [events, setEvents] = useState([]);
  const [suggestions, setSuggestions] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchContent();
    const interval = setInterval(fetchContent, 60000);
    return () => clearInterval(interval);
  }, []);

  const fetchContent = async () => {
    try {
      const responses = await Promise.all([
        fetch('/api/feed/trending'),
        fetch('/api/feed/events'),
        fetch('/api/feed/suggestions')
      ]);
      
      const [trendingData, eventsData, suggestionsData] = await Promise.all(
        responses.map(r => r.ok ? r.json() : Promise.resolve([]))
      );

      setTrending(trendingData.length > 0 ? trendingData : mockTrending);
      setEvents(eventsData.length > 0 ? eventsData : mockEvents);
      setSuggestions(suggestionsData.length > 0 ? suggestionsData : mockSuggestions);
    } catch (error) {
      console.warn('Error fetching sidebar content, using fallback data:', error.message);
      setTrending(mockTrending);
      setEvents(mockEvents);
      setSuggestions(mockSuggestions);
    } finally {
      setLoading(false);
    }
  };

  const mockTrending = [
    { id: 1, tag: '#bitcoin', posts: '125K', trend: '+12%' },
    { id: 2, tag: '#uniswap', posts: '89K', trend: '+8%' },
    { id: 3, tag: '#lenster', posts: '67K', trend: '+24%' },
    { id: 4, tag: '#base', posts: '54K', trend: '+15%' },
    { id: 5, tag: '#nike', posts: '43K', trend: '+6%' },
  ];

  const mockEvents = [
    {
      id: 1,
      title: 'Monthly Talk: Defining DeFi in an Omnichain Future',
      time: 'Starting in 2 hours',
      attendees: '8,247+ more coming',
      category: 'Hangout',
      location: 'Virtual',
      image: null
    },
    {
      id: 2,
      title: 'AI & Design Workshop',
      time: 'Tomorrow at 3PM',
      attendees: '324+ interested',
      category: 'Workshop',
      location: 'San Francisco, CA',
      image: null
    },
    {
      id: 3,
      title: 'Coffee Meetup - Tech Professionals',
      time: 'This Weekend',
      attendees: '45+ going',
      category: 'Social',
      location: 'New York, NY',
      image: null
    }
  ];

  const mockSuggestions = [
    { id: 1, name: 'Alex Bishop', bio: 'Sweet, simple, repeat! 🔄✨', username: '@alexbishop', mutual: 4 },
    { id: 2, name: 'Bella Bean', bio: 'Steering every flavor into...', username: '@bellabea', mutual: 12 },
    { id: 3, name: 'Tyra Dhillon', bio: 'Style the way to impress w...', username: '@tyradhillon', mutual: 8 },
  ];

  return (
    <VStack spacing={4} align="stretch">
      <Box
        p={4}
        bg="black"
        borderRadius="xl"
        border="1px solid"
        borderColor="whiteAlpha.200"
        position="relative"
        overflow="hidden"
      >
        <Box
          position="absolute"
          top={0}
          right={0}
          bottom={0}
          left={0}
          bgGradient="linear(135deg, purple.500 0%, pink.500 100%)"
          opacity={0.1}
        />
        <VStack spacing={3} position="relative">
          <Box
            p={3}
            borderRadius="lg"
            bgGradient="linear(to-br, purple.500, pink.500)"
          >
            <Icon as={Sparkles} boxSize={6} color="white" />
          </Box>
          <Text fontSize="sm" fontWeight="bold" color="white" textAlign="center">
            Premium Ad Space
          </Text>
          <Text fontSize="xs" color="gray.400" textAlign="center">
            Your brand here - Reach 2.4M users
          </Text>
          <Button
            size="sm"
            colorScheme="purple"
            rightIcon={<ExternalLink size={14} />}
            _hover={{ transform: 'translateY(-2px)' }}
            transition="all 0.2s"
          >
            Advertise
          </Button>
        </VStack>
      </Box>

      <Box
        p={4}
        bg="black"
        borderRadius="xl"
        border="1px solid"
        borderColor="whiteAlpha.200"
      >
        <HStack justify="space-between" mb={4}>
          <HStack spacing={2}>
            <Icon as={TrendingUp} color="purple.400" boxSize={5} />
            <Text fontSize="sm" fontWeight="bold" color="white">
              Trending Topics
            </Text>
          </HStack>
          <Text fontSize="xs" color="purple.400" cursor="pointer" _hover={{ textDecoration: 'underline' }}>
            View All
          </Text>
        </HStack>

        <VStack spacing={3} align="stretch">
          {(loading ? Array(5).fill(0) : trending).map((topic, idx) => (
            <Box
              key={topic.id || idx}
              p={3}
              borderRadius="lg"
              cursor="pointer"
              transition="all 0.2s"
              _hover={{ bg: 'whiteAlpha.50' }}
            >
              {loading ? (
                <Skeleton height="40px" />
              ) : (
                <VStack align="start" spacing={1}>
                  <HStack justify="space-between" w="100%">
                    <Text fontSize="sm" fontWeight="bold" color="purple.400">
                      {topic.tag}
                    </Text>
                    <Badge colorScheme="green" fontSize="xs">
                      {topic.trend}
                    </Badge>
                  </HStack>
                  <Text fontSize="xs" color="gray.400">
                    {topic.posts} posts
                  </Text>
                </VStack>
              )}
            </Box>
          ))}
        </VStack>
      </Box>

      <Box
        p={4}
        bg="black"
        borderRadius="xl"
        border="1px solid"
        borderColor="whiteAlpha.200"
      >
        <HStack justify="space-between" mb={4}>
          <HStack spacing={2}>
            <Icon as={Calendar} color="pink.400" boxSize={5} />
            <Text fontSize="sm" fontWeight="bold" color="white">
              Upcoming Events
            </Text>
          </HStack>
          <Text fontSize="xs" color="pink.400" cursor="pointer" _hover={{ textDecoration: 'underline' }}>
            View All
          </Text>
        </HStack>

        <VStack spacing={3} align="stretch">
          {(loading ? Array(2).fill(0) : events.slice(0, 2)).map((event, idx) => (
            <Box
              key={event.id || idx}
              p={3}
              borderRadius="lg"
              bg="whiteAlpha.50"
              cursor="pointer"
              transition="all 0.2s"
              _hover={{ bg: 'whiteAlpha.100', transform: 'translateY(-2px)' }}
            >
              {loading ? (
                <Skeleton height="80px" />
              ) : (
                <VStack align="start" spacing={2}>
                  <HStack justify="space-between" w="100%">
                    <Badge colorScheme="pink" fontSize="xs">
                      {event.category}
                    </Badge>
                  </HStack>
                  <Text fontSize="sm" fontWeight="bold" color="white" noOfLines={2}>
                    {event.title}
                  </Text>
                  <HStack spacing={1}>
                    <Icon as={Clock} boxSize={3} color="gray.400" />
                    <Text fontSize="xs" color="gray.400">
                      {event.time}
                    </Text>
                  </HStack>
                  <HStack spacing={1}>
                    <Icon as={MapPin} boxSize={3} color="gray.400" />
                    <Text fontSize="xs" color="gray.400">
                      {event.location}
                    </Text>
                  </HStack>
                  <HStack spacing={1}>
                    <Icon as={Users} boxSize={3} color="gray.400" />
                    <Text fontSize="xs" color="gray.400">
                      {event.attendees}
                    </Text>
                  </HStack>
                </VStack>
              )}
            </Box>
          ))}
        </VStack>
      </Box>

      <Box
        p={4}
        bg="black"
        borderRadius="xl"
        border="1px solid"
        borderColor="whiteAlpha.200"
      >
        <HStack justify="space-between" mb={4}>
          <HStack spacing={2}>
            <Icon as={Users} color="blue.400" boxSize={5} />
            <Text fontSize="sm" fontWeight="bold" color="white">
              Suggestions
            </Text>
          </HStack>
          <Text fontSize="xs" color="blue.400" cursor="pointer" _hover={{ textDecoration: 'underline' }}>
            See all
          </Text>
        </HStack>

        <VStack spacing={3} align="stretch">
          {(loading ? Array(3).fill(0) : suggestions).map((person, idx) => (
            <HStack
              key={person.id || idx}
              spacing={3}
              p={2}
              borderRadius="lg"
              cursor="pointer"
              transition="all 0.2s"
              _hover={{ bg: 'whiteAlpha.50' }}
            >
              {loading ? (
                <Skeleton height="50px" width="100%" />
              ) : (
                <>
                  <Avatar
                    size="sm"
                    name={person.name}
                    bg="linear-gradient(135deg, #667eea 0%, #764ba2 100%)"
                  />
                  <VStack align="start" spacing={0} flex={1}>
                    <Text fontSize="xs" fontWeight="bold" color="white">
                      {person.name}
                    </Text>
                    <Text fontSize="xs" color="gray.400" noOfLines={1}>
                      {person.bio}
                    </Text>
                    <Text fontSize="xs" color="gray.500">
                      {person.mutual} mutual friends
                    </Text>
                  </VStack>
                  <Button size="xs" colorScheme="purple" variant="outline">
                    Follow
                  </Button>
                </>
              )}
            </HStack>
          ))}
        </VStack>
      </Box>

      <Box
        p={4}
        bg="black"
        borderRadius="xl"
        border="1px solid"
        borderColor="whiteAlpha.200"
        position="relative"
        overflow="hidden"
      >
        <Box
          position="absolute"
          top={0}
          right={0}
          bottom={0}
          left={0}
          bgGradient="linear(135deg, green.500 0%, teal.500 100%)"
          opacity={0.1}
        />
        <VStack spacing={3} position="relative">
          <Box
            p={3}
            borderRadius="lg"
            bgGradient="linear(to-br, green.500, teal.500)"
          >
            <Icon as={TrendingUp} boxSize={6} color="white" />
          </Box>
          <Text fontSize="sm" fontWeight="bold" color="white" textAlign="center">
            Sponsored Content
          </Text>
          <Text fontSize="xs" color="gray.400" textAlign="center">
            Featured partners & premium content
          </Text>
          <Button
            size="sm"
            colorScheme="green"
            rightIcon={<ExternalLink size={14} />}
            _hover={{ transform: 'translateY(-2px)' }}
            transition="all 0.2s"
          >
            Explore
          </Button>
        </VStack>
      </Box>

      <VStack spacing={2} pb={4}>
        <HStack spacing={2} flexWrap="wrap" justify="center">
          <Text fontSize="xs" color="gray.500" cursor="pointer" _hover={{ color: 'gray.400' }}>About</Text>
          <Text fontSize="xs" color="gray.600">•</Text>
          <Text fontSize="xs" color="gray.500" cursor="pointer" _hover={{ color: 'gray.400' }}>Accessibility</Text>
          <Text fontSize="xs" color="gray.600">•</Text>
          <Text fontSize="xs" color="gray.500" cursor="pointer" _hover={{ color: 'gray.400' }}>Help Center</Text>
        </HStack>
        <HStack spacing={2} flexWrap="wrap" justify="center">
          <Text fontSize="xs" color="gray.500" cursor="pointer" _hover={{ color: 'gray.400' }}>Privacy & Terms</Text>
          <Text fontSize="xs" color="gray.600">•</Text>
          <Text fontSize="xs" color="gray.500" cursor="pointer" _hover={{ color: 'gray.400' }}>Advertising</Text>
        </HStack>
        <Text fontSize="xs" color="gray.600" textAlign="center">
          © 2025 Pollen AI Platform
        </Text>
      </VStack>
    </VStack>
  );
};

export default RightSidebar;
