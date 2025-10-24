import { useState, useEffect } from 'react';
import { Box, SimpleGrid, Text, VStack, HStack, Avatar, Badge, Icon } from '@chakra-ui/react';
import { useLocation } from 'react-router-dom';
import { FEATURES } from '@utils/constants';
import FeatureCard from '@components/common/FeatureCard';
import SearchBar from '@components/common/SearchBar';
import { Clock, TrendingUp, Sparkles } from 'lucide-react';

const Dashboard = () => {
  const location = useLocation();
  const [greeting, setGreeting] = useState('');
  const [userName] = useState('Jane');

  useEffect(() => {
    const hour = new Date().getHours();
    if (hour < 12) setGreeting('Good morning');
    else if (hour < 18) setGreeting('Good afternoon');
    else setGreeting('Good evening');
  }, []);

  const recentActivities = [
    { id: 1, title: 'Shopping recommendations', time: '2 hours ago', type: 'shopping' },
    { id: 2, title: 'Travel planning to Tokyo', time: 'Yesterday', type: 'travel' },
    { id: 3, title: 'Health & wellness tips', time: '2 days ago', type: 'health' },
  ];

  const trendingTopics = [
    { id: 1, title: 'AI-powered home automation', category: 'Smart Home', hot: true },
    { id: 2, title: 'Sustainable travel tips', category: 'Travel', hot: false },
    { id: 3, title: 'Productivity hacks for 2025', category: 'Education', hot: true },
  ];

  return (
    <Box px={4} py={4}>
      <SearchBar placeholder="What can I help you with today?" />

      <VStack align="start" spacing={6} mt={6}>
        <Box width="100%">
          <HStack spacing={4} mb={6}>
            <Avatar 
              name={userName} 
              bg="purple.500" 
              color="white" 
              size="lg"
            />
            <VStack align="start" spacing={0}>
              <Text fontSize="2xl" fontWeight="bold" color="gray.800">
                {greeting}, {userName}!
              </Text>
              <Text fontSize="sm" color="gray.600">
                Welcome back to your AI-powered hub
              </Text>
            </VStack>
          </HStack>

          <Text fontSize="lg" fontWeight="bold" color="gray.800" mb={4} mt={6}>
            Your AI Assistant
          </Text>
          <SimpleGrid columns={2} spacing={4}>
            {FEATURES.map((feature) => (
              <FeatureCard key={feature.id} feature={feature} />
            ))}
          </SimpleGrid>
        </Box>

        <Box width="100%" mt={4}>
          <HStack spacing={2} mb={3}>
            <Icon as={Clock} color="purple.500" />
            <Text fontSize="md" fontWeight="semibold" color="gray.700">
              Recent Activity
            </Text>
          </HStack>
          <VStack spacing={3}>
            {recentActivities.map((activity) => (
              <Box
                key={activity.id}
                w="100%"
                p={4}
                bg="whiteAlpha.800"
                backdropFilter="blur(10px)"
                borderRadius="xl"
                border="1px solid"
                borderColor="whiteAlpha.400"
                cursor="pointer"
                transition="all 0.2s"
                _hover={{ bg: 'whiteAlpha.900', transform: 'translateY(-2px)' }}
              >
                <HStack justify="space-between">
                  <VStack align="start" spacing={1}>
                    <Text fontSize="sm" fontWeight="medium" color="gray.800">
                      {activity.title}
                    </Text>
                    <Text fontSize="xs" color="gray.600">
                      {activity.time}
                    </Text>
                  </VStack>
                  <Badge colorScheme="purple" fontSize="xs">
                    {activity.type}
                  </Badge>
                </HStack>
              </Box>
            ))}
          </VStack>
        </Box>

        <Box width="100%" mt={4}>
          <HStack spacing={2} mb={3}>
            <Icon as={TrendingUp} color="orange.500" />
            <Text fontSize="md" fontWeight="semibold" color="gray.700">
              Trending Now
            </Text>
          </HStack>
          <VStack spacing={3}>
            {trendingTopics.map((topic) => (
              <Box
                key={topic.id}
                w="100%"
                p={4}
                bg="whiteAlpha.800"
                backdropFilter="blur(10px)"
                borderRadius="xl"
                border="1px solid"
                borderColor="whiteAlpha.400"
                cursor="pointer"
                transition="all 0.2s"
                _hover={{ bg: 'whiteAlpha.900', transform: 'translateY(-2px)' }}
              >
                <HStack justify="space-between">
                  <VStack align="start" spacing={1}>
                    <HStack>
                      <Text fontSize="sm" fontWeight="medium" color="gray.800">
                        {topic.title}
                      </Text>
                      {topic.hot && <Icon as={Sparkles} color="orange.500" boxSize={3} />}
                    </HStack>
                    <Badge colorScheme="blue" fontSize="xs">
                      {topic.category}
                    </Badge>
                  </VStack>
                </HStack>
              </Box>
            ))}
          </VStack>
        </Box>

        <Box 
          width="100%" 
          p={6} 
          bg="purple.500" 
          borderRadius="2xl" 
          mt={4}
          bgGradient="linear-gradient(135deg, #667eea 0%, #764ba2 100%)"
        >
          <VStack spacing={3} align="center" color="white">
            <Icon as={Sparkles} boxSize={10} />
            <Text fontSize="lg" fontWeight="bold" textAlign="center">
              Powered by Pollen AI
            </Text>
            <Text fontSize="sm" textAlign="center" opacity={0.9}>
              Your privacy-first AI assistant that learns and adapts to your needs
            </Text>
          </VStack>
        </Box>
      </VStack>
    </Box>
  );
};

export default Dashboard;