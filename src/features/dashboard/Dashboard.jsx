import { useState, useEffect } from 'react';
import { Box, SimpleGrid, Text, VStack, HStack, Badge, Icon } from '@chakra-ui/react';
import { FEATURES } from '@utils/constants';
import FeatureCard from '@components/common/FeatureCard';
import { Clock, TrendingUp, Sparkles } from 'lucide-react';

const Dashboard = () => {
  const [greeting, setGreeting] = useState('');

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
    <Box px={4} py={2}>
      <VStack align="start" spacing={6}>
        <Box width="100%">
          <Text 
            fontSize="2xl" 
            fontWeight="bold" 
            color="white" 
            mb={1}
            display={{ base: 'block', sm: 'none' }}
          >
            {greeting}!
          </Text>
          <Text fontSize="md" color="gray.400" mb={6}>
            What can I help you with today?
          </Text>

          <SimpleGrid columns={2} spacing={4}>
            {FEATURES.map((feature) => (
              <FeatureCard key={feature.id} feature={feature} />
            ))}
          </SimpleGrid>
        </Box>

        <Box width="100%" mt={2}>
          <HStack spacing={2} mb={3}>
            <Icon as={Clock} color="purple.400" boxSize={5} />
            <Text fontSize="md" fontWeight="semibold" color="white">
              Recent Activity
            </Text>
          </HStack>
          <VStack spacing={3}>
            {recentActivities.map((activity) => (
              <Box
                key={activity.id}
                w="100%"
                p={4}
                bg="whiteAlpha.100"
                backdropFilter="blur(10px)"
                borderRadius="xl"
                border="1px solid"
                borderColor="whiteAlpha.200"
                cursor="pointer"
                transition="all 0.2s"
                _hover={{ bg: 'whiteAlpha.200', transform: 'translateY(-2px)', boxShadow: 'lg' }}
              >
                <HStack justify="space-between">
                  <VStack align="start" spacing={1}>
                    <Text fontSize="sm" fontWeight="medium" color="white">
                      {activity.title}
                    </Text>
                    <Text fontSize="xs" color="gray.400">
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

        <Box width="100%">
          <HStack spacing={2} mb={3}>
            <Icon as={TrendingUp} color="orange.400" boxSize={5} />
            <Text fontSize="md" fontWeight="semibold" color="white">
              Trending Now
            </Text>
          </HStack>
          <VStack spacing={3}>
            {trendingTopics.map((topic) => (
              <Box
                key={topic.id}
                w="100%"
                p={4}
                bg="whiteAlpha.100"
                backdropFilter="blur(10px)"
                borderRadius="xl"
                border="1px solid"
                borderColor="whiteAlpha.200"
                cursor="pointer"
                transition="all 0.2s"
                _hover={{ bg: 'whiteAlpha.200', transform: 'translateY(-2px)', boxShadow: 'lg' }}
              >
                <HStack justify="space-between" align="start">
                  <VStack align="start" spacing={1} flex={1}>
                    <HStack>
                      <Text fontSize="sm" fontWeight="medium" color="white">
                        {topic.title}
                      </Text>
                      {topic.hot && <Icon as={Sparkles} color="orange.400" boxSize={4} />}
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
          borderRadius="2xl" 
          mt={2}
          bgGradient="linear-gradient(135deg, #667eea 0%, #764ba2 100%)"
          boxShadow="lg"
        >
          <VStack spacing={3} align="center" color="white">
            <Icon as={Sparkles} boxSize={12} />
            <Text fontSize="xl" fontWeight="bold" textAlign="center">
              Powered by Pollen AI
            </Text>
            <Text fontSize="sm" textAlign="center" opacity={0.95}>
              Your privacy-first AI assistant that learns and adapts to your needs
            </Text>
          </VStack>
        </Box>
      </VStack>
    </Box>
  );
};

export default Dashboard;