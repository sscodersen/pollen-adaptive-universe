import { useState } from 'react';
import {
  Box,
  VStack,
  HStack,
  Text,
  SimpleGrid,
  Badge,
  Icon,
  Tabs,
  TabList,
  TabPanels,
  Tab,
  TabPanel,
  Button,
  Avatar,
  AvatarGroup
} from '@chakra-ui/react';
import { useNavigate } from 'react-router-dom';
import { 
  TrendingUp, 
  Sparkles, 
  Clock, 
  Star,
  Zap,
  ShoppingBag,
  Plane,
  Newspaper,
  FileText,
  Home as HomeIcon,
  Heart,
  GraduationCap
} from 'lucide-react';
import FeatureCard from '@components/common/FeatureCard';
import { FEATURES } from '@utils/constants';

const Explore = () => {
  const navigate = useNavigate();
  const [activeCategory, setActiveCategory] = useState('all');

  const categories = [
    { id: 'all', name: 'All Features', icon: Sparkles },
    { id: 'productivity', name: 'Productivity', icon: Zap },
    { id: 'lifestyle', name: 'Lifestyle', icon: Star },
    { id: 'insights', name: 'Insights', icon: TrendingUp },
  ];

  const trendingNow = [
    {
      id: 1,
      title: 'AI-Powered Travel Planning',
      category: 'Travel',
      users: 1247,
      icon: Plane,
      gradient: 'linear(to-r, cyan.500, blue.600)',
      route: '/travel'
    },
    {
      id: 2,
      title: 'Smart Shopping Assistant',
      category: 'Shopping',
      users: 2156,
      icon: ShoppingBag,
      gradient: 'linear(to-r, purple.500, purple.600)',
      route: '/shopping'
    },
    {
      id: 3,
      title: 'Content Generation',
      category: 'Create',
      users: 989,
      icon: FileText,
      gradient: 'linear(to-r, orange.500, orange.600)',
      route: '/content'
    },
  ];

  const quickActions = [
    {
      title: 'Find Best Deals',
      subtitle: 'Shop smarter with AI',
      icon: ShoppingBag,
      route: '/shopping',
      color: 'purple'
    },
    {
      title: 'Plan a Trip',
      subtitle: 'Your next adventure',
      icon: Plane,
      route: '/travel',
      color: 'cyan'
    },
    {
      title: 'Get Latest News',
      subtitle: 'Stay informed',
      icon: Newspaper,
      route: '/news',
      color: 'pink'
    },
    {
      title: 'Generate Content',
      subtitle: 'Create with AI',
      icon: FileText,
      route: '/content',
      color: 'orange'
    },
  ];

  const recentlyActive = [
    { name: 'User', avatar: 'U', feature: 'Travel Planning', time: '2m ago' },
    { name: 'User', avatar: 'U', feature: 'Content Gen', time: '5m ago' },
    { name: 'User', avatar: 'U', feature: 'Shopping', time: '12m ago' },
    { name: 'User', avatar: 'U', feature: 'Health Tips', time: '18m ago' },
  ];

  return (
    <Box px={4} py={4}>
      <VStack align="start" spacing={6}>
        {/* Header */}
        <VStack align="start" spacing={2} w="100%">
          <HStack spacing={3}>
            <Icon as={Sparkles} boxSize={8} color="purple.400" />
            <Text fontSize="3xl" fontWeight="bold" color="white">
              Explore
            </Text>
          </HStack>
          <Text fontSize="sm" color="gray.400">
            Discover AI-powered features to enhance your life
          </Text>
        </VStack>

        {/* Trending Now */}
        <Box w="100%">
          <HStack spacing={2} mb={4}>
            <Icon as={TrendingUp} color="orange.400" boxSize={5} />
            <Text fontSize="lg" fontWeight="semibold" color="white">
              Trending Now
            </Text>
            <Badge colorScheme="orange" fontSize="xs">HOT</Badge>
          </HStack>
          <VStack spacing={3} w="100%">
            {trendingNow.map((item) => {
              const ItemIcon = item.icon;
              return (
                <Box
                  key={item.id}
                  w="100%"
                  p={4}
                  bg="whiteAlpha.100"
                  backdropFilter="blur(10px)"
                  borderRadius="xl"
                  border="1px solid"
                  borderColor="whiteAlpha.200"
                  cursor="pointer"
                  transition="all 0.3s"
                  _hover={{
                    transform: 'translateX(8px)',
                    bg: 'whiteAlpha.150',
                    borderColor: 'whiteAlpha.300'
                  }}
                  onClick={() => navigate(item.route)}
                >
                  <HStack justify="space-between">
                    <HStack spacing={3}>
                      <Box
                        p={2}
                        borderRadius="lg"
                        bgGradient={item.gradient}
                      >
                        <Icon as={ItemIcon} boxSize={5} color="white" />
                      </Box>
                      <VStack align="start" spacing={0}>
                        <Text fontSize="md" fontWeight="semibold" color="white">
                          {item.title}
                        </Text>
                        <HStack spacing={2}>
                          <Badge colorScheme="blue" fontSize="xs">
                            {item.category}
                          </Badge>
                          <HStack spacing={1}>
                            <AvatarGroup size="xs" max={3}>
                              <Avatar name="User 1" />
                              <Avatar name="User 2" />
                              <Avatar name="User 3" />
                            </AvatarGroup>
                            <Text fontSize="xs" color="gray.500">
                              {item.users.toLocaleString()} active
                            </Text>
                          </HStack>
                        </HStack>
                      </VStack>
                    </HStack>
                    <Icon as={Sparkles} boxSize={5} color="orange.400" />
                  </HStack>
                </Box>
              );
            })}
          </VStack>
        </Box>

        {/* Quick Actions */}
        <Box w="100%">
          <HStack spacing={2} mb={4}>
            <Icon as={Zap} color="yellow.400" boxSize={5} />
            <Text fontSize="lg" fontWeight="semibold" color="white">
              Quick Actions
            </Text>
          </HStack>
          <SimpleGrid columns={2} spacing={3}>
            {quickActions.map((action) => {
              const ActionIcon = action.icon;
              return (
                <Box
                  key={action.title}
                  p={4}
                  bg="whiteAlpha.100"
                  backdropFilter="blur(10px)"
                  borderRadius="xl"
                  border="1px solid"
                  borderColor="whiteAlpha.200"
                  cursor="pointer"
                  transition="all 0.2s"
                  _hover={{
                    transform: 'translateY(-4px)',
                    bg: 'whiteAlpha.150',
                    boxShadow: 'lg'
                  }}
                  onClick={() => navigate(action.route)}
                >
                  <VStack align="start" spacing={2}>
                    <Icon as={ActionIcon} boxSize={6} color={`${action.color}.400`} />
                    <Text fontSize="sm" fontWeight="semibold" color="white">
                      {action.title}
                    </Text>
                    <Text fontSize="xs" color="gray.500">
                      {action.subtitle}
                    </Text>
                  </VStack>
                </Box>
              );
            })}
          </SimpleGrid>
        </Box>

        {/* All Features */}
        <Box w="100%">
          <HStack spacing={2} mb={4}>
            <Icon as={Star} color="purple.400" boxSize={5} />
            <Text fontSize="lg" fontWeight="semibold" color="white">
              All Features
            </Text>
          </HStack>
          <SimpleGrid columns={2} spacing={4}>
            {FEATURES.map((feature) => (
              <FeatureCard key={feature.id} feature={feature} />
            ))}
          </SimpleGrid>
        </Box>

        {/* Recently Active */}
        <Box w="100%" pb={4}>
          <HStack spacing={2} mb={4}>
            <Icon as={Clock} color="cyan.400" boxSize={5} />
            <Text fontSize="lg" fontWeight="semibold" color="white">
              Platform Activity
            </Text>
            <Badge colorScheme="green" fontSize="xs">Anonymous</Badge>
          </HStack>
          <VStack spacing={3} w="100%">
            {recentlyActive.map((user, index) => (
              <Box
                key={index}
                w="100%"
                p={3}
                bg="whiteAlpha.100"
                backdropFilter="blur(10px)"
                borderRadius="lg"
                border="1px solid"
                borderColor="whiteAlpha.200"
              >
                <HStack justify="space-between">
                  <HStack spacing={3}>
                    <Avatar size="sm" name={user.name} bg="purple.500" />
                    <VStack align="start" spacing={0}>
                      <Text fontSize="sm" fontWeight="medium" color="white">
                        {user.name} used {user.feature}
                      </Text>
                      <Text fontSize="xs" color="gray.500">
                        {user.time}
                      </Text>
                    </VStack>
                  </HStack>
                  <Icon as={Sparkles} boxSize={4} color="purple.400" />
                </HStack>
              </Box>
            ))}
          </VStack>
        </Box>
      </VStack>
    </Box>
  );
};

export default Explore;
