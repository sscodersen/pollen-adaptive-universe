import { useState } from 'react';
import {
  Box,
  VStack,
  HStack,
  Text,
  Avatar,
  Badge,
  Icon,
  Tabs,
  TabList,
  Tab,
  TabPanels,
  TabPanel,
  Button,
  Image
} from '@chakra-ui/react';
import { formatDistanceToNow } from 'date-fns';
import {
  Activity as ActivityIcon,
  MessageSquare,
  ShoppingBag,
  Plane,
  Newspaper,
  Sparkles,
  Heart,
  Home as HomeIcon,
  GraduationCap,
  ThumbsUp,
  Share2,
  Bookmark
} from 'lucide-react';

const Activity = () => {
  const [selectedTab, setSelectedTab] = useState(0);

  const activities = [
    {
      id: 1,
      type: 'ai_interaction',
      agent: 'Shopping',
      icon: ShoppingBag,
      color: 'purple.400',
      title: 'Shopping Assistant helped you find laptops',
      description: 'You asked about "best laptops under $1000" and received 5 product recommendations with detailed comparisons.',
      time: new Date(Date.now() - 7200000),
      engagement: { likes: 12, comments: 3, saved: true },
      preview: {
        type: 'text',
        content: 'Based on your budget, I recommend the Dell XPS 13, MacBook Air M2, and Lenovo ThinkPad X1...'
      }
    },
    {
      id: 2,
      type: 'ai_interaction',
      agent: 'Travel',
      icon: Plane,
      color: 'cyan.400',
      title: 'Travel Planner created your Tokyo itinerary',
      description: 'Generated a 7-day trip plan to Tokyo, Japan including accommodations, activities, and dining recommendations.',
      time: new Date(Date.now() - 14400000),
      engagement: { likes: 24, comments: 8, saved: true },
      preview: {
        type: 'text',
        content: 'Day 1: Arrival in Tokyo - Check into hotel in Shibuya, explore Harajuku district, visit Meiji Shrine...'
      }
    },
    {
      id: 3,
      type: 'ai_interaction',
      agent: 'News',
      icon: Newspaper,
      color: 'pink.400',
      title: 'News Assistant summarized tech headlines',
      description: 'Curated and summarized the latest technology news from diverse sources.',
      time: new Date(Date.now() - 21600000),
      engagement: { likes: 8, comments: 2, saved: false },
      preview: {
        type: 'text',
        content: 'Top Tech News: AI breakthroughs in healthcare, new smartphone launches, cybersecurity updates...'
      }
    },
    {
      id: 4,
      type: 'ai_interaction',
      agent: 'Content',
      icon: Sparkles,
      color: 'orange.400',
      title: 'Content Generator wrote a blog post',
      description: 'Created a professional blog post about productivity tips for remote workers.',
      time: new Date(Date.now() - 43200000),
      engagement: { likes: 32, comments: 12, saved: true },
      preview: {
        type: 'text',
        content: '10 Productivity Tips for Remote Workers: 1. Create a dedicated workspace...'
      }
    },
    {
      id: 5,
      type: 'ai_interaction',
      agent: 'Wellness',
      icon: Heart,
      color: 'red.400',
      title: 'Wellness Coach created your health plan',
      description: 'Personalized wellness plan with nutrition tips, exercise routines, and mindfulness practices.',
      time: new Date(Date.now() - 86400000),
      engagement: { likes: 18, comments: 5, saved: true },
      preview: {
        type: 'text',
        content: 'Your Personalized Wellness Plan: Morning routine - 20 min meditation, healthy breakfast...'
      }
    }
  ];

  const ActivityCard = ({ activity }) => {
    const IconComponent = activity.icon;
    
    return (
      <Box
        bg="whiteAlpha.100"
        backdropFilter="blur(10px)"
        borderRadius="xl"
        border="1px solid"
        borderColor="whiteAlpha.200"
        p={4}
        transition="all 0.3s"
        _hover={{
          bg: 'whiteAlpha.150',
          borderColor: 'whiteAlpha.300',
          transform: 'translateY(-2px)',
          boxShadow: '0 8px 20px rgba(0,0,0,0.3)'
        }}
      >
        <VStack align="stretch" spacing={3}>
          <HStack spacing={3}>
            <Box
              p={2}
              borderRadius="lg"
              bg={activity.color}
              color="white"
            >
              <Icon as={IconComponent} boxSize={5} />
            </Box>
            
            <VStack align="start" spacing={0} flex="1">
              <Text fontSize="sm" fontWeight="bold" color="white">
                {activity.title}
              </Text>
              <Text fontSize="xs" color="gray.400">
                {formatDistanceToNow(activity.time, { addSuffix: true })}
              </Text>
            </VStack>
            
            <Badge colorScheme="purple" fontSize="xs">
              {activity.agent}
            </Badge>
          </HStack>
          
          <Text fontSize="sm" color="gray.300" lineHeight="tall">
            {activity.description}
          </Text>
          
          {activity.preview && (
            <Box
              p={3}
              bg="blackAlpha.400"
              borderRadius="lg"
              borderLeft="3px solid"
              borderColor={activity.color}
            >
              <Text fontSize="xs" color="gray.300" fontStyle="italic">
                {activity.preview.content}
              </Text>
            </Box>
          )}
          
          <HStack spacing={4} pt={2}>
            <HStack
              spacing={1}
              cursor="pointer"
              color="gray.400"
              _hover={{ color: 'purple.400' }}
            >
              <Icon as={ThumbsUp} boxSize={4} />
              <Text fontSize="xs">{activity.engagement.likes}</Text>
            </HStack>
            
            <HStack
              spacing={1}
              cursor="pointer"
              color="gray.400"
              _hover={{ color: 'cyan.400' }}
            >
              <Icon as={MessageSquare} boxSize={4} />
              <Text fontSize="xs">{activity.engagement.comments}</Text>
            </HStack>
            
            <HStack
              spacing={1}
              cursor="pointer"
              color="gray.400"
              _hover={{ color: 'pink.400' }}
            >
              <Icon as={Share2} boxSize={4} />
              <Text fontSize="xs">Share</Text>
            </HStack>
            
            <HStack
              spacing={1}
              cursor="pointer"
              color={activity.engagement.saved ? 'yellow.400' : 'gray.400'}
              _hover={{ color: 'yellow.400' }}
              ml="auto"
            >
              <Icon as={Bookmark} boxSize={4} fill={activity.engagement.saved ? 'currentColor' : 'none'} />
            </HStack>
          </HStack>
        </VStack>
      </Box>
    );
  };

  return (
    <Box px={4} py={4}>
      <VStack align="stretch" spacing={6}>
        <VStack align="start" spacing={2}>
          <HStack spacing={3}>
            <Icon as={ActivityIcon} boxSize={8} color="purple.400" />
            <Text fontSize="3xl" fontWeight="bold" color="white">
              Activity
            </Text>
          </HStack>
          <Text fontSize="sm" color="gray.400">
            Your AI interactions and achievements
          </Text>
        </VStack>

        <Tabs
          colorScheme="purple"
          onChange={(index) => setSelectedTab(index)}
        >
          <TabList borderColor="whiteAlpha.200">
            <Tab
              color="gray.400"
              _selected={{ color: 'white', borderColor: 'purple.400' }}
            >
              All Activity
            </Tab>
            <Tab
              color="gray.400"
              _selected={{ color: 'white', borderColor: 'purple.400' }}
            >
              Saved
            </Tab>
            <Tab
              color="gray.400"
              _selected={{ color: 'white', borderColor: 'purple.400' }}
            >
              Popular
            </Tab>
          </TabList>

          <TabPanels>
            <TabPanel px={0}>
              <VStack spacing={4} align="stretch">
                {activities.map((activity) => (
                  <ActivityCard key={activity.id} activity={activity} />
                ))}
              </VStack>
            </TabPanel>
            
            <TabPanel px={0}>
              <VStack spacing={4} align="stretch">
                {activities.filter(a => a.engagement.saved).map((activity) => (
                  <ActivityCard key={activity.id} activity={activity} />
                ))}
              </VStack>
            </TabPanel>
            
            <TabPanel px={0}>
              <VStack spacing={4} align="stretch">
                {activities
                  .sort((a, b) => b.engagement.likes - a.engagement.likes)
                  .map((activity) => (
                    <ActivityCard key={activity.id} activity={activity} />
                  ))}
              </VStack>
            </TabPanel>
          </TabPanels>
        </Tabs>
      </VStack>
    </Box>
  );
};

export default Activity;
