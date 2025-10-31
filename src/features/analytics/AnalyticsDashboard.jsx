import { useState, useEffect } from 'react';
import {
  Box,
  VStack,
  HStack,
  Heading,
  Text,
  Icon,
  Grid,
  Progress,
  Badge,
  Divider
} from '@chakra-ui/react';
import {
  TrendingUp,
  Eye,
  Heart,
  Bookmark,
  Star,
  Clock,
  Activity,
  BarChart3
} from 'lucide-react';
import { usePersonalization } from '@hooks/usePersonalization';

const AnalyticsDashboard = () => {
  const [stats, setStats] = useState(null);
  const [activityHistory, setActivityHistory] = useState([]);
  const { getActivityStats } = usePersonalization();

  useEffect(() => {
    loadAnalytics();
  }, []);

  const loadAnalytics = () => {
    const activityStats = getActivityStats();
    const activities = JSON.parse(localStorage.getItem('userActivities') || '[]');
    const likedPosts = JSON.parse(localStorage.getItem('likedPosts') || '[]');
    const bookmarkedPosts = JSON.parse(localStorage.getItem('bookmarkedPosts') || '[]');

    const now = new Date();
    const last7Days = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
    const last30Days = new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000);

    const last7DaysActivities = activities.filter(a => 
      new Date(a.timestamp) >= last7Days
    );
    const last30DaysActivities = activities.filter(a =>
      new Date(a.timestamp) >= last30Days
    );

    const timeSpent = last7DaysActivities.reduce((total, a) => {
      if (a.type === 'session') {
        return total + (a.data?.duration || 0);
      }
      return total + 30;
    }, 0);

    const categoryCounts = activities.reduce((acc, a) => {
      const category = a.data?.category || 'general';
      acc[category] = (acc[category] || 0) + 1;
      return acc;
    }, {});

    const topCategories = Object.entries(categoryCounts)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5);

    setStats({
      totalViews: last7DaysActivities.length,
      totalLikes: likedPosts.length,
      totalBookmarks: bookmarkedPosts.length,
      todayActivity: activityStats.todayActivities,
      weekActivity: last7DaysActivities.length,
      monthActivity: last30DaysActivities.length,
      timeSpent: Math.round(timeSpent / 60),
      topCategories,
      interests: activityStats.interests,
      favoriteTopics: activityStats.favoriteTopics
    });

    const recentActivities = activities
      .slice(0, 10)
      .map(a => ({
        ...a,
        timeAgo: getTimeAgo(new Date(a.timestamp))
      }));
    setActivityHistory(recentActivities);
  };

  const getTimeAgo = (date) => {
    const seconds = Math.floor((new Date() - date) / 1000);
    
    if (seconds < 60) return 'just now';
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
    if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
    return `${Math.floor(seconds / 86400)}d ago`;
  };

  if (!stats) return null;

  const StatCard = ({ icon, label, value, color, trend, trendValue }) => (
    <Box
      p={5}
      bg="black"
      borderRadius="xl"
      border="1px solid"
      borderColor="whiteAlpha.200"
      _hover={{ borderColor: 'whiteAlpha.400' }}
      transition="all 0.2s"
    >
      <VStack align="stretch" spacing={3}>
        <HStack justify="space-between">
          <Icon as={icon} boxSize={6} color={color} />
          {trend && (
            <Badge
              colorScheme={trendValue > 0 ? 'green' : 'gray'}
              fontSize="xs"
              px={2}
              py={1}
            >
              <HStack spacing={1}>
                <Icon as={TrendingUp} boxSize={3} />
                <Text>{trendValue > 0 ? '+' : ''}{trendValue}%</Text>
              </HStack>
            </Badge>
          )}
        </HStack>
        <Box>
          <Text fontSize="3xl" fontWeight="bold" color="white">
            {value}
          </Text>
          <Text fontSize="sm" color="gray.400">
            {label}
          </Text>
        </Box>
      </VStack>
    </Box>
  );

  return (
    <Box px={4} py={6}>
      <VStack spacing={6} align="stretch">
        <Box>
          <HStack spacing={3} mb={2}>
            <Icon as={BarChart3} boxSize={8} color="purple.500" />
            <Heading size="xl" color="white">
              Analytics Dashboard
            </Heading>
          </HStack>
          <Text color="gray.400">
            Your platform activity and engagement insights
          </Text>
        </Box>

        <Grid templateColumns="repeat(auto-fit, minmax(250px, 1fr))" gap={4}>
          <StatCard
            icon={Eye}
            label="Views This Week"
            value={stats.weekActivity}
            color="blue.400"
            trend={true}
            trendValue={12}
          />
          <StatCard
            icon={Heart}
            label="Total Likes"
            value={stats.totalLikes}
            color="red.400"
          />
          <StatCard
            icon={Bookmark}
            label="Bookmarks Saved"
            value={stats.totalBookmarks}
            color="purple.400"
          />
          <StatCard
            icon={Clock}
            label="Time Spent (min)"
            value={stats.timeSpent}
            color="green.400"
          />
        </Grid>

        <Grid templateColumns="repeat(auto-fit, minmax(300px, 1fr))" gap={6}>
          <Box
            p={5}
            bg="black"
            borderRadius="xl"
            border="1px solid"
            borderColor="whiteAlpha.200"
          >
            <HStack spacing={3} mb={4}>
              <Icon as={Activity} boxSize={6} color="purple.400" />
              <Heading size="md" color="white">
                Activity Trend
              </Heading>
            </HStack>
            <VStack spacing={3} align="stretch">
              <HStack justify="space-between">
                <Text fontSize="sm" color="gray.400">Today</Text>
                <HStack spacing={2}>
                  <Progress
                    value={(stats.todayActivity / stats.weekActivity) * 100}
                    w="100px"
                    colorScheme="purple"
                    borderRadius="full"
                  />
                  <Text fontSize="sm" fontWeight="bold" color="white">
                    {stats.todayActivity}
                  </Text>
                </HStack>
              </HStack>
              <HStack justify="space-between">
                <Text fontSize="sm" color="gray.400">This Week</Text>
                <HStack spacing={2}>
                  <Progress
                    value={(stats.weekActivity / stats.monthActivity) * 100}
                    w="100px"
                    colorScheme="blue"
                    borderRadius="full"
                  />
                  <Text fontSize="sm" fontWeight="bold" color="white">
                    {stats.weekActivity}
                  </Text>
                </HStack>
              </HStack>
              <HStack justify="space-between">
                <Text fontSize="sm" color="gray.400">This Month</Text>
                <HStack spacing={2}>
                  <Progress
                    value={100}
                    w="100px"
                    colorScheme="green"
                    borderRadius="full"
                  />
                  <Text fontSize="sm" fontWeight="bold" color="white">
                    {stats.monthActivity}
                  </Text>
                </HStack>
              </HStack>
            </VStack>
          </Box>

          <Box
            p={5}
            bg="black"
            borderRadius="xl"
            border="1px solid"
            borderColor="whiteAlpha.200"
          >
            <HStack spacing={3} mb={4}>
              <Icon as={Star} boxSize={6} color="yellow.400" />
              <Heading size="md" color="white">
                Top Categories
              </Heading>
            </HStack>
            <VStack spacing={3} align="stretch">
              {stats.topCategories.slice(0, 5).map(([category, count], index) => (
                <HStack key={category} justify="space-between">
                  <HStack>
                    <Badge colorScheme="purple" fontSize="xs">
                      #{index + 1}
                    </Badge>
                    <Text fontSize="sm" color="white" textTransform="capitalize">
                      {category}
                    </Text>
                  </HStack>
                  <Text fontSize="sm" fontWeight="bold" color="gray.400">
                    {count}
                  </Text>
                </HStack>
              ))}
            </VStack>
          </Box>
        </Grid>

        <Box
          p={5}
          bg="black"
          borderRadius="xl"
          border="1px solid"
          borderColor="whiteAlpha.200"
        >
          <HStack spacing={3} mb={4}>
            <Icon as={Clock} boxSize={6} color="blue.400" />
            <Heading size="md" color="white">
              Recent Activity
            </Heading>
          </HStack>
          <VStack spacing={3} align="stretch" divider={<Divider borderColor="whiteAlpha.200" />}>
            {activityHistory.length === 0 ? (
              <Text fontSize="sm" color="gray.400" textAlign="center" py={4}>
                No recent activity
              </Text>
            ) : (
              activityHistory.map((activity, index) => (
                <HStack key={index} justify="space-between">
                  <HStack>
                    <Icon
                      as={
                        activity.type === 'view' ? Eye :
                        activity.type === 'like' ? Heart :
                        activity.type === 'bookmark' ? Bookmark :
                        Activity
                      }
                      boxSize={4}
                      color="purple.400"
                    />
                    <Text fontSize="sm" color="white">
                      {activity.type === 'view' && 'Viewed content'}
                      {activity.type === 'like' && 'Liked a post'}
                      {activity.type === 'bookmark' && 'Saved a bookmark'}
                      {activity.type === 'share' && 'Shared a post'}
                      {!['view', 'like', 'bookmark', 'share'].includes(activity.type) && 'Activity'}
                    </Text>
                  </HStack>
                  <Text fontSize="xs" color="gray.500">
                    {activity.timeAgo}
                  </Text>
                </HStack>
              ))
            )}
          </VStack>
        </Box>

        <Grid templateColumns="repeat(2, 1fr)" gap={4}>
          <Box
            p={5}
            bg="purple.900"
            bgGradient="linear-gradient(135deg, rgba(103, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%)"
            borderRadius="xl"
            border="1px solid"
            borderColor="purple.700"
            textAlign="center"
          >
            <Text fontSize="4xl" fontWeight="bold" color="white" mb={1}>
              {stats.interests}
            </Text>
            <Text fontSize="sm" color="gray.300">
              Active Interests
            </Text>
          </Box>
          <Box
            p={5}
            bg="blue.900"
            bgGradient="linear-gradient(135deg, rgba(66, 153, 225, 0.2) 0%, rgba(49, 130, 206, 0.2) 100%)"
            borderRadius="xl"
            border="1px solid"
            borderColor="blue.700"
            textAlign="center"
          >
            <Text fontSize="4xl" fontWeight="bold" color="white" mb={1}>
              {stats.favoriteTopics}
            </Text>
            <Text fontSize="sm" color="gray.300">
              Favorite Topics
            </Text>
          </Box>
        </Grid>
      </VStack>
    </Box>
  );
};

export default AnalyticsDashboard;
