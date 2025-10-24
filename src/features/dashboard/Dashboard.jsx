import { Box, SimpleGrid, Text, VStack } from '@chakra-ui/react';
import { FEATURES } from '@utils/constants';
import FeatureCard from '@components/common/FeatureCard';
import SearchBar from '@components/common/SearchBar';

const Dashboard = () => {
  return (
    <Box px={4} py={4}>
      <SearchBar placeholder="What can I help you with today?" />

      <VStack align="start" spacing={6} mt={4}>
        <Box width="100%">
          <Text fontSize="lg" fontWeight="bold" color="gray.800" mb={4}>
            Your AI Assistant
          </Text>
          <SimpleGrid columns={2} spacing={4}>
            {FEATURES.map((feature) => (
              <FeatureCard key={feature.id} feature={feature} />
            ))}
          </SimpleGrid>
        </Box>

        <Box width="100%" mt={4}>
          <Text fontSize="md" fontWeight="semibold" color="gray.700" mb={3}>
            Quick Actions
          </Text>
          <VStack spacing={3}>
            <Box
              w="100%"
              p={4}
              bg="whiteAlpha.700"
              backdropFilter="blur(10px)"
              borderRadius="xl"
              border="1px solid"
              borderColor="whiteAlpha.400"
            >
              <Text fontSize="sm" fontWeight="medium" color="gray.800">
                Continue your last conversation
              </Text>
              <Text fontSize="xs" color="gray.600" mt={1}>
                Shopping recommendations for winter gear
              </Text>
            </Box>

            <Box
              w="100%"
              p={4}
              bg="whiteAlpha.700"
              backdropFilter="blur(10px)"
              borderRadius="xl"
              border="1px solid"
              borderColor="whiteAlpha.400"
            >
              <Text fontSize="sm" fontWeight="medium" color="gray.800">
                Suggested for you
              </Text>
              <Text fontSize="xs" color="gray.600" mt={1}>
                Plan your upcoming weekend trip
              </Text>
            </Box>
          </VStack>
        </Box>
      </VStack>
    </Box>
  );
};

export default Dashboard;
