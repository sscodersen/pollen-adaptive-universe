import { Box, Text, VStack, Icon, HStack } from '@chakra-ui/react';
import { useNavigate } from 'react-router-dom';
import * as Icons from 'lucide-react';
import { ArrowRight } from 'lucide-react';

const FeatureCard = ({ feature }) => {
  const navigate = useNavigate();
  const IconComponent = Icons[feature.icon];

  return (
    <Box
      bg="black"
      borderRadius="xl"
      p={6}
      border="1px solid"
      borderColor="whiteAlpha.200"
      cursor="pointer"
      transition="all 0.3s"
      _hover={{
        transform: 'translateY(-2px)',
        borderColor: 'purple.500',
        boxShadow: '0 8px 30px rgba(102, 126, 234, 0.3)',
      }}
      onClick={() => navigate(feature.path)}
      position="relative"
      overflow="hidden"
    >
      <VStack align="start" spacing={4} position="relative">
        <Box
          p={3}
          borderRadius="lg"
          bgGradient={feature.gradient}
          color="white"
        >
          {IconComponent && <Icon as={IconComponent} boxSize={6} />}
        </Box>

        <VStack align="start" spacing={1} flex="1" w="100%">
          <Text fontSize="lg" fontWeight="bold" color="white">
            {feature.title}
          </Text>
          <Text fontSize="sm" color="gray.500" fontWeight="medium">
            {feature.subtitle}
          </Text>
        </VStack>

        <HStack 
          w="100%" 
          justify="space-between" 
          pt={2}
          borderTop="1px solid"
          borderColor="whiteAlpha.100"
        >
          <Text fontSize="xs" color="gray.600">
            Start
          </Text>
          <Icon as={ArrowRight} boxSize={4} color="gray.600" />
        </HStack>
      </VStack>
    </Box>
  );
};

export default FeatureCard;
