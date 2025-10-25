import { Box, Text, VStack, Icon } from '@chakra-ui/react';
import { useNavigate } from 'react-router-dom';
import * as Icons from 'lucide-react';

const FeatureCard = ({ feature }) => {
  const navigate = useNavigate();
  const IconComponent = Icons[feature.icon];

  return (
    <Box
      bg="whiteAlpha.100"
      backdropFilter="blur(10px)"
      borderRadius="2xl"
      p={5}
      border="1px solid"
      borderColor="whiteAlpha.200"
      cursor="pointer"
      transition="all 0.3s"
      _hover={{
        transform: 'translateY(-4px)',
        boxShadow: 'xl',
        bg: 'whiteAlpha.200',
        borderColor: 'whiteAlpha.300',
      }}
      onClick={() => navigate(feature.path)}
      position="relative"
      overflow="hidden"
    >
      <Box
        position="absolute"
        top={0}
        right={0}
        w="100px"
        h="100px"
        bgGradient={feature.gradient}
        opacity={0.15}
        borderRadius="full"
        transform="translate(30%, -30%)"
      />

      <VStack align="start" spacing={3} position="relative">
        <Box
          p={3}
          borderRadius="xl"
          bgGradient={feature.gradient}
          color="white"
        >
          {IconComponent && <Icon as={IconComponent} boxSize={6} />}
        </Box>

        <VStack align="start" spacing={1}>
          <Text fontSize="lg" fontWeight="bold" color="white">
            {feature.title}
          </Text>
          <Text fontSize="sm" color="gray.400">
            {feature.subtitle}
          </Text>
        </VStack>

        <Text fontSize="xs" color="gray.500" noOfLines={2}>
          {feature.description}
        </Text>
      </VStack>
    </Box>
  );
};

export default FeatureCard;
