import { Box, HStack, keyframes } from '@chakra-ui/react';

const bounce = keyframes`
  0%, 80%, 100% { 
    transform: scale(0);
    opacity: 0.5;
  } 
  40% { 
    transform: scale(1);
    opacity: 1;
  }
`;

const TypingIndicator = ({ color = 'purple.400' }) => {
  return (
    <HStack spacing={2}>
      <Box
        w="8px"
        h="8px"
        borderRadius="full"
        bg={color}
        animation={`${bounce} 1.4s infinite ease-in-out both`}
        sx={{ animationDelay: '-0.32s' }}
      />
      <Box
        w="8px"
        h="8px"
        borderRadius="full"
        bg={color}
        animation={`${bounce} 1.4s infinite ease-in-out both`}
        sx={{ animationDelay: '-0.16s' }}
      />
      <Box
        w="8px"
        h="8px"
        borderRadius="full"
        bg={color}
        animation={`${bounce} 1.4s infinite ease-in-out both`}
      />
    </HStack>
  );
};

export default TypingIndicator;
