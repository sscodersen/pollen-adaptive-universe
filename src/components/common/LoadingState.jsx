import { Box, VStack, Spinner, Text } from '@chakra-ui/react';

const LoadingState = ({ message = 'Loading...', size = 'xl' }) => {
  return (
    <Box
      display="flex"
      alignItems="center"
      justifyContent="center"
      minH="200px"
      w="full"
    >
      <VStack spacing={4}>
        <Spinner
          thickness="4px"
          speed="0.65s"
          emptyColor="purple.100"
          color="purple.500"
          size={size}
        />
        <Text color="gray.600" fontSize="sm" fontWeight="500">
          {message}
        </Text>
      </VStack>
    </Box>
  );
};

export default LoadingState;