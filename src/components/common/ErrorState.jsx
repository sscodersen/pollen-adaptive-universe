import { Box, VStack, Icon, Heading, Text, Button } from '@chakra-ui/react';
import { AlertCircle, RefreshCw } from 'lucide-react';

const ErrorState = ({ 
  message = 'Something went wrong', 
  description = 'Please try again or contact support if the problem persists.',
  onRetry 
}) => {
  return (
    <Box
      display="flex"
      alignItems="center"
      justifyContent="center"
      minH="300px"
      w="full"
      p={6}
    >
      <VStack spacing={4} textAlign="center" maxW="md">
        <Icon as={AlertCircle} boxSize={16} color="red.500" />
        <Heading size="md" color="gray.800">
          {message}
        </Heading>
        <Text color="gray.600" fontSize="sm">
          {description}
        </Text>
        {onRetry && (
          <Button
            leftIcon={<RefreshCw size={18} />}
            colorScheme="purple"
            variant="outline"
            onClick={onRetry}
            mt={2}
          >
            Try Again
          </Button>
        )}
      </VStack>
    </Box>
  );
};

export default ErrorState;