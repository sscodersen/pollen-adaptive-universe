import { Component } from 'react';
import { Box, Heading, Text, Button, VStack, Icon } from '@chakra-ui/react';
import { AlertTriangle } from 'lucide-react';

class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error('Error caught by boundary:', error, errorInfo);
  }

  handleReset = () => {
    this.setState({ hasError: false, error: null });
    window.location.reload();
  };

  render() {
    if (this.state.hasError) {
      return (
        <Box
          minH="100vh"
          display="flex"
          alignItems="center"
          justifyContent="center"
          bgGradient="linear-gradient(135deg, #a8edea 0%, #fed6e3 50%, #ffecd2 100%)"
          p={6}
        >
          <VStack
            spacing={6}
            bg="white"
            p={8}
            borderRadius="2xl"
            boxShadow="xl"
            maxW="md"
            textAlign="center"
          >
            <Icon as={AlertTriangle} boxSize={16} color="orange.500" />
            <Heading size="lg" color="gray.800">
              Oops! Something went wrong
            </Heading>
            <Text color="gray.600">
              We encountered an unexpected error. Don't worry, your data is safe.
            </Text>
            {process.env.NODE_ENV === 'development' && this.state.error && (
              <Box
                bg="red.50"
                p={4}
                borderRadius="md"
                w="full"
                textAlign="left"
                fontSize="sm"
                fontFamily="mono"
                color="red.800"
                overflowX="auto"
              >
                {this.state.error.toString()}
              </Box>
            )}
            <Button
              colorScheme="purple"
              size="lg"
              onClick={this.handleReset}
              w="full"
            >
              Reload Application
            </Button>
          </VStack>
        </Box>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;