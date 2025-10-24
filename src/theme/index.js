import { extendTheme } from '@chakra-ui/react';

const theme = extendTheme({
  fonts: {
    heading: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
    body: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
  },
  colors: {
    brand: {
      50: '#f0e4ff',
      100: '#d4b8ff',
      200: '#b88cff',
      300: '#9c60ff',
      400: '#8034ff',
      500: '#6610f2',
      600: '#5209c7',
      700: '#3d069c',
      800: '#280371',
      900: '#130146',
    },
    gradient: {
      sunrise: 'linear-gradient(135deg, #F9BF3B 0%, #997c7e 100%)',
      ocean: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
      sunset: 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)',
      forest: 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)',
      lavender: 'linear-gradient(135deg, #a8edea 0%, #fed6e3 100%)',
      peach: 'linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%)',
    },
  },
  styles: {
    global: {
      body: {
        bg: 'gray.50',
      },
    },
  },
  components: {
    Button: {
      baseStyle: {
        fontWeight: 'semibold',
        borderRadius: 'lg',
      },
      variants: {
        primary: {
          bg: 'brand.500',
          color: 'white',
          _hover: {
            bg: 'brand.600',
            transform: 'translateY(-2px)',
            boxShadow: 'lg',
          },
          transition: 'all 0.2s',
        },
        glass: {
          bg: 'whiteAlpha.300',
          backdropFilter: 'blur(10px)',
          border: '1px solid',
          borderColor: 'whiteAlpha.400',
          color: 'gray.800',
          _hover: {
            bg: 'whiteAlpha.400',
          },
        },
      },
    },
    Card: {
      baseStyle: {
        container: {
          borderRadius: 'xl',
          boxShadow: 'lg',
          overflow: 'hidden',
        },
      },
      variants: {
        glass: {
          container: {
            bg: 'whiteAlpha.700',
            backdropFilter: 'blur(10px)',
            border: '1px solid',
            borderColor: 'whiteAlpha.400',
          },
        },
      },
    },
  },
  config: {
    initialColorMode: 'light',
    useSystemColorMode: false,
  },
});

export default theme;
