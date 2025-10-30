import { useState, useEffect } from 'react';
import {
  Box,
  Heading,
  VStack,
  Text,
  HStack,
  Icon,
  Select,
  Spinner,
  useToast,
  Button
} from '@chakra-ui/react';
import { Package, RefreshCw } from 'lucide-react';
import { API_BASE_URL } from '@utils/constants';
import PostCard from '@components/common/PostCard';

export default function Products() {
  const [products, setProducts] = useState([]);
  const [categories, setCategories] = useState([]);
  const [selectedCategory, setSelectedCategory] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [status, setStatus] = useState('');
  const toast = useToast();

  useEffect(() => {
    fetch(`${API_BASE_URL}/api/products/categories`)
      .then(res => res.json())
      .then(data => setCategories(data.categories || []))
      .catch(() => {});
    
    loadProducts();
  }, []);

  const loadProducts = () => {
    setIsLoading(true);
    setProducts([]);
    setStatus('Discovering products...');

    const params = new URLSearchParams();
    if (selectedCategory) params.append('category', selectedCategory);
    params.append('min_score', '50');
    params.append('max_results', '20');

    const eventSource = new EventSource(`${API_BASE_URL}/api/products/discover?${params}`);

    eventSource.onmessage = (event) => {
      try {
        const parsed = JSON.parse(event.data);
        
        if (parsed.type === 'status') {
          setStatus(parsed.message);
        } else if (parsed.type === 'product') {
          setProducts(prev => [...prev, parsed.data]);
        } else if (parsed.type === 'complete') {
          setStatus('');
          setIsLoading(false);
          toast({
            title: parsed.message,
            status: 'success',
            duration: 2000,
          });
        } else if (parsed.type === 'error') {
          setStatus('');
          setIsLoading(false);
          toast({
            title: 'Error loading products',
            description: parsed.error,
            status: 'error',
            duration: 4000,
          });
        }
      } catch (e) {}
    };

    eventSource.onerror = () => {
      eventSource.close();
      setIsLoading(false);
      setStatus('');
    };

    return () => eventSource.close();
  };

  return (
    <Box px={4} py={6}>
      <VStack align="start" spacing={5}>
        <Box
          w="100%"
          p={6}
          borderRadius="2xl"
          bgGradient="linear(135deg, green.600, teal.600)"
          color="white"
          position="relative"
          overflow="hidden"
        >
          <Box position="absolute" right="-20px" top="-20px" opacity={0.2}>
            <Package size={120} />
          </Box>
          <VStack align="start" spacing={2} position="relative">
            <HStack>
              <Icon as={Package} boxSize={8} />
              <Heading size="lg">Quality Products</Heading>
            </HStack>
            <Text fontSize="sm" opacity={0.9}>
              Curated apps and products with AI-powered quality scoring
            </Text>
          </VStack>
        </Box>

        <HStack w="100%" spacing={3}>
          <Select
            value={selectedCategory}
            onChange={(e) => setSelectedCategory(e.target.value)}
            bg="whiteAlpha.100"
            backdropFilter="blur(10px)"
            border="1px solid"
            borderColor="whiteAlpha.300"
            color="white"
            borderRadius="xl"
            _focus={{
              bg: 'whiteAlpha.150',
              borderColor: 'green.400',
            }}
          >
            <option value="" style={{ background: '#1a202c' }}>All Categories</option>
            {categories.map(cat => (
              <option key={cat} value={cat} style={{ background: '#1a202c' }}>
                {cat}
              </option>
            ))}
          </Select>
          <Button
            onClick={loadProducts}
            isLoading={isLoading}
            colorScheme="green"
            leftIcon={<RefreshCw size={16} />}
            size="md"
          >
            Refresh
          </Button>
        </HStack>

        {status && (
          <HStack w="100%" justify="center" p={4}>
            <Spinner size="sm" color="green.400" />
            <Text fontSize="sm" color="gray.400">{status}</Text>
          </HStack>
        )}

        {products.length > 0 && (
          <VStack spacing={4} w="100%">
            {products.map((product, idx) => (
              <PostCard 
                key={idx} 
                post={{
                  ...product,
                  content: product.title,
                  tags: product.price ? [`💰 ${product.price}`] : undefined
                }} 
                showImage={false} 
              />
            ))}
          </VStack>
        )}

        {!isLoading && products.length === 0 && (
          <Box
            w="100%"
            p={8}
            textAlign="center"
            bg="whiteAlpha.50"
            borderRadius="xl"
            border="1px dashed"
            borderColor="whiteAlpha.300"
          >
            <Package size={48} style={{ margin: '0 auto 16px', opacity: 0.3 }} />
            <Text color="gray.500">No products found. Try a different category.</Text>
          </Box>
        )}
      </VStack>
    </Box>
  );
}
