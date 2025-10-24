import { InputGroup, Input, InputLeftElement, Box } from '@chakra-ui/react';
import { Search } from 'lucide-react';

const SearchBar = ({ placeholder = 'Search anything...', value, onChange, onSearch }) => {
  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && onSearch) {
      onSearch();
    }
  };

  return (
    <Box px={4} mb={4}>
      <InputGroup>
        <InputLeftElement pointerEvents="none" color="gray.500">
          <Search size={20} />
        </InputLeftElement>
        <Input
          placeholder={placeholder}
          value={value}
          onChange={(e) => onChange && onChange(e.target.value)}
          onKeyPress={handleKeyPress}
          bg="whiteAlpha.800"
          backdropFilter="blur(10px)"
          border="1px solid"
          borderColor="whiteAlpha.500"
          borderRadius="full"
          _placeholder={{ color: 'gray.500' }}
          _focus={{
            bg: 'white',
            borderColor: 'purple.400',
            boxShadow: '0 0 0 1px var(--chakra-colors-purple-400)',
          }}
          fontSize="sm"
        />
      </InputGroup>
    </Box>
  );
};

export default SearchBar;