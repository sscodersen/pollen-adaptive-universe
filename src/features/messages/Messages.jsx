import { useState } from 'react';
import { Box, VStack, HStack, Text, Avatar, Icon, Input, InputGroup, InputLeftElement, Badge } from '@chakra-ui/react';
import { Search, MoreHorizontal, Send } from 'lucide-react';

const Messages = () => {
  const [selectedChat, setSelectedChat] = useState(null);
  const [messageText, setMessageText] = useState('');

  const conversations = [
    {
      id: 1,
      user: {
        name: 'Julie Mendez',
        avatar: null,
        location: 'Memphis, TN, US',
        online: true
      },
      lastMessage: 'That sounds great! When should we meet?',
      time: '2m',
      unread: 2
    },
    {
      id: 2,
      user: {
        name: 'Johnathan Hartley',
        avatar: null,
        location: 'Newark, NJ, US',
        online: true
      },
      lastMessage: 'Thanks for sharing that resource',
      time: '1h',
      unread: 0
    },
    {
      id: 3,
      user: {
        name: 'Maximus McKay',
        avatar: null,
        location: 'Fort Worth, TX, US',
        online: false
      },
      lastMessage: 'Sounds good, talk soon!',
      time: '3h',
      unread: 0
    },
    {
      id: 4,
      user: {
        name: 'Jasmin Alvarez',
        avatar: null,
        location: 'Springfield, MA, US',
        online: true
      },
      lastMessage: 'Did you see the latest updates?',
      time: '5h',
      unread: 1
    }
  ];

  const chatMessages = selectedChat ? [
    {
      id: 1,
      from: 'them',
      text: 'Hey! How are you doing?',
      time: '10:30 AM'
    },
    {
      id: 2,
      from: 'me',
      text: 'Hey! I\'m doing great, thanks! How about you?',
      time: '10:32 AM'
    },
    {
      id: 3,
      from: 'them',
      text: 'That sounds great! When should we meet?',
      time: '10:35 AM'
    }
  ] : [];

  return (
    <Box h="calc(100vh - 32px)">
      <HStack spacing={0} h="100%" align="stretch">
        <Box
          w="320px"
          borderRight="1px solid"
          borderColor="whiteAlpha.200"
          bg="black"
          borderRadius="xl"
          overflow="hidden"
        >
          <VStack spacing={0} align="stretch" h="100%">
            <Box p={4} borderBottom="1px solid" borderColor="whiteAlpha.200">
              <HStack justify="space-between" mb={4}>
                <Text fontSize="lg" fontWeight="bold" color="white">
                  Messages
                </Text>
                <Badge colorScheme="purple" borderRadius="full">
                  {conversations.filter(c => c.unread > 0).reduce((sum, c) => sum + c.unread, 0)}
                </Badge>
              </HStack>
              <InputGroup>
                <InputLeftElement>
                  <Icon as={Search} color="gray.400" boxSize={4} />
                </InputLeftElement>
                <Input
                  placeholder="Search messages..."
                  bg="whiteAlpha.100"
                  border="none"
                  color="white"
                  _placeholder={{ color: 'gray.500' }}
                  size="sm"
                  borderRadius="lg"
                />
              </InputGroup>
            </Box>

            <VStack spacing={0} align="stretch" flex={1} overflowY="auto">
              {conversations.map((conv) => (
                <HStack
                  key={conv.id}
                  spacing={3}
                  p={3}
                  cursor="pointer"
                  bg={selectedChat === conv.id ? 'whiteAlpha.100' : 'transparent'}
                  borderLeft={selectedChat === conv.id ? '3px solid' : '3px solid transparent'}
                  borderColor={selectedChat === conv.id ? 'purple.500' : 'transparent'}
                  _hover={{ bg: 'whiteAlpha.50' }}
                  transition="all 0.2s"
                  onClick={() => setSelectedChat(conv.id)}
                >
                  <Box position="relative">
                    <Avatar
                      size="sm"
                      name={conv.user.name}
                      bg="linear-gradient(135deg, #667eea 0%, #764ba2 100%)"
                    />
                    {conv.user.online && (
                      <Box
                        position="absolute"
                        bottom="0"
                        right="0"
                        w="10px"
                        h="10px"
                        bg="green.400"
                        borderRadius="full"
                        border="2px solid black"
                      />
                    )}
                  </Box>
                  <VStack align="start" spacing={0} flex={1}>
                    <HStack justify="space-between" w="100%">
                      <Text fontSize="sm" fontWeight="bold" color="white">
                        {conv.user.name}
                      </Text>
                      <Text fontSize="xs" color="gray.500">
                        {conv.time}
                      </Text>
                    </HStack>
                    <Text fontSize="xs" color="gray.400" noOfLines={1}>
                      {conv.lastMessage}
                    </Text>
                  </VStack>
                  {conv.unread > 0 && (
                    <Badge colorScheme="purple" borderRadius="full" fontSize="xs">
                      {conv.unread}
                    </Badge>
                  )}
                </HStack>
              ))}
            </VStack>
          </VStack>
        </Box>

        <Box flex={1} display="flex" flexDirection="column" bg="black" borderRadius="xl" ml={4}>
          {selectedChat ? (
            <>
              <HStack
                p={4}
                borderBottom="1px solid"
                borderColor="whiteAlpha.200"
                justify="space-between"
              >
                <HStack spacing={3}>
                  <Avatar
                    size="sm"
                    name={conversations.find(c => c.id === selectedChat)?.user.name}
                    bg="linear-gradient(135deg, #667eea 0%, #764ba2 100%)"
                  />
                  <VStack align="start" spacing={0}>
                    <Text fontSize="sm" fontWeight="bold" color="white">
                      {conversations.find(c => c.id === selectedChat)?.user.name}
                    </Text>
                    <Text fontSize="xs" color="gray.400">
                      {conversations.find(c => c.id === selectedChat)?.user.location}
                    </Text>
                  </VStack>
                </HStack>
                <Icon as={MoreHorizontal} boxSize={5} color="gray.400" cursor="pointer" />
              </HStack>

              <VStack
                flex={1}
                p={4}
                spacing={3}
                overflowY="auto"
                align="stretch"
              >
                {chatMessages.map((msg) => (
                  <HStack
                    key={msg.id}
                    justify={msg.from === 'me' ? 'flex-end' : 'flex-start'}
                  >
                    <Box
                      maxW="70%"
                      p={3}
                      borderRadius="lg"
                      bg={msg.from === 'me' ? 'purple.600' : 'whiteAlpha.100'}
                    >
                      <Text fontSize="sm" color="white">
                        {msg.text}
                      </Text>
                      <Text fontSize="xs" color="gray.400" mt={1}>
                        {msg.time}
                      </Text>
                    </Box>
                  </HStack>
                ))}
              </VStack>

              <HStack
                p={4}
                borderTop="1px solid"
                borderColor="whiteAlpha.200"
                spacing={3}
              >
                <Input
                  placeholder="Write a message..."
                  value={messageText}
                  onChange={(e) => setMessageText(e.target.value)}
                  bg="whiteAlpha.100"
                  border="none"
                  color="white"
                  _placeholder={{ color: 'gray.500' }}
                  borderRadius="lg"
                />
                <Icon
                  as={Send}
                  boxSize={5}
                  color={messageText ? 'purple.400' : 'gray.600'}
                  cursor={messageText ? 'pointer' : 'not-allowed'}
                />
              </HStack>
            </>
          ) : (
            <VStack flex={1} justify="center" align="center" spacing={4}>
              <Box
                p={4}
                borderRadius="lg"
                bgGradient="linear(to-br, purple.500, pink.500)"
              >
                <Icon as={Send} boxSize={8} color="white" />
              </Box>
              <Text fontSize="lg" fontWeight="bold" color="white">
                Select a conversation
              </Text>
              <Text fontSize="sm" color="gray.400" textAlign="center" maxW="300px">
                Choose a conversation from the list to start messaging
              </Text>
            </VStack>
          )}
        </Box>
      </HStack>
    </Box>
  );
};

export default Messages;
