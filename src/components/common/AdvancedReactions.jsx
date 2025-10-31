import { useState } from 'react';
import {
  Box,
  HStack,
  Text,
  Icon,
  Tooltip,
  Popover,
  PopoverTrigger,
  PopoverContent,
  PopoverBody,
  VStack,
  useToast
} from '@chakra-ui/react';
import {
  Heart,
  Lightbulb,
  Sparkles,
  ThumbsUp,
  Laugh,
  Frown,
  Star
} from 'lucide-react';

const REACTIONS = [
  { id: 'like', icon: Heart, label: 'Like', color: 'red.400' },
  { id: 'love', icon: Heart, label: 'Love', color: 'pink.400', fill: true },
  { id: 'insightful', icon: Lightbulb, label: 'Insightful', color: 'yellow.400' },
  { id: 'inspiring', icon: Sparkles, label: 'Inspiring', color: 'purple.400' },
  { id: 'helpful', icon: ThumbsUp, label: 'Helpful', color: 'blue.400' },
  { id: 'funny', icon: Laugh, label: 'Funny', color: 'orange.400' },
  { id: 'concerning', icon: Frown, label: 'Concerning', color: 'gray.400' }
];

const AdvancedReactions = ({ post }) => {
  const [reactions, setReactions] = useState(() => {
    const stored = localStorage.getItem(`reactions_${post?.id || post?.content_id}`);
    return stored ? JSON.parse(stored) : {};
  });
  const [myReaction, setMyReaction] = useState(null);
  const toast = useToast();

  const handleReaction = (reactionId) => {
    const newReactions = { ...reactions };
    
    if (myReaction) {
      newReactions[myReaction] = (newReactions[myReaction] || 0) - 1;
      if (newReactions[myReaction] <= 0) {
        delete newReactions[myReaction];
      }
    }

    if (myReaction === reactionId) {
      setMyReaction(null);
    } else {
      newReactions[reactionId] = (newReactions[reactionId] || 0) + 1;
      setMyReaction(reactionId);
      
      const reaction = REACTIONS.find(r => r.id === reactionId);
      toast({
        title: `Reacted with ${reaction.label}`,
        status: 'success',
        duration: 1000,
        position: 'bottom'
      });
    }

    setReactions(newReactions);
    localStorage.setItem(
      `reactions_${post?.id || post?.content_id}`,
      JSON.stringify(newReactions)
    );
  };

  const totalReactions = Object.values(reactions).reduce((sum, count) => sum + count, 0);

  return (
    <HStack spacing={3}>
      <Popover placement="top" trigger="hover">
        <PopoverTrigger>
          <HStack
            spacing={2}
            cursor="pointer"
            _hover={{ transform: 'scale(1.05)' }}
            transition="all 0.2s"
          >
            {myReaction ? (
              <>
                <Icon
                  as={REACTIONS.find(r => r.id === myReaction)?.icon}
                  boxSize={5}
                  color={REACTIONS.find(r => r.id === myReaction)?.color}
                  fill={REACTIONS.find(r => r.id === myReaction)?.fill ? 'currentColor' : 'none'}
                />
                <Text fontSize="sm" color="gray.400" fontWeight="medium">
                  {totalReactions}
                </Text>
              </>
            ) : (
              <>
                <Icon as={Heart} boxSize={5} color="gray.400" />
                <Text fontSize="sm" color="gray.400" fontWeight="medium">
                  {totalReactions || 'React'}
                </Text>
              </>
            )}
          </HStack>
        </PopoverTrigger>
        <PopoverContent
          bg="gray.800"
          border="1px solid"
          borderColor="whiteAlpha.300"
          borderRadius="xl"
          w="auto"
          _focus={{ boxShadow: 'none' }}
        >
          <PopoverBody p={2}>
            <HStack spacing={1}>
              {REACTIONS.map((reaction) => (
                <Tooltip key={reaction.id} label={reaction.label} placement="top">
                  <Box
                    p={2}
                    borderRadius="lg"
                    cursor="pointer"
                    bg={myReaction === reaction.id ? 'whiteAlpha.200' : 'transparent'}
                    _hover={{ bg: 'whiteAlpha.200', transform: 'scale(1.2)' }}
                    transition="all 0.2s"
                    onClick={() => handleReaction(reaction.id)}
                  >
                    <Icon
                      as={reaction.icon}
                      boxSize={6}
                      color={reaction.color}
                      fill={reaction.fill ? 'currentColor' : 'none'}
                    />
                  </Box>
                </Tooltip>
              ))}
            </HStack>
          </PopoverBody>
        </PopoverContent>
      </Popover>

      {Object.keys(reactions).length > 0 && (
        <HStack spacing={1}>
          {Object.entries(reactions)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 3)
            .map(([reactionId, count]) => {
              const reaction = REACTIONS.find(r => r.id === reactionId);
              return (
                <Tooltip key={reactionId} label={`${count} ${reaction.label}`}>
                  <HStack
                    spacing={0.5}
                    px={2}
                    py={1}
                    bg="whiteAlpha.100"
                    borderRadius="full"
                    fontSize="xs"
                  >
                    <Icon as={reaction.icon} boxSize={3} color={reaction.color} />
                    <Text color="gray.400">{count}</Text>
                  </HStack>
                </Tooltip>
              );
            })}
        </HStack>
      )}
    </HStack>
  );
};

export default AdvancedReactions;
