import { useState, useRef, useEffect } from 'react';
import {
  Box,
  VStack,
  HStack,
  Heading,
  Text,
  Button,
  Textarea,
  Select,
  Icon,
  Grid,
  Badge,
  useToast,
  Tabs,
  TabList,
  TabPanels,
  Tab,
  TabPanel,
  Image,
  Code,
  Input,
  Divider,
  Progress,
  IconButton,
  Tooltip
} from '@chakra-ui/react';
import {
  Mic,
  MicOff,
  Image as ImageIcon,
  Video,
  Code2,
  Workflow,
  Sparkles,
  Send,
  Download,
  Copy,
  Play,
  Pause,
  Volume2,
  VolumeX,
  Trash2,
  RefreshCw
} from 'lucide-react';

const PollenPlayground = () => {
  const [mode, setMode] = useState('chat');
  const [input, setInput] = useState('');
  const [response, setResponse] = useState('');
  const [loading, setLoading] = useState(false);
  const [recording, setRecording] = useState(false);
  const [generatedImage, setGeneratedImage] = useState(null);
  const [generatedVideo, setGeneratedVideo] = useState(null);
  const [generatedCode, setGeneratedCode] = useState('');
  const [audioUrl, setAudioUrl] = useState(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [mediaRecorder, setMediaRecorder] = useState(null);
  const audioRef = useRef(null);
  const toast = useToast();

  const modes = [
    { id: 'chat', label: 'Chat', icon: Sparkles, desc: 'General AI chat' },
    { id: 'voice', label: 'Voice', icon: Mic, desc: 'Voice input & output' },
    { id: 'image', label: 'Image', icon: ImageIcon, desc: 'Generate images' },
    { id: 'video', label: 'Video', icon: Video, desc: 'Generate videos' },
    { id: 'code', label: 'Code', icon: Code2, desc: 'Code assistance' },
    { id: 'tasks', label: 'Tasks', icon: Workflow, desc: 'Task automation' },
    { id: 'react', label: 'ReAct', icon: RefreshCw, desc: 'Reasoning & acting' }
  ];

  const currentMode = modes.find(m => m.id === mode);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      const chunks = [];

      recorder.ondataavailable = (e) => chunks.push(e.data);
      recorder.onstop = async () => {
        const blob = new Blob(chunks, { type: 'audio/webm' });
        const formData = new FormData();
        formData.append('audio', blob);

        setLoading(true);
        try {
          const response = await fetch('/api/playground/voice-to-text', {
            method: 'POST',
            body: formData
          });
          const data = await response.json();
          setInput(data.text);
          toast({
            title: 'Voice transcribed',
            status: 'success',
            duration: 2000
          });
        } catch (error) {
          toast({
            title: 'Transcription failed',
            description: error.message,
            status: 'error',
            duration: 3000
          });
        }
        setLoading(false);
        stream.getTracks().forEach(track => track.stop());
      };

      recorder.start();
      setMediaRecorder(recorder);
      setRecording(true);
    } catch (error) {
      toast({
        title: 'Microphone access denied',
        description: error.message,
        status: 'error',
        duration: 3000
      });
    }
  };

  const stopRecording = () => {
    if (mediaRecorder) {
      mediaRecorder.stop();
      setRecording(false);
    }
  };

  const handleSubmit = async () => {
    if (!input.trim()) return;

    setLoading(true);
    setResponse('');

    try {
      let endpoint = '';
      let body = {};

      switch (mode) {
        case 'chat':
          endpoint = '/api/playground/chat';
          body = { message: input };
          break;
        case 'voice':
          endpoint = '/api/playground/text-to-speech';
          body = { text: input };
          break;
        case 'image':
          endpoint = '/api/playground/generate-image';
          body = { prompt: input };
          break;
        case 'video':
          endpoint = '/api/playground/generate-video';
          body = { prompt: input };
          break;
        case 'code':
          endpoint = '/api/playground/code-assist';
          body = { request: input };
          break;
        case 'tasks':
          endpoint = '/api/playground/automate-task';
          body = { task: input };
          break;
        case 'react':
          endpoint = '/api/playground/react-mode';
          body = { query: input };
          break;
      }

      if (mode === 'voice') {
        const response = await fetch(endpoint, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body)
        });
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        setAudioUrl(url);
        toast({
          title: 'Audio generated',
          status: 'success',
          duration: 2000
        });
      } else if (mode === 'image') {
        const response = await fetch(endpoint, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body)
        });
        const data = await response.json();
        setGeneratedImage(data.image_url);
        setResponse(data.description || 'Image generated successfully');
      } else if (mode === 'video') {
        const response = await fetch(endpoint, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body)
        });
        const data = await response.json();
        setGeneratedVideo(data.video_url);
        setResponse(data.description || 'Video generated successfully');
      } else if (mode === 'code') {
        const eventSource = new EventSource(`${endpoint}?request=${encodeURIComponent(input)}`);
        let code = '';
        
        eventSource.onmessage = (event) => {
          const data = JSON.parse(event.data);
          if (data.type === 'code') {
            code += data.content;
            setGeneratedCode(code);
          } else if (data.type === 'explanation') {
            setResponse(data.content);
          } else if (data.type === 'complete') {
            eventSource.close();
          }
        };
        
        eventSource.onerror = () => {
          eventSource.close();
          setLoading(false);
        };
      } else {
        const eventSource = new EventSource(`${endpoint}?${mode === 'tasks' ? 'task' : mode === 'react' ? 'query' : 'message'}=${encodeURIComponent(input)}`);
        let fullResponse = '';
        
        eventSource.onmessage = (event) => {
          const data = JSON.parse(event.data);
          if (data.text) {
            fullResponse += data.text;
            setResponse(fullResponse);
          } else if (data.type === 'complete') {
            eventSource.close();
          }
        };
        
        eventSource.onerror = () => {
          eventSource.close();
          setLoading(false);
        };
      }

      setLoading(false);
    } catch (error) {
      toast({
        title: 'Request failed',
        description: error.message,
        status: 'error',
        duration: 3000
      });
      setLoading(false);
    }
  };

  const toggleAudio = () => {
    if (audioRef.current) {
      if (isPlaying) {
        audioRef.current.pause();
      } else {
        audioRef.current.play();
      }
      setIsPlaying(!isPlaying);
    }
  };

  const copyToClipboard = (text) => {
    navigator.clipboard.writeText(text);
    toast({
      title: 'Copied to clipboard',
      status: 'success',
      duration: 2000
    });
  };

  const downloadImage = () => {
    const link = document.createElement('a');
    link.href = generatedImage;
    link.download = 'generated-image.png';
    link.click();
  };

  const clearResponse = () => {
    setResponse('');
    setGeneratedImage(null);
    setGeneratedVideo(null);
    setGeneratedCode('');
    setAudioUrl(null);
  };

  return (
    <Box px={4} py={6}>
      <VStack spacing={6} align="stretch">
        <Box>
          <HStack spacing={3} mb={2}>
            <Icon as={Sparkles} boxSize={8} color="purple.500" />
            <Heading size="xl" color="white">
              Pollen AI Playground
            </Heading>
          </HStack>
          <Text color="gray.400">
            Explore AI capabilities: voice, image, video, code, automation & more
          </Text>
        </Box>

        <Grid templateColumns="repeat(auto-fit, minmax(150px, 1fr))" gap={3}>
          {modes.map((m) => (
            <Box
              key={m.id}
              p={4}
              bg={mode === m.id ? 'purple.600' : 'black'}
              borderRadius="xl"
              border="2px solid"
              borderColor={mode === m.id ? 'purple.500' : 'whiteAlpha.200'}
              cursor="pointer"
              onClick={() => {
                setMode(m.id);
                clearResponse();
              }}
              _hover={{
                borderColor: mode === m.id ? 'purple.400' : 'whiteAlpha.400',
                transform: 'translateY(-2px)'
              }}
              transition="all 0.2s"
            >
              <VStack spacing={2}>
                <Icon as={m.icon} boxSize={6} color={mode === m.id ? 'white' : 'purple.400'} />
                <Text fontSize="sm" fontWeight="bold" color="white">
                  {m.label}
                </Text>
                <Text fontSize="xs" color="gray.400" textAlign="center">
                  {m.desc}
                </Text>
              </VStack>
            </Box>
          ))}
        </Grid>

        <Box
          p={6}
          bg="black"
          borderRadius="2xl"
          border="1px solid"
          borderColor="whiteAlpha.200"
        >
          <VStack spacing={4} align="stretch">
            <HStack justify="space-between">
              <HStack spacing={3}>
                <Icon as={currentMode.icon} boxSize={6} color="purple.400" />
                <Heading size="md" color="white">
                  {currentMode.label} Mode
                </Heading>
              </HStack>
              <Badge colorScheme="purple" fontSize="sm" px={3} py={1}>
                {currentMode.desc}
              </Badge>
            </HStack>

            <Divider borderColor="whiteAlpha.200" />

            <VStack spacing={4} align="stretch">
              <HStack spacing={2}>
                <Textarea
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  placeholder={`Enter your ${mode} request...`}
                  bg="whiteAlpha.100"
                  border="1px solid"
                  borderColor="whiteAlpha.300"
                  color="white"
                  minH="120px"
                  _focus={{ borderColor: 'purple.500' }}
                />
                {mode === 'voice' && (
                  <VStack spacing={2}>
                    <Tooltip label={recording ? 'Stop recording' : 'Start recording'}>
                      <IconButton
                        aria-label="Record"
                        icon={<Icon as={recording ? MicOff : Mic} />}
                        onClick={recording ? stopRecording : startRecording}
                        colorScheme={recording ? 'red' : 'purple'}
                        size="lg"
                        isLoading={loading}
                      />
                    </Tooltip>
                  </VStack>
                )}
              </HStack>

              <HStack justify="space-between">
                <Button
                  leftIcon={<Icon as={Send} />}
                  onClick={handleSubmit}
                  colorScheme="purple"
                  isLoading={loading}
                  loadingText="Processing..."
                  size="lg"
                >
                  Generate
                </Button>
                {response || generatedImage || generatedVideo || generatedCode || audioUrl ? (
                  <Button
                    leftIcon={<Icon as={Trash2} />}
                    onClick={clearResponse}
                    variant="ghost"
                    colorScheme="red"
                  >
                    Clear
                  </Button>
                ) : null}
              </HStack>
            </VStack>

            {loading && (
              <Box>
                <Progress colorScheme="purple" size="xs" isIndeterminate borderRadius="full" />
                <Text fontSize="sm" color="gray.400" mt={2} textAlign="center">
                  Generating your {mode} response...
                </Text>
              </Box>
            )}

            {response && (
              <Box
                p={4}
                bg="whiteAlpha.50"
                borderRadius="xl"
                border="1px solid"
                borderColor="whiteAlpha.200"
              >
                <HStack justify="space-between" mb={2}>
                  <Text fontSize="sm" fontWeight="bold" color="white">
                    Response
                  </Text>
                  <IconButton
                    aria-label="Copy"
                    icon={<Icon as={Copy} />}
                    size="sm"
                    variant="ghost"
                    onClick={() => copyToClipboard(response)}
                  />
                </HStack>
                <Text color="white" whiteSpace="pre-wrap">
                  {response}
                </Text>
              </Box>
            )}

            {generatedCode && (
              <Box
                bg="gray.950"
                borderRadius="xl"
                border="1px solid"
                borderColor="whiteAlpha.200"
                overflow="hidden"
              >
                <HStack justify="space-between" p={3} bg="whiteAlpha.100">
                  <Text fontSize="sm" fontWeight="bold" color="white">
                    Generated Code
                  </Text>
                  <IconButton
                    aria-label="Copy code"
                    icon={<Icon as={Copy} />}
                    size="sm"
                    variant="ghost"
                    onClick={() => copyToClipboard(generatedCode)}
                  />
                </HStack>
                <Box p={4} overflowX="auto">
                  <Code
                    w="full"
                    bg="transparent"
                    color="green.300"
                    fontSize="sm"
                    whiteSpace="pre"
                    fontFamily="monospace"
                  >
                    {generatedCode}
                  </Code>
                </Box>
              </Box>
            )}

            {generatedImage && (
              <Box
                borderRadius="xl"
                border="1px solid"
                borderColor="whiteAlpha.200"
                overflow="hidden"
              >
                <Image src={generatedImage} alt="Generated" w="full" />
                <HStack justify="center" p={3} bg="whiteAlpha.100">
                  <Button
                    leftIcon={<Icon as={Download} />}
                    onClick={downloadImage}
                    size="sm"
                    colorScheme="purple"
                  >
                    Download
                  </Button>
                </HStack>
              </Box>
            )}

            {generatedVideo && (
              <Box
                borderRadius="xl"
                border="1px solid"
                borderColor="whiteAlpha.200"
                overflow="hidden"
              >
                <video src={generatedVideo} controls style={{ width: '100%' }} />
                <HStack justify="center" p={3} bg="whiteAlpha.100">
                  <Button
                    leftIcon={<Icon as={Download} />}
                    onClick={() => {
                      const link = document.createElement('a');
                      link.href = generatedVideo;
                      link.download = 'generated-video.mp4';
                      link.click();
                    }}
                    size="sm"
                    colorScheme="purple"
                  >
                    Download
                  </Button>
                </HStack>
              </Box>
            )}

            {audioUrl && (
              <Box
                p={4}
                bg="whiteAlpha.50"
                borderRadius="xl"
                border="1px solid"
                borderColor="whiteAlpha.200"
              >
                <HStack spacing={4} justify="center">
                  <IconButton
                    aria-label={isPlaying ? 'Pause' : 'Play'}
                    icon={<Icon as={isPlaying ? Pause : Play} />}
                    onClick={toggleAudio}
                    colorScheme="purple"
                    size="lg"
                    borderRadius="full"
                  />
                  <Icon as={Volume2} boxSize={6} color="purple.400" />
                  <Text fontSize="sm" color="gray.400">
                    {isPlaying ? 'Playing...' : 'Audio ready'}
                  </Text>
                </HStack>
                <audio ref={audioRef} src={audioUrl} onEnded={() => setIsPlaying(false)} />
              </Box>
            )}
          </VStack>
        </Box>

        <Box
          p={4}
          bg="purple.900"
          bgGradient="linear-gradient(135deg, rgba(103, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%)"
          borderRadius="xl"
          border="1px solid"
          borderColor="purple.700"
        >
          <VStack spacing={2}>
            <HStack>
              <Icon as={Sparkles} boxSize={5} color="purple.300" />
              <Text fontSize="sm" fontWeight="bold" color="white">
                Tip
              </Text>
            </HStack>
            <Text fontSize="sm" color="gray.300" textAlign="center">
              {mode === 'chat' && 'Ask me anything! I can help with information, ideas, and conversations.'}
              {mode === 'voice' && 'Click the microphone to record or type text for speech synthesis.'}
              {mode === 'image' && 'Describe the image you want to create in detail for best results.'}
              {mode === 'video' && 'Describe your video concept, and I\'ll generate a short video clip.'}
              {mode === 'code' && 'Request code snippets, debugging help, or explanations.'}
              {mode === 'tasks' && 'Describe a task you want automated, and I\'ll help break it down.'}
              {mode === 'react' && 'I\'ll reason through your query and take actions to solve it.'}
            </Text>
          </VStack>
        </Box>
      </VStack>
    </Box>
  );
};

export default PollenPlayground;
