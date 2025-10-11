import ws from 'ws';
import { chatStorage, gamificationStorage, notificationStorage } from './storage.js';

interface ChatClient {
  ws: ws;
  userId: string;
  roomId: string;
  username?: string;
}

export class ChatServer {
  private wss: ws.Server;
  private clients: Map<string, ChatClient> = new Map();
  private roomClients: Map<string, Set<string>> = new Map();

  constructor(port: number = 8080) {
    this.wss = new ws.Server({ port });
    this.setupWebSocketServer();
    console.log(`💬 Chat WebSocket server running on ws://0.0.0.0:${port}`);
  }

  private setupWebSocketServer() {
    this.wss.on('connection', (socket: ws) => {
      let clientId: string | null = null;

      socket.on('message', async (data: any) => {
        try {
          const message = JSON.parse(data.toString());

          switch (message.type) {
            case 'join':
              clientId = await this.handleJoin(ws, message);
              break;
            case 'message':
              await this.handleMessage(clientId, message);
              break;
            case 'leave':
              await this.handleLeave(clientId);
              break;
            case 'typing':
              this.handleTyping(clientId, message);
              break;
            case 'reaction':
              await this.handleReaction(clientId, message);
              break;
          }
        } catch (error) {
          console.error('WebSocket message error:', error);
          socket.send(JSON.stringify({ type: 'error', error: 'Invalid message format' }));
        }
      });

      socket.on('close', async () => {
        if (clientId) {
          await this.handleDisconnect(clientId);
        }
      });

      socket.on('error', (error) => {
        console.error('WebSocket error:', error);
      });
    });
  }

  private async handleJoin(socket: ws, message: any): Promise<string> {
    const { userId, roomId, username } = message;
    const clientId = `${userId}_${roomId}_${Date.now()}`;

    const client: ChatClient = {
      ws: socket,
      userId,
      roomId,
      username
    };

    this.clients.set(clientId, client);

    if (!this.roomClients.has(roomId)) {
      this.roomClients.set(roomId, new Set());
    }
    this.roomClients.get(roomId)?.add(clientId);

    await chatStorage.joinRoom({
      roomId,
      userId,
      role: 'participant',
      status: 'active'
    });

    const participants = await chatStorage.getRoomParticipants(roomId);
    const messages = await chatStorage.getMessages(roomId, 50);

    socket.send(JSON.stringify({
      type: 'joined',
      roomId,
      participants,
      messages: messages.reverse()
    }));

    this.broadcastToRoom(roomId, {
      type: 'user_joined',
      userId,
      username,
      timestamp: new Date()
    }, clientId);

    return clientId;
  }

  private async handleMessage(clientId: string | null, message: any) {
    if (!clientId) return;

    const client = this.clients.get(clientId);
    if (!client) return;

    const { content, replyToId } = message;

    const savedMessage = await chatStorage.createMessage({
      roomId: client.roomId,
      userId: client.userId,
      content,
      replyToId,
      messageType: 'text'
    });

    await gamificationStorage.addPoints({
      userId: client.userId,
      points: 5,
      action: 'chat_message',
      reason: 'Sent a chat message'
    });

    this.broadcastToRoom(client.roomId, {
      type: 'message',
      message: savedMessage
    });
  }

  private async handleLeave(clientId: string | null) {
    if (!clientId) return;

    const client = this.clients.get(clientId);
    if (!client) return;

    await chatStorage.leaveRoom(client.roomId, client.userId);

    this.broadcastToRoom(client.roomId, {
      type: 'user_left',
      userId: client.userId,
      username: client.username,
      timestamp: new Date()
    }, clientId);

    this.roomClients.get(client.roomId)?.delete(clientId);
    this.clients.delete(clientId);
  }

  private handleTyping(clientId: string | null, message: any) {
    if (!clientId) return;

    const client = this.clients.get(clientId);
    if (!client) return;

    this.broadcastToRoom(client.roomId, {
      type: 'typing',
      userId: client.userId,
      username: client.username,
      isTyping: message.isTyping
    }, clientId);
  }

  private async handleReaction(clientId: string | null, message: any) {
    if (!clientId) return;

    const client = this.clients.get(clientId);
    if (!client) return;

    this.broadcastToRoom(client.roomId, {
      type: 'reaction',
      messageId: message.messageId,
      userId: client.userId,
      reaction: message.reaction
    });
  }

  private async handleDisconnect(clientId: string) {
    const client = this.clients.get(clientId);
    if (!client) return;

    this.roomClients.get(client.roomId)?.delete(clientId);
    this.clients.delete(clientId);

    this.broadcastToRoom(client.roomId, {
      type: 'user_offline',
      userId: client.userId,
      username: client.username
    });
  }

  private broadcastToRoom(roomId: string, message: any, excludeClientId?: string) {
    const roomClientIds = this.roomClients.get(roomId);
    if (!roomClientIds) return;

    const messageStr = JSON.stringify(message);

    roomClientIds.forEach(clientId => {
      if (clientId === excludeClientId) return;

      const client = this.clients.get(clientId);
      if (client && client.ws.readyState === WebSocket.OPEN) {
        client.ws.send(messageStr);
      }
    });
  }

  public getServer() {
    return this.wss;
  }
}
