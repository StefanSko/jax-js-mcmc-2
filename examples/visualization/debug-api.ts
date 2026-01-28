/**
 * Debug API Plugin for Vite
 *
 * Provides REST endpoints for agentic debugging of MCMC visualization.
 * Uses command queue pattern for browser-server communication without HMR.
 *
 * Endpoints:
 * - GET  /__debug/state   - Get current sampler state
 * - POST /__debug/step    - Trigger one sampler step
 * - POST /__debug/reset   - Reset sampler to initial state
 * - POST /__debug/config  - Update sampler configuration
 * - GET  /__debug/logs    - Get recent console messages
 *
 * Internal (for frontend polling):
 * - GET  /__debug/poll    - Frontend polls for pending commands
 * - POST /__debug/result  - Frontend posts command results
 *
 * Usage:
 *   curl http://localhost:5173/__debug/state
 *   curl -X POST http://localhost:5173/__debug/step
 *   curl -X POST -d '{"algorithm":"rwm"}' http://localhost:5173/__debug/config
 */

import type { Plugin } from 'vite';

export interface DebugCommand {
  id: string;
  type: 'getState' | 'step' | 'reset' | 'setConfig';
  payload?: unknown;
}

export interface DebugLogEntry {
  level: string;
  message: string;
  timestamp: string;
}

export function debugApiPlugin(): Plugin {
  // Command queue for browser communication
  let pendingCommand: DebugCommand | null = null;
  const commandWaiters = new Map<string, {
    resolve: (value: unknown) => void;
    reject: (error: Error) => void;
  }>();

  // Log buffer for /__debug/logs
  const logBuffer: DebugLogEntry[] = [];
  const MAX_LOG_BUFFER = 200;

  return {
    name: 'debug-api',
    configureServer(server) {
      // Intercept console bridge messages to populate log buffer
      server.middlewares.use((req, res, next) => {
        if (req.url === '/__console' && req.method === 'POST') {
          let body = '';
          req.on('data', (chunk) => { body += chunk; });
          req.on('end', () => {
            try {
              const data = JSON.parse(body) as DebugLogEntry;
              logBuffer.push({ level: data.level, message: data.message, timestamp: data.timestamp });
              if (logBuffer.length > MAX_LOG_BUFFER) {
                logBuffer.shift();
              }
            } catch { /* ignore parse errors */ }
          });
        }
        return next();
      });

      // Debug API middleware
      server.middlewares.use((req, res, next) => {
        if (!req.url?.startsWith('/__debug')) {
          return next();
        }

        const url = new URL(req.url, 'http://localhost');
        const endpoint = url.pathname.replace('/__debug/', '').replace('/__debug', '');

        // Set JSON response headers
        res.setHeader('Content-Type', 'application/json');
        res.setHeader('Access-Control-Allow-Origin', '*');

        // Handle CORS preflight
        if (req.method === 'OPTIONS') {
          res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
          res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
          res.statusCode = 204;
          res.end();
          return;
        }

        // Helper to queue command and wait for result
        const queueCommand = (type: DebugCommand['type'], payload?: unknown, timeout = 5000): Promise<unknown> => {
          return new Promise((resolve, reject) => {
            const id = `${Date.now()}-${Math.random().toString(36).slice(2)}`;
            pendingCommand = { id, type, payload };
            commandWaiters.set(id, { resolve, reject });

            setTimeout(() => {
              if (commandWaiters.has(id)) {
                commandWaiters.delete(id);
                if (pendingCommand?.id === id) {
                  pendingCommand = null;
                }
                reject(new Error('Timeout - browser may not be connected or polling'));
              }
            }, timeout);
          });
        };

        // Helper to send JSON response
        const sendJson = (status: number, data: unknown) => {
          res.statusCode = status;
          res.end(JSON.stringify(data));
        };

        // Helper to read request body
        const readBody = (): Promise<string> => {
          return new Promise((resolve) => {
            let body = '';
            req.on('data', (chunk) => { body += chunk; });
            req.on('end', () => resolve(body));
          });
        };

        // Route handlers
        switch (endpoint) {
          case 'poll':
            // Frontend polls for pending commands
            if (req.method !== 'GET') {
              return sendJson(405, { error: 'Method Not Allowed' });
            }
            if (pendingCommand) {
              sendJson(200, pendingCommand);
            } else {
              sendJson(200, null);
            }
            break;

          case 'result':
            // Frontend posts command results
            if (req.method !== 'POST') {
              return sendJson(405, { error: 'Method Not Allowed' });
            }
            readBody().then((body) => {
              try {
                const data = JSON.parse(body) as { id: string; result: unknown };
                const waiter = commandWaiters.get(data.id);
                if (waiter) {
                  commandWaiters.delete(data.id);
                  if (pendingCommand?.id === data.id) {
                    pendingCommand = null;
                  }
                  waiter.resolve(data.result);
                }
                sendJson(200, { ok: true });
              } catch {
                sendJson(400, { error: 'Invalid JSON' });
              }
            });
            break;

          case 'state':
            if (req.method !== 'GET') {
              return sendJson(405, { error: 'Method Not Allowed' });
            }
            queueCommand('getState')
              .then((result) => sendJson(200, result))
              .catch((err) => sendJson(503, { error: err.message }));
            break;

          case 'step':
            if (req.method !== 'POST') {
              return sendJson(405, { error: 'Method Not Allowed' });
            }
            queueCommand('step')
              .then((result) => sendJson(200, result))
              .catch((err) => sendJson(503, { error: err.message }));
            break;

          case 'reset':
            if (req.method !== 'POST') {
              return sendJson(405, { error: 'Method Not Allowed' });
            }
            queueCommand('reset')
              .then((result) => sendJson(200, result))
              .catch((err) => sendJson(503, { error: err.message }));
            break;

          case 'config':
            if (req.method !== 'POST') {
              return sendJson(405, { error: 'Method Not Allowed' });
            }
            readBody().then((body) => {
              try {
                const config = JSON.parse(body);
                queueCommand('setConfig', config)
                  .then((result) => sendJson(200, result))
                  .catch((err) => sendJson(503, { error: err.message }));
              } catch {
                sendJson(400, { error: 'Invalid JSON' });
              }
            });
            break;

          case 'logs':
            if (req.method !== 'GET') {
              return sendJson(405, { error: 'Method Not Allowed' });
            }
            const limit = parseInt(url.searchParams.get('limit') || '50', 10);
            sendJson(200, logBuffer.slice(-limit));
            break;

          default:
            sendJson(404, {
              error: 'Not Found',
              endpoints: ['state', 'step', 'reset', 'config', 'logs'],
            });
        }
      });
    },
  };
}
