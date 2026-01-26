import { defineConfig, type Plugin, type ViteDevServer } from 'vite';

/**
 * Console Bridge Plugin
 *
 * Receives console messages from the browser and logs them to the terminal.
 * Makes debugging browser code much easier during development.
 */
function consoleBridgePlugin(): Plugin {
  return {
    name: 'console-bridge',
    configureServer(server) {
      // Add middleware directly (runs before Vite's internal middleware)
      server.middlewares.use((req, res, next) => {
          // Only handle our console endpoint
          if (!req.url?.startsWith('/__console')) {
            return next();
          }

          if (req.method !== 'POST') {
            res.statusCode = 405;
            res.end('Method Not Allowed');
            return;
          }

          let body = '';
          req.on('data', (chunk) => {
            body += chunk;
          });

          req.on('end', () => {
            try {
              const data = JSON.parse(body) as {
                level: string;
                message: string;
                timestamp: string;
                url: string;
              };

              // Color codes for different log levels
              const colors: Record<string, string> = {
                log: '\x1b[36m',    // cyan
                info: '\x1b[34m',   // blue
                warn: '\x1b[33m',   // yellow
                error: '\x1b[31m',  // red
                debug: '\x1b[90m',  // gray
              };
              const reset = '\x1b[0m';
              const dim = '\x1b[2m';

              const color = colors[data.level] || colors.log;
              const levelPadded = data.level.toUpperCase().padEnd(5);
              const time = new Date(data.timestamp).toLocaleTimeString();

              // Format: [HH:MM:SS] LEVEL  message
              console.log(
                `${dim}[${time}]${reset} ${color}${levelPadded}${reset} ${data.message}`
              );

              res.statusCode = 200;
              res.end('OK');
            } catch {
              res.statusCode = 400;
              res.end('Bad Request');
            }
          });
        });
    },
  };
}

/**
 * API Control Plugin
 *
 * Provides HTTP API endpoints to control the HMC visualization from the terminal.
 * Communicates with the browser via Vite's HMR WebSocket.
 */
function apiControlPlugin(): Plugin {
  let viteServer: ViteDevServer | null = null;
  const pendingRequests = new Map<string, {
    resolve: (value: unknown) => void;
    timeout: ReturnType<typeof setTimeout>;
  }>();

  return {
    name: 'api-control',
    configureServer(server) {
      viteServer = server;

      // Handle HMR messages from browser (responses to our commands)
      server.ws.on('hmcviz:response', (data: { requestId: string; result: unknown }) => {
        const pending = pendingRequests.get(data.requestId);
        if (pending) {
          clearTimeout(pending.timeout);
          pendingRequests.delete(data.requestId);
          pending.resolve(data.result);
        }
      });

      // Helper to send command to browser and wait for response
      const sendCommand = (command: string, timeout = 5000): Promise<unknown> => {
        return new Promise((resolve, reject) => {
          const requestId = `${Date.now()}-${Math.random().toString(36).slice(2)}`;

          const timeoutHandle = setTimeout(() => {
            pendingRequests.delete(requestId);
            reject(new Error('Request timeout - browser may not be connected'));
          }, timeout);

          pendingRequests.set(requestId, { resolve, timeout: timeoutHandle });

          server.ws.send({
            type: 'custom',
            event: 'hmcviz:command',
            data: { requestId, command },
          });
        });
      };

      // Add API middleware
      server.middlewares.use((req, res, next) => {
        if (!req.url?.startsWith('/__api/')) {
          return next();
        }

        const endpoint = req.url.replace('/__api/', '');

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

        const handleCommand = async (command: string) => {
          try {
            const result = await sendCommand(command);
            res.statusCode = 200;
            res.end(JSON.stringify(result));
          } catch (error) {
            res.statusCode = 503;
            res.end(JSON.stringify({
              error: error instanceof Error ? error.message : 'Unknown error',
            }));
          }
        };

        switch (endpoint) {
          case 'status':
            if (req.method === 'GET') {
              handleCommand('getStatus');
            } else {
              res.statusCode = 405;
              res.end(JSON.stringify({ error: 'Method Not Allowed' }));
            }
            break;

          case 'play':
            if (req.method === 'POST') {
              handleCommand('play');
            } else {
              res.statusCode = 405;
              res.end(JSON.stringify({ error: 'Method Not Allowed' }));
            }
            break;

          case 'pause':
            if (req.method === 'POST') {
              handleCommand('pause');
            } else {
              res.statusCode = 405;
              res.end(JSON.stringify({ error: 'Method Not Allowed' }));
            }
            break;

          case 'step':
            if (req.method === 'POST') {
              handleCommand('step');
            } else {
              res.statusCode = 405;
              res.end(JSON.stringify({ error: 'Method Not Allowed' }));
            }
            break;

          case 'reset':
            if (req.method === 'POST') {
              handleCommand('reset');
            } else {
              res.statusCode = 405;
              res.end(JSON.stringify({ error: 'Method Not Allowed' }));
            }
            break;

          default:
            res.statusCode = 404;
            res.end(JSON.stringify({ error: 'Not Found', endpoints: ['status', 'play', 'pause', 'step', 'reset'] }));
        }
      });
    },
  };
}

export default defineConfig({
  plugins: [consoleBridgePlugin(), apiControlPlugin()],
  // Pass CONSOLE_BRIDGE env var to client as VITE_CONSOLE_BRIDGE
  define: {
    'import.meta.env.VITE_CONSOLE_BRIDGE': JSON.stringify(process.env.CONSOLE_BRIDGE === '1'),
  },
  // Optimize for development
  server: {
    open: true,  // Auto-open browser
  },
});
