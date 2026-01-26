import { defineConfig, type Plugin } from 'vite';

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

export default defineConfig({
  plugins: [consoleBridgePlugin()],
  // Pass CONSOLE_BRIDGE env var to client as VITE_CONSOLE_BRIDGE
  define: {
    'import.meta.env.VITE_CONSOLE_BRIDGE': JSON.stringify(process.env.CONSOLE_BRIDGE === '1'),
  },
  // Optimize for development
  server: {
    open: false,
  },
});
