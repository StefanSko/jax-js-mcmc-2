/**
 * Console Bridge - Development Mode
 *
 * Intercepts console.log/warn/error/info/debug and sends them to a backend
 * endpoint for server-side logging. Useful for debugging browser code from
 * the terminal.
 *
 * Self-protective: won't try to send its own network errors to avoid loops.
 */

const BRIDGE_ENDPOINT = '/__console';
const BRIDGE_TIMEOUT_MS = 2000;

// Track if we're currently sending to avoid recursive loops
let isSending = false;

// Store original console methods
const originalConsole = {
  log: console.log.bind(console),
  warn: console.warn.bind(console),
  error: console.error.bind(console),
  info: console.info.bind(console),
  debug: console.debug.bind(console),
};

type LogLevel = keyof typeof originalConsole;

/**
 * Serialize a value for transmission. Handles common types gracefully.
 */
function serialize(value: unknown): string {
  if (value === undefined) return 'undefined';
  if (value === null) return 'null';
  if (typeof value === 'string') return value;
  if (typeof value === 'number' || typeof value === 'boolean') return String(value);
  if (value instanceof Error) {
    return `${value.name}: ${value.message}${value.stack ? '\n' + value.stack : ''}`;
  }
  if (typeof value === 'function') return `[Function: ${value.name || 'anonymous'}]`;

  try {
    return JSON.stringify(value, (_key, v) => {
      if (typeof v === 'bigint') return `${v}n`;
      if (v instanceof Error) return `${v.name}: ${v.message}`;
      if (typeof v === 'function') return `[Function: ${v.name || 'anonymous'}]`;
      if (v instanceof HTMLElement) return `[${v.tagName}${v.id ? '#' + v.id : ''}]`;
      return v;
    }, 2);
  } catch {
    return String(value);
  }
}

/**
 * Format arguments like console does
 */
function formatArgs(args: unknown[]): string {
  return args.map(serialize).join(' ');
}

/**
 * Send a log message to the backend. Fire-and-forget with timeout.
 */
async function sendToBackend(level: LogLevel, args: unknown[]): Promise<void> {
  // Guard against recursive sending (e.g., if fetch itself logs something)
  if (isSending) {
    // Queue for later instead of dropping
    setTimeout(() => sendToBackend(level, args), 10);
    return;
  }

  isSending = true;
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), BRIDGE_TIMEOUT_MS);

  try {
    await fetch(BRIDGE_ENDPOINT, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        level,
        message: formatArgs(args),
        timestamp: new Date().toISOString(),
        url: window.location.href,
      }),
      signal: controller.signal,
    });
  } catch {
    // Silently ignore errors - don't log them or we create a loop
  } finally {
    clearTimeout(timeout);
    isSending = false;
  }
}

/**
 * Create a wrapped console method that logs locally AND sends to backend
 */
function createWrapper(level: LogLevel) {
  return (...args: unknown[]) => {
    // Always call original first (so user sees immediate feedback)
    originalConsole[level](...args);

    // Send to backend asynchronously (fire-and-forget)
    sendToBackend(level, args);
  };
}

/**
 * Install the console bridge. Call once at app startup.
 */
export function installConsoleBridge(): void {
  // Only install when VITE_CONSOLE_BRIDGE=1 is set
  // Run with: CONSOLE_BRIDGE=1 npm run viz
  if (!import.meta.env.VITE_CONSOLE_BRIDGE) {
    return;
  }

  // Don't install in production
  if (import.meta.env.PROD) {
    return;
  }

  console.log = createWrapper('log');
  console.warn = createWrapper('warn');
  console.error = createWrapper('error');
  console.info = createWrapper('info');
  console.debug = createWrapper('debug');

  // Capture uncaught exceptions
  window.addEventListener('error', (event) => {
    sendToBackend('error', [`[UNCAUGHT] ${event.message} at ${event.filename}:${event.lineno}:${event.colno}`]);
  });

  // Capture unhandled promise rejections
  window.addEventListener('unhandledrejection', (event) => {
    const reason = event.reason instanceof Error
      ? `${event.reason.name}: ${event.reason.message}`
      : String(event.reason);
    sendToBackend('error', [`[UNHANDLED REJECTION] ${reason}`]);
  });

  // Log that we're active (using original to avoid sending this to backend)
  originalConsole.info('[console-bridge] Active - logs will appear in terminal');
}

/**
 * Restore original console methods
 */
export function uninstallConsoleBridge(): void {
  console.log = originalConsole.log;
  console.warn = originalConsole.warn;
  console.error = originalConsole.error;
  console.info = originalConsole.info;
  console.debug = originalConsole.debug;
}

// Auto-install on import in development
installConsoleBridge();
