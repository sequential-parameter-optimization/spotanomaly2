"""Lightweight HTTP server for live report updates using Server-Sent Events."""

import asyncio
import json
import logging
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from spotanomaly2.domain.constants import FILE_CHANGE_DEBOUNCE_SECONDS, LIVE_REPORT_SERVER_PORT


class ReportFileWatcher(FileSystemEventHandler):
    """File system event handler for watching report data files."""

    def __init__(self, callback, watch_files: set[str]):
        """Initialize file watcher.

        Args:
            callback: Function to call when files change
            watch_files: Set of filenames to watch (e.g., {'metadata.json', 'figures.json'})
        """
        self.callback = callback
        self.watch_files = watch_files
        self.last_modification_time = {}
        self.last_event_time = 0.0

    def on_modified(self, event):
        """Handle file modification events."""
        if event.is_directory:
            return

        filename = Path(event.src_path).name
        if filename not in self.watch_files:
            return

        # Debounce rapid file changes (only trigger if > debounce interval since last global change)
        current_time = time.time()
        if current_time - self.last_event_time <= FILE_CHANGE_DEBOUNCE_SECONDS:
            return

        last_time = self.last_modification_time.get(filename, 0)

        if current_time - last_time > FILE_CHANGE_DEBOUNCE_SECONDS:
            self.last_modification_time[filename] = current_time
            self.last_event_time = current_time
            self.callback(filename)


class LiveReportServer:
    """HTTP server with SSE support for live report updates."""

    def __init__(self, results_dir: Path, port: int = LIVE_REPORT_SERVER_PORT, logger=None):
        """Initialize server.

        Args:
            results_dir: Directory containing report files
            port: Port to serve on
            logger: Optional logger instance
        """
        self.results_dir = results_dir
        self.port = port
        self.logger = logger or logging.getLogger("LiveReportServer")
        self.clients = []
        self.observer: Optional[Observer] = None
        self.loop = None  # Will be set when server starts

    async def handle_sse(self, reader, writer):
        """Handle Server-Sent Events connection."""
        client_addr = writer.get_extra_info("peername")
        self.logger.info(f"SSE client connected: {client_addr}")

        # Send SSE headers
        response = (
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: text/event-stream\r\n"
            "Cache-Control: no-cache\r\n"
            "Connection: keep-alive\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "\r\n"
        )
        writer.write(response.encode())
        await writer.drain()

        # Register client
        client_queue = asyncio.Queue()
        self.clients.append(client_queue)

        try:
            # Send initial connection message
            await self._send_event(writer, "connected", {"timestamp": datetime.now().isoformat()})

            # Keep connection alive and send updates
            while True:
                try:
                    event_data = await asyncio.wait_for(client_queue.get(), timeout=30)
                    self.logger.info(f"Sending refresh event to client {client_addr}")
                    await self._send_event(writer, "update", event_data)
                except asyncio.TimeoutError:
                    # Send keepalive comment every 30 seconds
                    writer.write(b": keepalive\n\n")
                    await writer.drain()
        except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError):
            self.logger.debug(f"SSE client disconnected: {client_addr}")
        finally:
            self.clients.remove(client_queue)
            try:
                writer.close()
                await writer.wait_closed()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                # Client already disconnected, ignore cleanup errors
                pass

    async def _send_event(self, writer, event_type: str, data: dict):
        """Send an SSE event."""
        message = f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
        writer.write(message.encode())
        await writer.drain()

    async def handle_http(self, reader, writer):
        """Handle HTTP requests."""
        try:
            # Read request
            request_line = await reader.readline()
            request_line = request_line.decode().strip()

            if not request_line:
                writer.close()
                await writer.wait_closed()
                return

            # Parse request
            parts = request_line.split()
            if len(parts) < 2:
                writer.close()
                await writer.wait_closed()
                return

            _method, path = parts[0], parts[1]

            # Read headers (consume but don't parse)
            while True:
                line = await reader.readline()
                if line == b"\r\n" or line == b"\n" or not line:
                    break

            # Route request
            if path == "/events":
                await self.handle_sse(reader, writer)
                return
            elif path == "/" or path == "/report.html":
                await self._serve_file(writer, "report.html", "text/html")
            elif path == "/figures.json":
                await self._serve_file(writer, "figures.json", "application/json")
            elif path == "/metadata.json":
                await self._serve_file(writer, "metadata.json", "application/json")
            elif path == "/current_status.json":
                await self._serve_file(writer, "current_status.json", "application/json")
            elif path == "/fetch_status.json":
                await self._serve_file(writer, "fetch_status.json", "application/json")
            elif path == "/spotlogo_red.png":
                await self._serve_file(writer, "spotlogo_red.png", "image/png")
            else:
                await self._send_404(writer)

        except Exception as e:
            self.logger.error(f"Error handling request: {e}", exc_info=True)
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                pass

    async def _serve_file(self, writer, filename: str, content_type: str):
        """Serve a file from the results directory."""
        filepath = self.results_dir / filename

        if not filepath.exists():
            await self._send_404(writer)
            return

        try:
            content = filepath.read_bytes()
            response = (
                f"HTTP/1.1 200 OK\r\n"
                f"Content-Type: {content_type}\r\n"
                f"Content-Length: {len(content)}\r\n"
                f"Cache-Control: no-cache\r\n"
                f"Access-Control-Allow-Origin: *\r\n"
                f"\r\n"
            )
            writer.write(response.encode())
            writer.write(content)
            await writer.drain()
        except Exception as e:
            self.logger.error(f"Error serving file {filename}: {e}")
            await self._send_404(writer)

    async def _send_404(self, writer):
        """Send 404 response."""
        response = "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n"
        writer.write(response.encode())
        await writer.drain()

    def _on_file_changed(self, filename: str):
        """Handle file change notification (called from watchdog thread)."""
        self.logger.info(f"File changed: {filename} - triggering client refresh")

        # This is called from the watchdog thread, so we need to schedule
        # the notification in the asyncio event loop thread-safely
        if self.loop is not None:
            try:
                # Use call_soon_threadsafe to safely schedule from another thread
                self.loop.call_soon_threadsafe(self._notify_clients)
            except RuntimeError as e:
                self.logger.warning(f"Could not schedule notification: {e}")
        else:
            self.logger.debug("Event loop not yet initialized, skipping notification")

    def _notify_clients(self):
        """Notify all connected clients (called in event loop thread)."""
        # Notify all connected clients with a generic refresh signal
        # When metadata.json changes, both it and figures.json should be reloaded
        event_data = {"type": "refresh", "timestamp": datetime.now().isoformat()}

        # Count successful notifications
        notified = 0
        for client_queue in self.clients:
            try:
                client_queue.put_nowait(event_data)
                notified += 1
            except asyncio.QueueFull:
                self.logger.warning("Client queue full, skipping notification")

        self.logger.info(f"Notified {notified} client(s) to refresh")

    async def start(self):
        """Start the server."""
        # Store the event loop reference for thread-safe callbacks
        # Use get_running_loop() to ensure we get the currently running loop
        try:
            self.loop = asyncio.get_running_loop()
        except RuntimeError:
            # Fallback if called before loop is running
            self.loop = asyncio.get_event_loop()

        # Start file watcher
        # Watch metadata.json and current_status.json for updates
        # These are written last in the update cycle
        event_handler = ReportFileWatcher(
            callback=self._on_file_changed, watch_files={"metadata.json", "current_status.json"}
        )
        self.observer = Observer()
        self.observer.schedule(event_handler, str(self.results_dir), recursive=False)
        self.observer.start()
        self.logger.info(f"Started file watcher on {self.results_dir}")

        # Start HTTP server (0.0.0.0 = all interfaces; use localhost or LAN IP in browser)
        server = await asyncio.start_server(self.handle_http, "0.0.0.0", self.port)

        addrs = ", ".join(str(sock.getsockname()) for sock in server.sockets)
        self.logger.info(f"Live report server listening on 0.0.0.0:{self.port} ({addrs})")
        self.logger.info(
            f"Open http://localhost:{self.port} on this machine, "
            f"or http://<this-host-ip>:{self.port} from another device"
        )

        async with server:
            await server.serve_forever()

    def stop(self):
        """Stop the server."""
        if self.observer:
            self.observer.stop()
            self.observer.join()
        self.logger.info("Server stopped")


def run_server(results_dir: Path, port: int = LIVE_REPORT_SERVER_PORT):
    """Run the live report server.

    Args:
        results_dir: Directory containing report files
        port: Port to serve on
    """
    server = LiveReportServer(results_dir, port)

    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print("\nShutting down server...")
        server.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # Run server
    asyncio.run(server.start())


if __name__ == "__main__":
    # Example usage
    results_dir = Path("data/results/live")
    run_server(results_dir)
