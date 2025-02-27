import asyncio
import json

class MyServer(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == '/api/chat':
            try:
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                request_data = json.loads(post_data.decode('utf-8'))
                message = request_data.get('message', '')
                
                # Send headers for event stream
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.send_header('Cache-Control', 'no-cache')
                self.send_header('Connection', 'keep-alive')
                self.end_headers()
                
                # Run async chat handler
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(self.handle_chat(message))
                finally:
                    asyncio.set_event_loop(None)
                    loop.close()
                
            except Exception as e:
                print(f"Error in do_POST: {str(e)}")
                self.send_response(500)
                self.send_header('Content-Type', 'text/event-stream')
                self.end_headers()
                write_stream(self, {
                    "type": "error",
                    "error": str(e),
                    "status": "error"
                }, "error")
        else:
            self.send_error(404, "Not Found") 