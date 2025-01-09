from flask import Flask, send_file, Response, render_template_string
import threading
import queue
import os
import json

class AudioStreamer:
    def __init__(self):
        self.app = Flask(__name__)
        self.audio_queue = queue.Queue()
        self.current_status = "Initializing..."
        self.progress = {
            "current_step": "initializing",
            "total_segments": 0,
            "current_segment": 0,
            "steps": {
                "pdf_processing": {"status": "pending", "progress": 0},
                "conversation": {"status": "pending", "progress": 0},
                "audio": {"status": "pending", "progress": 0}
            }
        }
        self.setup_routes()

    def setup_routes(self):
        @self.app.route('/')
        def index():
            return render_template_string('''
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Podcast Streamer</title>
                    <style>
                        body { 
                            font-family: Arial, sans-serif; 
                            margin: 20px;
                            max-width: 800px;
                            margin: 0 auto;
                            padding: 20px;
                        }
                        .progress-container {
                            margin: 20px 0;
                            padding: 15px;
                            border-radius: 8px;
                            background: #f5f5f5;
                        }
                        .step {
                            margin: 10px 0;
                        }
                        .step-header {
                            display: flex;
                            justify-content: space-between;
                            margin-bottom: 5px;
                        }
                        .progress-bar {
                            width: 100%;
                            height: 20px;
                            background-color: #ddd;
                            border-radius: 10px;
                            overflow: hidden;
                        }
                        .progress-fill {
                            height: 100%;
                            background-color: #4CAF50;
                            transition: width 0.3s ease;
                            border-radius: 10px;
                        }
                        .status-badge {
                            padding: 3px 8px;
                            border-radius: 12px;
                            font-size: 12px;
                            font-weight: bold;
                        }
                        .status-pending { background: #ffd700; }
                        .status-in-progress { background: #87CEEB; }
                        .status-completed { background: #90EE90; }
                        .status-error { background: #ffcccb; }
                        #audioPlayer {
                            width: 100%;
                            margin: 20px 0;
                        }
                        .current-status {
                            margin: 20px 0;
                            padding: 10px;
                            background: #e8f5e9;
                            border-radius: 5px;
                        }
                    </style>
                </head>
                <body>
                    <h1>PDF to Podcast Converter</h1>
                    
                    <div class="current-status">
                        <strong>Status:</strong> <span id="status">Initializing...</span>
                    </div>

                    <div class="progress-container">
                        <div class="step">
                            <div class="step-header">
                                <span>PDF Processing</span>
                                <span id="pdf-status" class="status-badge status-pending">Pending</span>
                            </div>
                            <div class="progress-bar">
                                <div id="pdf-progress" class="progress-fill" style="width: 0%"></div>
                            </div>
                        </div>

                        <div class="step">
                            <div class="step-header">
                                <span>Conversation Generation</span>
                                <span id="conversation-status" class="status-badge status-pending">Pending</span>
                            </div>
                            <div class="progress-bar">
                                <div id="conversation-progress" class="progress-fill" style="width: 0%"></div>
                            </div>
                        </div>

                        <div class="step">
                            <div class="step-header">
                                <span>Audio Generation</span>
                                <span id="audio-status" class="status-badge status-pending">Pending</span>
                            </div>
                            <div class="progress-bar">
                                <div id="audio-progress" class="progress-fill" style="width: 0%"></div>
                            </div>
                        </div>
                    </div>

                    <audio id="audioPlayer" controls autoplay>
                        <source src="/stream" type="audio/mp3">
                    </audio>

                    <script>
                        const audio = document.getElementById('audioPlayer');
                        const status = document.getElementById('status');
                        
                        function updateProgressBars(progress) {
                            // Update PDF Processing
                            const pdfProgress = document.getElementById('pdf-progress');
                            const pdfStatus = document.getElementById('pdf-status');
                            pdfProgress.style.width = progress.steps.pdf_processing.progress + '%';
                            updateStatusBadge(pdfStatus, progress.steps.pdf_processing.status);

                            // Update Conversation Generation
                            const convProgress = document.getElementById('conversation-progress');
                            const convStatus = document.getElementById('conversation-status');
                            convProgress.style.width = progress.steps.conversation.progress + '%';
                            updateStatusBadge(convStatus, progress.steps.conversation.status);

                            // Update Audio Generation
                            const audioProgress = document.getElementById('audio-progress');
                            const audioStatus = document.getElementById('audio-status');
                            audioProgress.style.width = progress.steps.audio.progress + '%';
                            updateStatusBadge(audioStatus, progress.steps.audio.status);
                        }

                        function updateStatusBadge(element, status) {
                            element.className = 'status-badge';
                            element.classList.add('status-' + status.toLowerCase());
                            element.textContent = status.charAt(0).toUpperCase() + status.slice(1);
                        }

                        // Check for new audio every second
                        setInterval(() => {
                            if (!audio.src || audio.ended) {
                                fetch('/stream')
                                    .then(response => {
                                        if (response.ok) {
                                            audio.src = '/stream';
                                            audio.load();
                                            audio.play().catch(console.error);
                                        }
                                    })
                                    .catch(console.error);
                            }
                        }, 1000);

                        // Update status and progress
                        setInterval(() => {
                            fetch('/progress')
                                .then(response => response.json())
                                .then(progress => {
                                    status.textContent = progress.current_status;
                                    updateProgressBars(progress);
                                })
                                .catch(console.error);
                        }, 1000);
                    </script>
                </body>
                </html>
            ''')

        @self.app.route('/stream')
        def stream():
            try:
                audio_path = self.audio_queue.get(timeout=1)
                if os.path.exists(audio_path):
                    return send_file(audio_path, mimetype='audio/mp3')
                return Response("Audio file not found", status=404)
            except queue.Empty:
                return Response("No audio available", status=404)
            except Exception as e:
                print(f"Streaming error: {str(e)}")
                return Response("Error streaming audio", status=500)

        @self.app.route('/progress')
        def get_progress():
            return Response(
                json.dumps({
                    "current_status": self.current_status,
                    **self.progress
                }), 
                mimetype='application/json'
            )

    def start(self, port=5000):
        self.flask_thread = threading.Thread(
            target=self.app.run, 
            kwargs={'port': port, 'host': '0.0.0.0'}, 
            daemon=True
        )
        self.flask_thread.start()
        print(f"\nStreaming server started at http://localhost:{port}")

    def add_segment(self, audio_path: str):
        if os.path.exists(audio_path):
            self.audio_queue.put(audio_path)
        else:
            print(f"Warning: Audio segment not found: {audio_path}")

    def update_status(self, status: str):
        self.current_status = status

    def update_progress(self, step: str, status: str, progress: float):
        """Update progress for a specific step."""
        if step in self.progress["steps"]:
            self.progress["steps"][step]["status"] = status
            self.progress["steps"][step]["progress"] = min(100, max(0, progress))

    def set_total_segments(self, total: int):
        """Set the total number of segments to be processed."""
        self.progress["total_segments"] = total

    def update_current_segment(self, current: int):
        """Update the current segment being processed."""
        self.progress["current_segment"] = current