"""Serve generated gallery files and exact RGB source frames on loopback."""

from __future__ import annotations

import json
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, urlsplit

import cv2

from src.utils.checksum import dual_sha256

if TYPE_CHECKING:
    from src.tennis_scene.scripts.visualize_component_store import Review


def make_server(review: Review, *, port: int) -> ThreadingHTTPServer:
    """Expose only the bound source camera/frame, never a caller-supplied path."""
    lock = threading.Lock()
    initial_index = dual_sha256(review.index_path)
    if initial_index != review.snapshot["index_sha256"]:
        raise ValueError("Index changed after gallery capture")
    for source in review.snapshot["sources"]:
        if source["available"] and dual_sha256(review.videos[source["camera_id"]]) != source["sha256"]:
            raise ValueError("Source changed after gallery capture")
    media_stats = {camera: path.stat() for camera, path in review.videos.items() if path.is_file()}

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, directory=str(review.output), **kwargs)

        def do_GET(self) -> None:
            route = urlsplit(self.path)
            if route.path != "/api/frame":
                super().do_GET()
                return
            try:
                query = parse_qs(route.query, strict_parsing=True)
                if set(query) != {"camera", "frame"} or any(len(values) != 1 for values in query.values()):
                    raise ValueError("Use one camera and one source frame")
                camera, frame = query["camera"][0], int(query["frame"][0])
                if camera not in review.videos or not 0 <= frame < review.frame_count:
                    raise ValueError("Camera or source frame outside the bound clip")
            except (ValueError, KeyError) as error:
                self.send_error(400, str(error))
                return
            try:
                path = review.videos[camera]
                if camera not in media_stats:
                    raise OSError(f"Source RGB is unavailable: {camera}")
                stat, original = path.stat(), media_stats[camera]
                if (stat.st_mtime_ns, stat.st_size) != (original.st_mtime_ns, original.st_size) or dual_sha256(review.index_path) != initial_index:
                    raise OSError("Source/index changed after gallery capture; rebuild the gallery")
                with lock:
                    image = review.frame(camera, frame)
                okay, encoded = cv2.imencode(".jpg", image)
                if not okay:
                    raise OSError("Could not encode source frame")
            except OSError as error:
                self.send_error(409, str(error))
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Source-Frame", str(frame))
            self.send_header("X-Source-Camera", camera)
            self.end_headers()
            self.wfile.write(encoded.tobytes())

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    return server


def serve(review: Review, *, port: int) -> None:
    server = make_server(review, port=port)
    print(json.dumps({"url": f"http://127.0.0.1:{server.server_port}", "store": str(review.root), "mode": "read-only"}), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
