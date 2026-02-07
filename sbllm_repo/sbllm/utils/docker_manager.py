import subprocess
import logging
import os
import atexit
import time

logger = logging.getLogger(__name__)

class DockerManager:
    _instance = None
    _container_name = "riscv-eval-session"
    _image_name = "riscv-opt-env"
    _is_running = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DockerManager, cls).__new__(cls)
        return cls._instance

    def initialize(self, project_root: str):
        """Starts a background container if not already running."""
        if self._is_running:
            return

        # Double check if container exists/runs from a previous interrupted session
        try:
            check_cmd = ["docker", "ps", "-q", "-f", f"name={self._container_name}"]
            result = subprocess.run(check_cmd, capture_output=True, text=True)
            if result.stdout.strip():
                logger.info(f"Container {self._container_name} already running.")
                self._is_running = True
                return
        except Exception:
            pass

        logger.info(f"Starting persistent Docker container: {self._container_name}...")
        abs_root = os.path.abspath(project_root).replace('\\', '/')
        
        # Start background container
        # tail -f /dev/null keeps the container alive
        cmd = [
            "docker", "run", "-d", 
            "--rm", 
            "--name", self._container_name, 
            "-v", f"{abs_root}:/work", 
            self._image_name, 
            "tail", "-f", "/dev/null"
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            self._is_running = True
            atexit.register(self.cleanup)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to start Docker container: {e.stderr}")
            # Fallback handled by callers (they might try run --rm as backup)

    def exec(self, command: list, workdir: str = "/work", timeout: int = 60, input_data: str = None) -> subprocess.CompletedProcess:
        """Executes a command inside the persistent container."""
        if not self._is_running:
            # Emergency fallback logic or just error out
            logger.warning("DockerManager not running, falling back to one-shot run...")
            # Note: This is a placeholder, actual evaluators should handle fallback if needed
            raise RuntimeError("Docker persistent container not initialized")

        exec_cmd = ["docker", "exec", "-w", workdir, "-i", self._container_name] + command
        
        return subprocess.run(exec_cmd, input=input_data, capture_output=True, text=True, timeout=timeout)

    def cleanup(self):
        """Stops and removes the background container."""
        if not self._is_running:
            return
        
        logger.info(f"Cleaning up Docker container: {self._container_name}...")
        try:
            subprocess.run(["docker", "stop", self._container_name], check=True, capture_output=True)
            self._is_running = False
        except Exception as e:
            logger.error(f"Error stopping container: {e}")

# Global instance
docker_manager = DockerManager()
