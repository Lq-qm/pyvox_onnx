"""Backend Wyoming — narração via servidor `wyoming-kokoro-torch`.

Este módulo dá suporte ao backend "wyoming": ele sobe o servidor
`wyoming-kokoro-torch` como subprocesso (TCP local), conversa com ele pelo
protocolo Wyoming (WebSocket-like, JSON + PCM cru) e devolve o áudio.

Uso rápido:

    from wyoming_backend import WyomingBackend, find_model_dir

    with WyomingBackend(voice="pf_dora", speed=1.0) as wb:
        pcm, rate = wb.synthesize("Olá, mundo.", voice="pf_dora")
"""
from __future__ import annotations

import asyncio
import glob
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Optional

SAMPLE_RATE = 24_000
HOST = "127.0.0.1"

# Diretório local para vozes baixadas/instaladas pelo servidor
DEFAULT_DATA_DIR = Path("data") / "kokoro"


class WyomingBackendError(RuntimeError):
    """Falha genérica do backend Wyoming."""


class WyomingSynthesisError(WyomingBackendError):
    """O servidor devolveu um erro de síntese."""


def find_model_dir() -> Path:
    """Localiza o snapshot do Kokoro-82M no cache do Hugging Face.

    O snapshot precisa conter `kokoro-v1_0.pth` e `config.json`.
    """
    pattern = os.path.expanduser(
        "~/.cache/huggingface/hub/models--hexgrad--Kokoro-82M/snapshots/*"
    )
    for snap in sorted(glob.glob(pattern)):
        if (Path(snap) / "kokoro-v1_0.pth").exists() and (Path(snap) / "config.json").exists():
            return Path(snap)
    raise WyomingBackendError(
        "modelo Kokoro-82M não encontrado no cache do Hugging Face "
        f"({pattern}); rode primeiro um backend torch ou defina --model-dir"
    )


def free_port(host: str = HOST) -> int:
    """Retorna uma porta TCP livre na interface local."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


def _wait_port(host: str, port: int, timeout: float, proc: subprocess.Popen) -> None:
    """Aguarda a porta aceitar conexões; falha cedo se o servidor morrer no boot."""
    deadline = time.monotonic() + timeout
    last_err: Exception | None = None
    while True:
        if proc.poll() is not None:
            # servidor terminou antes de abrir a porta (ex.: voz padrão inválida)
            raise WyomingBackendError(
                f"servidor Wyoming terminou ao iniciar (rc={proc.returncode}); "
                "verifique --voice/--model-dir/--data-dir"
            )
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return
        except OSError as e:
            last_err = e
            if time.monotonic() >= deadline:
                raise WyomingBackendError(
                    f"servidor Wyoming não abriu a porta {host}:{port} em {timeout:.0f}s ({last_err})"
                )
            time.sleep(0.25)


class WyomingBackend:
    """Cliente do servidor `wyoming-kokoro-torch` com ciclo de vida próprio.

    - `start()`: sobe o servidor subprocesso e espera a porta;
    - `synthesize(text, voice)`: retorna (pcm_int16_bytes, sample_rate);
    - `stop()`: derruba o servidor.

    Também funciona como context manager: `with WyomingBackend(...) as wb:`.
    """

    def __init__(
        self,
        voice: str,  # voz padrão do servidor (deve existir; a voz por requisição é passada em synthesize)
        *,
        speed: float = 1.0,
        device: str = "cpu",
        host: str = HOST,
        port: Optional[int] = None,
        model_dir: Optional[str | Path] = None,
        data_dir: Optional[str | Path] = None,
        samples_per_chunk: int = 1024,
        ready_timeout: float = 180.0,
        python: Optional[str] = None,
    ) -> None:
        self.voice = voice
        self.speed = speed
        self.device = device
        self.host = host
        self.port = port or free_port(host)
        self.model_dir = Path(model_dir) if model_dir else find_model_dir()
        self.data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.samples_per_chunk = samples_per_chunk
        self.ready_timeout = ready_timeout
        self.python = python or sys.executable
        self._proc: Optional[subprocess.Popen] = None
        self._uri = f"tcp://{self.host}:{self.port}"

    # ------------------------------------------------------------------ #
    # Ciclo de vida
    # ------------------------------------------------------------------ #
    def _server_command(self) -> list[str]:
        return [
            self.python, "-m", "wyoming_kokoro_torch",
            "--voice", self.voice,
            "--uri", self._uri,
            "--data-dir", str(self.model_dir),
            "--data-dir", str(self.data_dir),
            "--download-dir", str(self.data_dir),
            "--speed", str(self.speed),
            "--device", self.device,
            "--samples-per-chunk", str(self.samples_per_chunk),
        ]

    def start(self) -> None:
        """Sobe o servidor Wyoming e aguarda a porta ficar disponível."""
        if self._proc is not None:
            return
        self._proc = subprocess.Popen(
            self._server_command(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        try:
            _wait_port(self.host, self.port, self.ready_timeout, self._proc)
        except WyomingBackendError as e:
            self.stop()
            tail = self._server_tail(20)
            raise WyomingBackendError(f"{e}\nlog do servidor:\n{tail}" if tail else str(e)) from e
        except Exception:
            self.stop()
            raise

    def stop(self) -> None:
        """Encerra o servidor (idempotente)."""
        proc, self._proc = self._proc, None
        if proc is None:
            return
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

    def _server_tail(self, lines: int = 15) -> str:
        """Últimas linhas do log do servidor (para diagnostics)."""
        if self._proc is None or self._proc.stdout is None:
            return ""
        try:
            data = self._proc.stdout.read().splitlines()
            return "\n".join(data[-lines:])
        except Exception:
            return ""

    # ------------------------------------------------------------------ #
    # Síntese
    # ------------------------------------------------------------------ #
    def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        on_chunk: Optional[Callable[[int], None]] = None,
    ) -> tuple[bytes, int]:
        """Sintetiza `text` e retorna (pcm_int16_bytes, sample_rate).

        `on_chunk` (opcional) é chamada a cada bloco de áudio recebido,
        recebendo o total de bytes de PCM acumulado — útil para progresso.
        """
        if self._proc is None:
            self.start()
        return asyncio.run(self._synthesize_async(text, voice, on_chunk))

    async def _synthesize_async(
        self,
        text: str,
        voice: Optional[str],
        on_chunk: Optional[Callable[[int], None]],
    ) -> tuple[bytes, int]:
        from wyoming.audio import AudioChunk, AudioStart, AudioStop
        from wyoming.client import AsyncClient
        from wyoming.error import Error
        from wyoming.tts import Synthesize, SynthesizeVoice

        name = voice or self.voice
        audio = bytearray()
        rate: Optional[int] = None

        try:
            async with AsyncClient.from_uri(self._uri) as client:
                req = Synthesize(text=text, voice=SynthesizeVoice(name=name))
                await client.write_event(req.event())
                while True:
                    event = await client.read_event()
                    if event is None:
                        raise WyomingBackendError("servidor fechou a conexão durante a síntese")
                    if Error.is_type(event.type):
                        raise WyomingSynthesisError(
                            f"{event.data.get('code', 'erro')}: {event.data.get('text', '')}"
                        )
                    if AudioStart.is_type(event.type):
                        rate = int(event.data["rate"])
                    elif AudioChunk.is_type(event.type):
                        audio += event.payload or b""
                        if on_chunk is not None:
                            on_chunk(len(audio))
                    elif AudioStop.is_type(event.type):
                        break
        except (ConnectionRefusedError, ConnectionResetError, BrokenPipeError) as e:
            tail = self._server_tail()
            raise WyomingBackendError(f"falha de conexão com o servidor ({e})\nlog do servidor:\n{tail}") from e

        if rate is None:
            raise WyomingBackendError("servidor não informou a taxa de amostragem")
        return bytes(audio), rate

    # ------------------------------------------------------------------ #
    # Context manager
    # ------------------------------------------------------------------ #
    def __enter__(self) -> "WyomingBackend":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()
