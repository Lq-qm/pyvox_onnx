"""Backend ONNX — narração via `kokoro-onnx` (onnxruntime, somente CPU).

Este módulo dá suporte ao backend "onnx": ele baixa uma única vez o modelo
ONNX e o arquivo de vozes para `./data/kokoro/`, abre a sessão `onnxruntime`
(CPUExecutionProvider) e sintetiza lote a lote (fonemas → ONNX → trim),
devolvendo PCM 16-bit a 24 kHz — mesmo contrato de saída dos demais backends.

Obs.: a síntese é executada diretamente na sessão (via `_run_chunk`) porque o
`kokoro-onnx` 0.4.7 envia `speed` como int32 para exports com `input_ids`,
então o modelo oficial (que espera float) rejeita o input. Os inputs são
montados a partir dos metadados da sessão, o que cobre as duas gerações de export.

Arquivos (release `model-files-v1.1` do `thewh1teagle/kokoro-onnx`):
- `kokoro-v1.0.onnx` (~310 MB; existe também `kokoro-v1.0.int8.onnx` ~108 MB)
- `voices-v1.0.bin`  (~26 MB, 54 vozes em 9 idiomas)

Uso rápido:

    from onnx_backend import OnnxBackend

    with OnnxBackend(voice="pf_dora", lang="pt-br", speed=1.0) as ob:
        pcm, rate = ob.synthesize("Olá, mundo.", voice="pf_dora")
"""
from __future__ import annotations

import time
import urllib.request
from pathlib import Path
from typing import Callable, Optional

SAMPLE_RATE = 24_000

_RELEASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1"
MODEL_FILENAME = "kokoro-v1.0.onnx"
VOICES_FILENAME = "voices-v1.0.bin"
MODEL_URL = f"{_RELEASE}/{MODEL_FILENAME}"
VOICES_URL = f"{_RELEASE}/{VOICES_FILENAME}"

DEFAULT_DATA_DIR = Path("data") / "kokoro"


class OnnxBackendError(RuntimeError):
    """Falha genérica do backend ONNX."""


class OnnxVoiceError(OnnxBackendError):
    """Voz inexistente no arquivo de vozes do ONNX."""


# --------------------------------------------------------------------------- #
# Arquivos do modelo
# --------------------------------------------------------------------------- #
def _download(url: str, dest: Path, description: str) -> None:
    """Baixa `url` para `dest` (arquivo .part intermediário + progresso)."""
    tmp = dest.with_name(dest.name + ".part")
    print(f"baixando {description} ({url})…", flush=True)
    with urllib.request.urlopen(url, timeout=60) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        last = 0.0
        with open(tmp, "wb") as fh:
            while block := resp.read(1 << 20):
                fh.write(block)
                done += len(block)
                now = time.monotonic()
                if now - last >= 0.5 or (total and done >= total):
                    last = now
                    if total:
                        print(f"\r  {done/1e6:8.1f} / {total/1e6:5.0f} MB "
                              f"({done * 100 // total}%)", end="", flush=True)
                    else:
                        print(f"\r  {done/1e6:8.1f} MB", end="", flush=True)
    print()
    tmp.replace(dest)


def ensure_files(
    data_dir: Path | str = DEFAULT_DATA_DIR,
    model_dir: Path | str | None = None,
) -> tuple[Path, Path]:
    """Garante o modelo ONNX + vozes, baixando do release oficial o que faltar.

    Modelo: `model_dir` (arquivo `.onnx` ou diretório que o contenha) >
            `data_dir/kokoro-v1.0.onnx` (baixado se não existir).
    Vozes:  `data_dir/voices-v1.0.bin` (baixado se não existir).

    Retorna `(model_path, voices_path)`.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    model_path: Path | None = None
    if model_dir:
        md = Path(model_dir)
        if md.is_file():
            model_path = md
        elif (md / MODEL_FILENAME).is_file():
            model_path = md / MODEL_FILENAME
        else:
            candidates = sorted(md.glob("*.onnx")) if md.is_dir() else []
            if candidates:
                model_path = candidates[0]

    if model_path is None:
        model_path = data_dir / MODEL_FILENAME
        if not model_path.is_file():
            _download(MODEL_URL, model_path, "modelo ONNX Kokoro-82M")

    voices_path = data_dir / VOICES_FILENAME
    if not voices_path.is_file():
        _download(VOICES_URL, voices_path, "arquivo de vozes")

    return model_path, voices_path


# --------------------------------------------------------------------------- #
# Backend
# --------------------------------------------------------------------------- #
class OnnxBackend:
    """Sessão `kokoro-onnx` com ciclo de vida próprio (somente CPU).

    - `start()`: garante os arquivos e cria a sessão onnxruntime;
    - `synthesize(text, voice)`: retorna (pcm_int16_bytes, sample_rate);
    - `stop()`: libera a sessão.

    Também funciona como context manager: `with OnnxBackend(...) as ob:`.
    """

    def __init__(
        self,
        voice: str,
        *,
        lang: str = "pt-br",
        speed: float = 1.0,
        threads: int | None = None,
        data_dir: Path | str = DEFAULT_DATA_DIR,
        model_dir: Path | str | None = None,
    ) -> None:
        if not 0.5 <= speed <= 2.0:
            raise OnnxBackendError(f"velocidade {speed} fora do intervalo permitido (0.5 a 2.0)")
        self.voice = voice
        self.lang = lang
        self.speed = speed
        self.threads = threads
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir) if model_dir else None
        self._model: Optional[object] = None

    # ------------------------------------------------------------------ #
    # Ciclo de vida
    # ------------------------------------------------------------------ #
    def start(self) -> None:
        """Baixa (se preciso) modelo/vozes e abre a sessão onnxruntime (CPU)."""
        if self._model is not None:
            return
        import onnxruntime as rt
        from kokoro_onnx import Kokoro

        model_path, voices_path = ensure_files(self.data_dir, self.model_dir)
        opts = rt.SessionOptions()
        opts.log_severity_level = 2  # WARNING (silencia o log INFO do onnxruntime)
        if self.threads:
            opts.intra_op_num_threads = max(1, self.threads)
            opts.inter_op_num_threads = 1
        sess = rt.InferenceSession(
            str(model_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._model = Kokoro.from_session(sess, str(voices_path))

    def stop(self) -> None:
        """Libera a sessão (idempotente)."""
        self._model = None

    # ------------------------------------------------------------------ #
    # Síntese
    # ------------------------------------------------------------------ #
    def _run_chunk(self, phonemes: str, style_bank, speed: float):
        """Executa um lote de fonemas na sessão ONNX.

        O `kokoro-onnx` 0.4.7 hard-codifica `speed` como int32 para exports
        com entrada `input_ids`, mas o modelo oficial (`kokoro-v1.0.onnx`)
        declara `speed` como float — então os inputs são montados a partir dos
        próprios metadados da sessão (funciona com as duas gerações de export).
        """
        import numpy as np

        sess = self._model.sess
        infos = {i.name: i.type for i in sess.get_inputs()}
        ids_key = "input_ids" if "input_ids" in infos else "tokens"
        npd = {
            "tensor(int64)": np.int64,
            "tensor(int32)": np.int32,
            "tensor(float16)": np.float16,
            "tensor(float)": np.float32,
            "tensor(double)": np.float64,
        }
        tokens = self._model.tokenizer.tokenize(phonemes)
        assert len(tokens) <= 510, f"lote de fonemas excede o contexto (N={len(tokens)})"
        inputs = {
            ids_key: np.array([[0, *[int(t) for t in tokens], 0]],
                              dtype=npd.get(infos.get(ids_key), np.int64)),  # shape [1, N]
            "style": np.asarray(style_bank[len(tokens)],
                                dtype=npd.get(infos.get("style"), np.float32)),
            "speed": np.array([speed], dtype=npd.get(infos.get("speed"), np.float32)),
        }
        return sess.run(None, inputs)[0]

    def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        on_chunk: Optional[Callable[[int], None]] = None,
    ) -> tuple[bytes, int]:
        """Sintetiza `text` e retorna (pcm_int16_bytes, sample_rate).

        `on_chunk` (opcional) é chamada a cada bloco de áudio produzido,
        recebendo o total de bytes de PCM acumulado — útil para progresso.
        """
        import numpy as np
        from kokoro_onnx.trim import trim as trim_audio

        if self._model is None:
            self.start()
        name = voice or self.voice
        if name not in self._model.voices:
            sample = ", ".join(sorted(self._model.voices)[:6])
            raise OnnxVoiceError(
                f"voz {name!r} não existe no arquivo de vozes ONNX "
                f"({sample}, …); use --list-voices"
            )

        model = self._model
        style_bank = model.voices[name]
        phonemes = model.tokenizer.phonemize(text, self.lang)
        # lote por lote (mesma estratégia de `create_stream`, sem o assíncrono)
        batches = model._split_phonemes(phonemes)  # noqa: SLF001 — estável em kokoro-onnx 0.4.x

        pcm = bytearray()
        for batch in batches:
            audio = self._run_chunk(batch, style_bank, self.speed)
            audio, _ = trim_audio(audio)
            chunk = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
            pcm += chunk
            if on_chunk is not None:
                on_chunk(len(pcm))
        return bytes(pcm), SAMPLE_RATE

    # ------------------------------------------------------------------ #
    # Context manager
    # ------------------------------------------------------------------ #
    def __enter__(self) -> "OnnxBackend":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()
