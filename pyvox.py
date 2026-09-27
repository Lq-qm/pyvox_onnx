#!/usr/bin/env python3
"""pyvox — CLI de narração de textos (.txt) com o motor Kokoro TTS (CPU).

Exemplos:
    # Narrar um arquivo (padrão: português brasileiro, voz pf_dora)
    python pyvox.py "meu_texto.txt" -o narracao.wav

    # Narrador masculino / narradora feminina
    python pyvox.py texto.txt --narrador masculino
    python pyvox.py texto.txt --narrador feminino --speed 1.1 --play

    # Escolher o backend de inferência (auto | torch | wyoming)
    python pyvox.py texto.txt --backend wyoming
    python pyvox.py texto.txt --backend torch --threads 8

    # Listar vozes disponíveis para o idioma
    python pyvox.py --list-voices --lang pt-br

    # Texto direto, sem arquivo
    python pyvox.py --text "Olá, mundo."
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

SAMPLE_RATE = 24_000
DEFAULT_REPO = "hexgrad/Kokoro-82M"

# Mapa de idiomas -> código interno do Kokoro
LANGS: dict[str, str] = {
    "pt-br": "p",
    "en-us": "a",
    "en-gb": "b",
    "es": "e",
    "fr-fr": "f",
    "hi": "h",
    "it": "i",
    "ja": "j",
    "zh": "z",
}

# Vozes padrão por idioma (oficiais do Kokoro-82M)
DEFAULT_VOICES: dict[str, str] = {
    "p": "pf_dora",
    "a": "af_heart",
    "b": "bf_emma",
    "e": "ef_dora",
    "f": "ff_siwis",
    "h": "hf_alpha",
    "i": "if_sara",
    "j": "jf_alpha",
    "z": "zf_xiaobei",
}

# Narrador (m) e narradora (f) por idioma — escolha curada do Kokoro-82M.
# Obs.: fr-fr só tem voz feminina disponível no modelo (ff_siwis).
NARRATOR_VOICES: dict[str, dict[str, str]] = {
    "p": {"m": "pm_alex", "f": "pf_dora"},
    "a": {"m": "am_michael", "f": "af_heart"},
    "b": {"m": "bm_fable", "f": "bf_emma"},
    "e": {"m": "em_alex", "f": "ef_dora"},
    "f": {"f": "ff_siwis"},
    "h": {"m": "hm_omega", "f": "hf_alpha"},
    "i": {"m": "im_nicola", "f": "if_sara"},
    "j": {"m": "jm_kumo", "f": "jf_alpha"},
    "z": {"m": "zm_yunjian", "f": "zf_xiaobei"},
}


def _gender(value: str) -> str:
    """Normaliza a opção --narrador para 'm' ou 'f'."""
    v = value.strip().lower()
    if v in ("m", "masculino", "masculine", "homem", "male"):
        return "m"
    if v in ("f", "feminino", "feminine", "mulher", "female"):
        return "f"
    raise argparse.ArgumentTypeError(f"use 'm' (masculino) ou 'f' (feminino), recebido: {value!r}")


# --------------------------------------------------------------------------- #
# Argumentos
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="pyvox",
        description="Narra um arquivo .txt usando o motor Kokoro TTS (somente CPU por enquanto).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "exemplos:\n"
            '  pyvox.py "meu texto.txt" -o saida.wav --play\n'
            "  pyvox.py texto.txt --narrador masculino\n"
            "  pyvox.py texto.txt --narrador feminino --speed 1.1\n"
            "  pyvox.py --list-voices --lang pt-br\n"
            '  pyvox.py --text "Olá, mundo."\n'
        ),
    )
    p.add_argument("file", nargs="?", help="arquivo .txt a ser narrado (opcional se --text for usado)")
    p.add_argument("--text", help="texto para narrar diretamente (ignora o arquivo)")
    p.add_argument("-l", "--lang", default="pt-br", choices=sorted(LANGS),
                   help="idioma do texto (padrão: pt-br)")
    p.add_argument("-g", "--narrador", type=_gender,
                   help="narrador: m/masculino ou f/feminino (pt-br: m=pm_alex, f=pf_dora)")
    p.add_argument("-v", "--voice",
                   help=f"voz exata a usar (ex.: {DEFAULT_VOICES['p']} para pt-br; --narrador ou --list-voices)")
    p.add_argument("-s", "--speed", type=float, default=1.0,
                   help="velocidade da fala, 0.5 = metade, 1.5 = 1.5x (padrão: 1.0)")
    p.add_argument("-o", "--output", help="arquivo .wav de saída (padrão: <arquivo>.wav ou narração.wav)")
    p.add_argument("--threads", type=int, default=os.cpu_count(),
                   help="número de threads da CPU (padrão: todos os núcleos)")
    p.add_argument("--max-chars", type=int, default=None,
                   help="limitar a narração aos N primeiros caracteres (útil para testes)")
    p.add_argument("--play", action="store_true", help="tocar o áudio ao final (ffplay/mpv/aplay)")
    p.add_argument("--list-voices", action="store_true", help="listar vozes disponíveis e sair")
    p.add_argument("--backend", default="auto", choices=["auto", "torch", "wyoming"],
                   help="motor de inferência: auto (wyoming se disponível, senão torch; padrão), "
                        "torch (in-process, CPU) ou wyoming (servidor wyoming-kokoro-torch)")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"],
                   help="device do backend wyoming (padrão: cpu; o backend torch usa sempre CPU por enquanto)")
    p.add_argument("--model-dir", help="diretório com o modelo Kokoro-82M (padrão: cache do Hugging Face) — backend wyoming")
    p.add_argument("--data-dir", help="diretório local de vozes (padrão: ./data/kokoro) — backend wyoming")
    p.add_argument("--wyoming-port", type=int, default=None,
                   help="porta TCP do servidor Wyoming (padrão: porta livre automática)")
    return p.parse_args(argv)


# --------------------------------------------------------------------------- #
# Vozes
# --------------------------------------------------------------------------- #
def list_voices(lang_code: str | None = None, repo_id: str = DEFAULT_REPO) -> list[str]:
    """Busca as vozes do idioma no Hugging Face (com fallback offline).

    Vozes seguem o padrão <idioma><gênero>_, ex.: pf_dora (pt feminina), pm_alex (pt masculina).
    """
    prefixes = (f"{lang_code}f", f"{lang_code}m") if lang_code else None

    def _match(voice: str) -> bool:
        return prefixes is None or voice.startswith(prefixes)

    url = f"https://huggingface.co/api/models/{repo_id}/tree/main/voices"
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            data = json.load(resp)
        voices = sorted(d["path"].rsplit("/", 1)[-1][:-3]
                        for d in data if d["path"].endswith(".pt") and _match(d["path"].rsplit("/", 1)[-1][:-3]))
        if voices:
            return voices
    except Exception as e:  # offline, repo inexistente, etc.
        print(f"aviso: não foi possível consultar o Hugging Face ({e}); mostrando só as vozes padrão", file=sys.stderr)
    pool = set(DEFAULT_VOICES.values()) | set(v for d in NARRATOR_VOICES.values() for v in d.values())
    return sorted(v for v in pool if _match(v))


# --------------------------------------------------------------------------- #
# Síntese
# --------------------------------------------------------------------------- #
def synthesize(text: str, lang: str, voice: str, speed: float, output: str, threads: int) -> dict:
    """Gera o áudio com Kokoro e grava em WAV. Retorna estatísticas da síntese."""
    import os as _os
    _os.environ.setdefault("HF_HUB_DISABLE_UNAUTHENTICATED_WARNING", "1")

    import warnings
    warnings.simplefilter("ignore")  # silencia warnings internos do torch no init do modelo

    import torch  # import tardio: mantém --list-voices/--help rápidos
    torch.set_num_threads(max(1, threads))

    from loguru import logger
    logger.remove()  # silencia os logs internos do Kokoro/misaki

    from kokoro import KPipeline
    import soundfile as sf

    lang_code = LANGS[lang]
    print(f"carregando modelo Kokoro-82M ({lang}, voz={voice}, cpu)…")
    t0 = time.time()
    pipeline = KPipeline(lang_code=lang_code, repo_id=DEFAULT_REPO, device="cpu")
    t_load = time.time() - t0
    print(f"modelo pronto em {t_load:.1f}s")

    t0 = time.time()
    audio_seconds = 0.0
    chunks = 0
    writer = sf.SoundFile(output, samplerate=SAMPLE_RATE, channels=1, subtype="PCM_16", mode="w")
    try:
        for result in pipeline(text, voice=voice, speed=speed):
            audio = result.audio
            if audio is None:
                continue
            audio = audio.detach().cpu().float()
            writer.write(audio.numpy())
            audio_seconds += audio.numel() / SAMPLE_RATE
            chunks += 1
            preview = (result.graphemes or "").strip().replace("\n", " ")
            preview = preview[:48] + ("…" if len(preview) > 48 else "")
            line = (f"\r[{chunks:>4}] {audio_seconds:8.1f}s de áudio | "
                    f"{time.time() - t0:8.1f}s cpu | “{preview}”")
            print(line[:120], end="", flush=True)
    finally:
        writer.close()
    print()
    elapsed = time.time() - t0
    return {
        "output": output,
        "audio_seconds": audio_seconds,
        "chunks": chunks,
        "elapsed": elapsed,
        "rtf": (elapsed / audio_seconds) if audio_seconds else float("inf"),
    }


def synthesize_wyoming(
    text: str, voice: str, speed: float, output: str,
    default_voice: str | None = None,
    device: str = "cpu", model_dir: str | None = None, data_dir: str | None = None,
    port: int | None = None,
) -> dict:
    """Gera o áudio via servidor Wyoming (wyoming-kokoro-torch) e grava em WAV.

    `default_voice` é a voz padrão do servidor (precisa existir — ele valida
    no boot); a voz efetivamente usada é `voice`, enviada por requisição.
    """
    import numpy as np
    import soundfile as sf
    from wyoming_backend import WyomingBackend

    wb = WyomingBackend(
        default_voice or voice, speed=speed, device=device,
        model_dir=model_dir, data_dir=data_dir, port=port,
    )
    print(f"subindo servidor Wyoming (voz={voice}, device={device})…")
    t0 = time.time()
    with wb:
        t_up = time.time() - t0
        print(f"servidor no ar em {t_up:.1f}s (tcp://{wb.host}:{wb.port})")

        t0 = time.time()
        chunks = 0
        last_reported = -1.0

        def on_chunk(total_bytes: int) -> None:
            nonlocal chunks, last_reported
            chunks += 1
            secs = total_bytes / 2 / SAMPLE_RATE
            if secs - last_reported < 0.2 and total_bytes > 0:
                return  # throttling: evita redraw a cada chunk de ~40 ms
            last_reported = secs
            print(f"\r[wyoming] {secs:8.1f}s de áudio | {time.time() - t0:8.1f}s | "
                  f"{chunks} chunks", end="", flush=True)

        pcm, rate = wb.synthesize(text, voice=voice, on_chunk=on_chunk)
    print()

    audio_seconds = len(pcm) // 2 / rate
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    sf.write(output, samples, rate, subtype="PCM_16")

    elapsed = time.time() - t0
    return {
        "output": output,
        "audio_seconds": audio_seconds,
        "chunks": chunks,
        "elapsed": elapsed,
        "rtf": (elapsed / audio_seconds) if audio_seconds else float("inf"),
        "backend": "wyoming",
    }


# --------------------------------------------------------------------------- #
# Reprodução
# --------------------------------------------------------------------------- #
def play(path: str) -> bool:
    for cmd in (
        ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", path],
        ["mpv", "--no-video", "--really-quiet", path],
        ["aplay", "-q", path],
    ):
        if shutil.which(cmd[0]):
            subprocess.call(cmd)
            return True
    return False


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # --list-voices: não precisa carregar o modelo
    if args.list_voices:
        lang_code = LANGS[args.lang]
        default = DEFAULT_VOICES.get(lang_code, "")
        roles = {v: ("narrador" if g == "m" else "narradora")
                 for g, v in NARRATOR_VOICES.get(lang_code, {}).items()}
        print(f"vozes disponíveis para {args.lang} ({DEFAULT_REPO}):")
        for v in list_voices(lang_code):
            mark = f"  ({roles[v]})" if v in roles else ("  (padrão)" if v == default else "")
            print(f"  {v}{mark}")
        return 0

    # Texto a narrar
    if args.text is not None:
        text = args.text
        base = "narração"
    elif args.file:
        p = Path(args.file)
        if not p.exists():
            print(f"erro: arquivo não encontrado: {p}", file=sys.stderr)
            return 1
        if p.suffix.lower() != ".txt":
            print(f"aviso: {p.name} não termina em .txt (tentando mesmo assim)", file=sys.stderr)
        text = p.read_text(encoding="utf-8", errors="replace")
        base = p.with_suffix("").name
    else:
        print("erro: informe um arquivo .txt ou use --text", file=sys.stderr)
        return 1

    if args.max_chars is not None:
        text = text[: args.max_chars]
    if not text.strip():
        print("erro: o texto está vazio", file=sys.stderr)
        return 1

    output = args.output or f"{base}.wav"

    # Voz: --voice (exata) > --narrador (m/f) > padrão do idioma
    lang_code = LANGS[args.lang]
    if args.voice:
        voice = args.voice
    elif args.narrador:
        candidates = NARRATOR_VOICES.get(lang_code, {})
        if args.narrador not in candidates:
            nome = "masculino" if args.narrador == "m" else "feminino"
            print(f"erro: o idioma {args.lang} não tem voz {nome} no Kokoro-82M "
                  f"(use --voice para escolher outra)", file=sys.stderr)
            return 1
        voice = candidates[args.narrador]
    else:
        voice = DEFAULT_VOICES[lang_code]

    # Resolve o backend (auto = wyoming se instalado, senão torch)
    backend = args.backend
    if backend == "auto":
        try:
            import wyoming_kokoro_torch  # noqa: F401
            backend = "wyoming"
        except ImportError:
            backend = "torch"

    try:
        if backend == "wyoming":
            stats = synthesize_wyoming(
                text, voice, args.speed, output,
                default_voice=DEFAULT_VOICES[LANGS[args.lang]],
                device=args.device, model_dir=args.model_dir,
                data_dir=args.data_dir, port=args.wyoming_port,
            )
        else:
            stats = synthesize(text, args.lang, voice, args.speed, output, args.threads)
    except KeyboardInterrupt:
        print("\ninterrompido. áudio parcial mantido em:", output, file=sys.stderr)
        return 130
    except Exception as e:
        print(f"erro na síntese ({e.__class__.__name__}): {e}", file=sys.stderr)
        return 1

    print(
        f"✔ salva em {stats['output']} | "
        f"backend={stats.get('backend', 'torch')} | "
        f"{stats['audio_seconds']:.1f}s de áudio | "
        f"{stats['chunks']} chunks | "
        f"{stats['elapsed']:.1f}s de processamento | "
        f"RTF {stats['rtf']:.2f}"
    )

    if args.play:
        if play(output):
            print("▶ tocando… (Ctrl+C para parar)")
        else:
            print("aviso: nenhum player encontrado (ffplay/mpv/aplay); abra o arquivo manualmente", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
