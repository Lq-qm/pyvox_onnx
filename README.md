# pyvox 🎙️

> CLI em Python que narra arquivos de texto (`.txt`) usando o motor **Kokoro TTS** ([Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)).

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/)
[![Kokoro TTS](https://img.shields.io/badge/motor-Kokoro--82M-8A2BE2)](https://huggingface.co/hexgrad/Kokoro-82M)
[![Device](https://img.shields.io/badge/device-CPU-orange)](https://pytorch.org/)

## ✨ Funcionalidades

- 📖 Narra qualquer arquivo `.txt` (ou texto direto via `--text`)
- 🗣️ Motor **Kokoro-82M** com vozes em 9 idiomas (pt-br, en-us, en-gb, es, fr-fr, hi, it, ja, zh)
- 🎤 Narrador masculino / narradora feminina com `--narrador` (ou voz exata com `--voice`)
- 🧩 **Backends de inferência** trocáveis: `torch` (in-process), `onnx` (`kokoro-onnx`/onnxruntime, o mais rápido em CPU) e `wyoming` (servidor `wyoming-kokoro-torch`, streaming por protocolo Wyoming)
- 📡 Áudio em **streaming**: chunks de ~40 ms chegam enquanto o servidor sintetiza — progresso em tempo real
- 📥 Vozes faltantes são **baixadas automaticamente** para `./data/kokoro/`
- 📊 Métricas ao final: tempo de áudio, nº de chunks, tempo de processamento e RTF (razão tempo real)
- 🔊 Saída em **WAV** (padrão) ou **MP3** com `-f mp3` (conversão WAV→MP3 via `ffmpeg-python`)
- ▶️ Reprodução do resultado com `--play` (ffplay/mpv/aplay)
- 🎚️ Velocidade ajustável, controle de threads e `--max-chars` para testes rápidos
- 🛡️ Erros limpos: voz inexistente, servidor com falha ou porta indisponível retornam mensagem clara (sem processos órfãos)

## 📋 Requisitos

| Item | Versão |
|---|---|
| Python | 3.11+ (o `wyoming-kokoro-torch` exige 3.11+) |
| gerenciador de pacotes | `uv` (ou `pip`) |
| disco | ~1 GB (venv) + ~330 MB (modelo torch, baixado na 1ª execução) + ~340 MB (modelo ONNX + vozes, baixados na 1ª execução do backend `onnx`) |
| opcional (para `--play`) | `ffplay`, `mpv` ou `aplay` no `PATH` |
| opcional (para `-f mp3`) | binário `ffmpeg` no `PATH` (ex.: `sudo apt install ffmpeg`) |

## 🚀 Instalação

```bash
# 1) Crie o ambiente virtual
uv venv .venv

# 2) PyTorch SOMENTE CPU (use o índice CPU para não baixar a build CUDA)
uv pip install -i https://download.pytorch.org/whl/cpu torch

# 3) Dependências do projeto
uv pip install -r requirements.txt
```

## 📖 Uso

```bash
# Narrar um arquivo (padrão: pt-br, narradora pf_dora) → gera <arquivo>.wav
.venv/bin/python pyvox.py meu_texto.txt -o saida.wav

# Narrador masculino (pt-br: pm_alex)
.venv/bin/python pyvox.py meu_texto.txt --narrador masculino

# Narradora feminina + velocidade + tocar o resultado
.venv/bin/python pyvox.py meu_texto.txt --narrador feminino --speed 1.2 --play

# Escolher uma voz exata
.venv/bin/python pyvox.py meu_texto.txt --voice pm_santa

# Listar as vozes disponíveis para um idioma
.venv/bin/python pyvox.py --list-voices --lang pt-br

# Escolher o backend de inferência (auto | onnx | wyoming | torch)
.venv/bin/python pyvox.py meu_texto.txt --backend onnx --threads 8
.venv/bin/python pyvox.py meu_texto.txt --backend wyoming
.venv/bin/python pyvox.py meu_texto.txt --backend torch --threads 8

# Narra texto direto, sem arquivo
.venv/bin/python pyvox.py --text "Olá, mundo."

# Teste rápido: limita a síntese aos primeiros N caracteres
.venv/bin/python pyvox.py meu_texto.txt --max-chars 500

# Saída em MP3 (padrão é WAV; a extensão .mp3 em -o também ativa o MP3)
.venv/bin/python pyvox.py meu_texto.txt -f mp3
.venv/bin/python pyvox.py meu_texto.txt -o narracao.mp3 --play
```

### Opções

| Opção | Descrição |
|---|---|
| `file` | arquivo `.txt` a narrar (opcional com `--text`) |
| `--text TEXT` | narra texto direto, sem arquivo |
| `-l, --lang` | idioma: `pt-br` (padrão), `en-us`, `en-gb`, `es`, `fr-fr`, `hi`, `it`, `ja`, `zh` |
| `-g, --narrador` | `m`/`masculino` (narrador) ou `f`/`feminino` (narradora) — pt-br: m=`pm_alex`, f=`pf_dora` |
| `-v, --voice` | voz exata a usar (ex.: `pf_dora`, `pm_alex` — veja `--list-voices`) |
| `-s, --speed` | velocidade da fala (`0.5` = metade, `1.5` = 50% mais rápido; padrão `1.0`) |
| `-o, --output` | arquivo de áudio de saída (padrão: `<arquivo>.<formato>` ou `narração.<formato>`) |
| `-f, --format` | formato de saída: `wav` (padrão) ou `mp3` (conversão WAV→MP3 via `ffmpeg-python`; extensão `.mp3` em `-o` também ativa) |
| `--backend` | `auto` (padrão), `onnx`, `torch` ou `wyoming` — veja [Backends](#-backends-de-inferência) |
| `--device` | device do backend `wyoming`: `cpu` (padrão), `cuda`, `mps` |
| `--model-dir` | backend `wyoming`: snapshot do Kokoro-82M (padrão: cache do Hugging Face); backend `onnx`: arquivo `.onnx` ou diretório que o contenha (padrão: baixa `kokoro-v1.0.onnx` para `./data/kokoro/`) |
| `--data-dir` | diretório local de vozes/modelo (padrão: `./data/kokoro`) — backends `wyoming` e `onnx` |
| `--wyoming-port` | porta TCP do servidor Wyoming (padrão: porta livre automática) |
| `--threads N` | threads de CPU (padrão: todos os núcleos) — backends `torch` e `onnx` |
| `--max-chars N` | limita a síntese aos N primeiros caracteres (útil para testes) |
| `--play` | toca o áudio ao final |
| `--list-voices` | lista as vozes do idioma e sai |

### Exemplo de saída

```bash
$ .venv/bin/python pyvox.py meu_texto.txt --backend wyoming --narrador feminino
subindo servidor Wyoming (voz=pf_dora, device=cpu)…
servidor no ar em 2.8s (tcp://127.0.0.1:41233)
[wyoming]      6.5s de áudio |      3.5s | 153 chunks
✔ salva em meu_texto.wav | backend=wyoming | 6.5s de áudio | 153 chunks | 3.5s de processamento | RTF 0.54

$ .venv/bin/python pyvox.py meu_texto.txt --backend onnx
preparando backend ONNX (voz=pf_dora, pt-br, cpu)…
modelo ONNX pronto em 0.9s
[onnx]      6.1s de áudio |      1.5s | 1 chunks
✔ salva em meu_texto.wav | backend=onnx | 6.1s de áudio | 1 chunks | 1.5s de processamento | RTF 0.24

$ .venv/bin/python pyvox.py meu_texto.txt --backend torch
modelo pronto em 1.3s
[   1]      6.1s de áudio |      1.8s cpu | “A família vivia numa cidade pequena…”
✔ salva em meu_texto.wav | backend=torch | 6.1s de áudio | 1 chunks | 1.8s de processamento | RTF 0.30

$ .venv/bin/python pyvox.py meu_texto.txt -f mp3
preparando backend ONNX (voz=pf_dora, pt-br, cpu)…
modelo ONNX pronto em 0.9s
[onnx]      6.1s de áudio |      1.5s | 1 chunks
✔ salva em meu_texto.mp3 | backend=onnx | formato=mp3 | 6.1s de áudio | 1 chunks | 1.5s de processamento | RTF 0.24
```

## 🧩 Backends de inferência

O `pyvox` tem três backends trocáveis (mesma interface de CLI, mesmo modelo Kokoro-82M):

| Backend | Como funciona | Prós | Uso |
|---|---|---|---|
| `onnx` | síntese in-process via `kokoro-onnx`/onnxruntime (`CPUExecutionProvider`) | **o mais rápido em CPU**, sem subprocessos | `--backend onnx` |
| `torch` | síntese in-process via `kokoro`/PyTorch | simples, sem subprocessos | `--backend torch` |
| `wyoming` | sobe o servidor `wyoming-kokoro-torch` e consome o áudio **em streaming** (protocolo Wyoming, TCP local) | streaming, isolamento do processo, `--device` flexível | `--backend wyoming` |

`--backend auto` (padrão) usa o mais rápido disponível, na ordem **`onnx` → `wyoming` → `torch`**.

### Backend ONNX — detalhes

- **100% CPU**: a wheel `onnxruntime` instalada via `kokoro-onnx` é a variante CPU (a GPU exigiria o pacote separado `onnxruntime-gpu`, que **não** é usado aqui).
- **1ª execução** baixa para `./data/kokoro/` (ou `--data-dir`): `kokoro-v1.0.onnx` (~310 MB) e `voices-v1.0.bin` (~26 MB, 54 vozes) — com progresso na linha de comando.
- `--model-dir` aponta para um `.onnx` alternativo (ex.: a variante quantizada `kokoro-v1.0.int8.onnx`, ~108 MB, do [release](https://github.com/thewh1teagle/kokoro-onnx/releases) do `kokoro-onnx`); o restante (vozes) continua em `--data-dir`.
- `--threads N` controla as threads intra-op do onnxruntime (padrão: todos os núcleos).
- Vozes: as mesmas do Kokoro-82M (`pf_dora`, `pm_alex`, `pm_santa`, …) — o arquivo `.bin` já traz as 54 vozes em 9 idiomas; voz inexistente vira erro claro (`OnnxVoiceError`).
- Nota: o `kokoro-onnx` 0.4.7 envia `speed` como `int32` para exports com entrada `input_ids`, mas o modelo oficial espera `float` — por isso o backend monta os inputs a partir dos metadados da própria sessão (`onnx_backend.py`), cobrindo as duas gerações de export.

### Backend Wyoming — detalhes

- O servidor é lançado como subprocesso numa **porta TCP local automática** e encerrado ao final (`--wyoming-port` fixa a porta; sem processos órfãos em caso de erro).
- Modelo: o snapshot do cache do Hugging Face é usado como `--data-dir` do servidor (`--model-dir` sobrescreve).
- Vozes: faltantes são **baixadas automaticamente** para `./data/kokoro/` (`--data-dir` sobrescreve) — ex.: `pm_santa.pt`.
- `--device` aceita `cpu`, `cuda` ou `mps` (depende da build do PyTorch instalada; o setup padrão deste repositório usa PyTorch CPU).
- Voz por requisição: o servidor sobe com uma voz padrão válida e cada pedido usa a voz escolhida (`--narrador`/`--voice`).
- Erros do servidor (ex.: voz inexistente) voltam como exceção clara: `WyomingSynthesisError: ModelNotFoundError: pf_inexistente`.
- Para usar o servidor de forma independente (fora do pyvox):

  ```bash
  .venv/bin/wyoming-kokoro-torch \
    --voice pf_dora --uri tcp://127.0.0.1:10200 \
    --data-dir ~/.cache/huggingface/hub/models--hexgrad--Kokoro-82M/snapshots/<hash> \
    --data-dir ./data/kokoro --download-dir ./data/kokoro \
    --device cpu --speed 1.0
  ```

## ⚙️ Como funciona

```
.txt ──► backend onnx:    G2P (phonemizer-fork/espeak-ng) → Kokoro-82M (ONNX, onnxruntime CPU) → WAV
       ├─► backend torch: G2P (misaki) → Kokoro-82M (PyTorch, CPU) → WAV
       └─► backend wyoming: servidor wyoming-kokoro-torch (protocolo Wyoming, TCP local)
                              └─► sentenças → Kokoro-82M → chunks PCM (24 kHz) em streaming → WAV
```

1. O texto é dividido em segmentos (linhas, com chunking por sentenças/lotes de fonemas).
2. Cada segmento é convertido em fonemas (G2P) e sintetizado na CPU.
3. O áudio (24 kHz, 16-bit PCM) é gravado no `.wav`, com progresso na linha de comando e RTF ao final.

## 🎤 Narradores por idioma

| Idioma | Narrador (m) | Narradora (f) |
|---|---|---|
| pt-br | `pm_alex` | `pf_dora` |
| en-us | `am_michael` | `af_heart` |
| en-gb | `bm_fable` | `bf_emma` |
| es | `em_alex` | `ef_dora` |
| fr-fr | — (não disponível) | `ff_siwis` |
| hi | `hm_omega` | `hf_alpha` |
| it | `im_nicola` | `if_sara` |
| ja | `jm_kumo` | `jf_alpha` |
| zh | `zm_yunjian` | `zf_xiaobei` |

Outras vozes por idioma existem — veja `--list-voices` e use `--voice`.

## 📁 Estrutura do projeto

| Arquivo | Descrição |
|---|---|
| `pyvox.py` | CLI principal (argparse, backends `onnx`/`torch`/`wyoming`, progresso, reprodução) |
| `onnx_backend.py` | módulo do backend ONNX (download de modelo/vozes, sessão onnxruntime, síntese em lotes) |
| `wyoming_backend.py` | módulo do backend Wyoming (subprocesso do servidor, cliente do protocolo, ciclo de vida) |
| `requirements.txt` | dependências (com pins validados) |
| `data/kokoro/` | vozes baixadas/instaladas pelos backends (gerado em runtime; no `onnx`, também guarda `kokoro-v1.0.onnx` + `voices-v1.0.bin`) |

## 📝 Notas

- A **1ª execução** baixa o modelo torch (~330 MB) do Hugging Face (cache em `~/.cache/huggingface`); vozes extras são baixadas sob demanda para `./data/kokoro/`. O backend `onnx` baixa, uma única vez, `kokoro-v1.0.onnx` + `voices-v1.0.bin` (~340 MB) para `./data/kokoro/`.
- O backend `onnx` é **somente CPU** (por design): a wheel padrão do onnxruntime é CPU; para GPU seria preciso instalar `onnxruntime-gpu` (extra `kokoro-onnx[gpu]`) — fora do escopo deste projeto.
- Velocidade medida (mesma máquina, 6 núcleos, ~47 s de áudio): `onnx` RTF ~0.22 < `torch` RTF ~0.24 < `wyoming` RTF ~0.68 (inclui subida do servidor); os números variam por texto/hardware.
- A duração do áudio pode variar levemente entre backends (G2P diferente: `phonemizer-fork` no ONNX vs. `misaki` no torch; o servidor Wyoming também aplica normalização de volume e pontuação automática).
- Textos longos em CPU podem demorar — use `--max-chars` para testar; `Ctrl+C` interrompe mantendo o áudio já gerado.
- `-f mp3`: todos os backends gravam um WAV intermediário, que é convertido para MP3 (libmp3lame, 192 kbps, 24 kHz mono) via `ffmpeg-python` e removido ao final; `Ctrl+C` durante a conversão mantém o WAV. Exige o binário `ffmpeg` no PATH.
- GPU: use `--backend wyoming --device cuda` (ou `mps`) quando houver uma build PyTorch compatível instalada; o setup padrão deste repositório usa PyTorch CPU (e o backend `onnx` roda sempre em CPU).

