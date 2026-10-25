"""Voice-over: one WAV per slide (44.1 kHz mono 16-bit), at the voice's normal rate (no speed-up).
Engines: Microsoft neural voice via edge-tts (default, needs internet), or the built-in Windows voice.
An entry with empty text gets SILENT_S seconds of silence (credits slide).
Usage: python make_audio.py [edge|windows] [voice]
       defaults: edge en-US-GuyNeural; windows Zira"""
import asyncio
import subprocess
import sys
import wave
from pathlib import Path

import imageio_ffmpeg

from script import SLIDES

HERE = Path(__file__).resolve().parent
AUD = HERE / "audio"
AUD.mkdir(exist_ok=True)
ENGINE = sys.argv[1] if len(sys.argv) > 1 else "edge"
VOICE = sys.argv[2] if len(sys.argv) > 2 else ("en-US-GuyNeural" if ENGINE == "edge" else "Zira")
RATE = 44100
SILENT_S = 5.0
FF = imageio_ffmpeg.get_ffmpeg_exe()

PS = r"""
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$s.SelectVoice('Microsoft {voice} Desktop')
$s.Rate = 0
$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(44100, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
$s.SetOutputToWaveFile('{wav}', $fmt)
$s.Speak([IO.File]::ReadAllText('{txt}'))
$s.Dispose()
"""


def edge(text, wav):
    import edge_tts
    mp3 = wav.with_suffix(".mp3")
    asyncio.run(edge_tts.Communicate(text, VOICE).save(str(mp3)))
    subprocess.run([FF, "-y", "-loglevel", "error", "-i", str(mp3), "-ar", str(RATE), "-ac", "1",
                    "-sample_fmt", "s16", str(wav)], check=True)
    mp3.unlink()


def windows(text, wav):
    txt = wav.with_suffix(".txt")
    txt.write_text(text, encoding="utf-8")
    subprocess.run(["powershell", "-NoProfile", "-Command", PS.format(voice=VOICE, wav=wav, txt=txt)], check=True)


def silence(wav, seconds):
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(b"\x00\x00" * int(seconds * RATE))


total = 0.0
print(f"engine {ENGINE}, voice {VOICE}")
for name, text in SLIDES:
    wav = AUD / f"{name}.wav"
    if not text:
        silence(wav, SILENT_S)
    else:
        (edge if ENGINE == "edge" else windows)(text, wav)
    with wave.open(str(wav)) as w:
        dur = w.getnframes() / w.getframerate()
    total += dur
    print(f"{name:12s} {len(text.split()):4d} words  {dur:6.1f} s")
print(f"total {total:.1f} s ({total / 60:.2f} min), {sum(len(t.split()) for _, t in SLIDES)} words")
