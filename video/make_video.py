"""Join slides and voice-over into one MP4 (1920x1080, 30 fps). Each slide stays up exactly as long as its
narration plus a short pause; nothing is sped up. Prints the measured length of the final file.
Usage: python make_video.py [slide_folder] [output.mp4]   (defaults: slides, CM617_video.mp4)"""
import re
import subprocess
import sys
import wave
from pathlib import Path

import imageio_ffmpeg

from script import SLIDES

HERE = Path(__file__).resolve().parent
AUD = HERE / "audio"
SL = HERE / (sys.argv[1] if len(sys.argv) > 1 else "slides")
OUT = HERE / (sys.argv[2] if len(sys.argv) > 2 else "CM617_video.mp4")
FF = imageio_ffmpeg.get_ffmpeg_exe()
LEAD, TAIL, END = 0.4, 0.5, 1.0          # s of silence before / after each slide's speech, extra at the very end

frames, durs = [], []
params = None
for i, (name, _) in enumerate(SLIDES):
    with wave.open(str(AUD / f"{name}.wav")) as w:
        params = params or w.getparams()
        data = w.readframes(w.getnframes())
    rate, width = params.framerate, params.sampwidth * params.nchannels
    tail = TAIL + (END if i == len(SLIDES) - 1 else 0.0)
    pad0, pad1 = b"\x00" * width * int(LEAD * rate), b"\x00" * width * int(tail * rate)
    frames.append(pad0 + data + pad1)
    durs.append((len(pad0) + len(data) + len(pad1)) / width / rate)

with wave.open(str(HERE / "narration.wav"), "wb") as w:
    w.setparams(params)
    w.writeframes(b"".join(frames))

lst = HERE / "slides.txt"
lines = []
for (name, _), d in zip(SLIDES, durs):
    lines += [f"file '{(SL / f'{name}.png').as_posix()}'", f"duration {d:.3f}"]
lines.append(f"file '{(SL / f'{SLIDES[-1][0]}.png').as_posix()}'")
lst.write_text("\n".join(lines) + "\n")

subprocess.run([FF, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
                "-i", str(HERE / "narration.wav"), "-vf", "fps=30,format=yuv420p", "-c:v", "libx264",
                "-preset", "medium", "-crf", "20", "-c:a", "aac", "-b:a", "160k", "-shortest", str(OUT)], check=True)

info = subprocess.run([FF, "-i", str(OUT)], capture_output=True, text=True).stderr
h, m, s = re.search(r"Duration: (\d+):(\d+):([\d.]+)", info).groups()
total = int(h) * 3600 + int(m) * 60 + float(s)
print("slide times (s):", [round(d, 1) for d in durs])
print(f"planned {sum(durs):.1f} s; final file {OUT.name}: {total:.2f} s = {int(total // 60)}:{total % 60:05.2f}, "
      f"{OUT.stat().st_size / 1e6:.1f} MB")
