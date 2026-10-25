"""Slides for the video, 1920x1080 PNG, one per narration entry in script.py."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent
REP = HERE.parent / "report"
SL = HERE / "slides"
SL.mkdir(exist_ok=True)

NAVY, ACC, GREY, LIGHT = "#1f3b5c", "#e07b28", "#555555", "#eef2f7"
CHEM = {"LFP": "tab:blue", "NMC": "tab:orange", "NCA": "tab:green"}


def new(title=None, n=None):
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1920)
    ax.set_ylim(1080, 0)
    ax.axis("off")
    if title:
        ax.add_patch(plt.Rectangle((0, 0), 1920, 130, color=NAVY))
        ax.text(70, 68, title, color="white", fontsize=40, weight="bold", va="center")
    if n:
        ax.text(1880, 1050, f"{n} / 10", color=GREY, fontsize=16, ha="right")
        ax.text(40, 1050, "SoH-aware synthetic inertia  |  CM617, IIT Bombay", color=GREY, fontsize=16)
    return fig, ax


def bullets(ax, items, x=90, y=210, dy=95, size=31):
    for i, t in enumerate(items):
        ax.text(x, y + i * dy, "•", fontsize=size, color=ACC, va="center")
        ax.text(x + 45, y + i * dy, t, fontsize=size, color="#222222", va="center")


def box(ax, x, y, w, h, text, fc=LIGHT, ec=NAVY, size=24, color="#222222", weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=6,rounding_size=18", fc=fc, ec=ec, lw=2.5))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=color, weight=weight,
            linespacing=1.4)


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=35, lw=3, color=ACC))


def image(ax, path, x, y, w):
    img = mpimg.imread(path)
    h = w * img.shape[0] / img.shape[1]
    ax.imshow(img, extent=(x, x + w, y + h, y), zorder=2)
    return h


def save(fig, name):
    fig.savefig(SL / f"{name}.png", dpi=100)
    plt.close(fig)


# 1 title
fig, ax = new()
ax.add_patch(plt.Rectangle((0, 0), 1920, 1080, color=NAVY))
ax.text(960, 400, "A State-of-Health-Aware Framework for\nSynthetic Inertia Delivery in\nSolar and Wind-Integrated Grids",
        color="white", fontsize=52, weight="bold", ha="center", va="center", linespacing=1.35)
ax.plot([660, 1260], [610, 610], color=ACC, lw=5)
ax.text(960, 690, "Akshat Sharma, Rudraksh Sharma, Krishna Gahlod,\nSuryangshu Das, Sai Charan", color="white",
        fontsize=32, ha="center", va="top", linespacing=1.5)
ax.text(960, 910, "CM617, IIT Bombay", color="#c8d3e0", fontsize=28, ha="center")
save(fig, "title")

# 2 problem
fig, ax = new("The problem: inverters have no inertia, batteries age", 2)
bullets(ax, ["A plant trips  →  grid frequency starts to fall",
             "Spinning generators slow the fall: this is inertia",
             "Solar and wind connect through inverters: no inertia",
             "Grid-forming batteries can provide synthetic inertia",
             "Most studies assume the battery always gives full rated power",
             "Real batteries age: resistance rises, deliverable power falls"], y=240, dy=110)
box(ax, 90, 890, 1240, 100, "Question: how does ageing limit synthetic inertia?", fc="#fdf0e6", ec=ACC, size=28,
    weight="bold")
save(fig, "problem")

# 3 system
fig, ax = new("Our system: the battery is the only inertia", 3)
box(ax, 200, 190, 660, 170, "100 MW plant: solar OR wind\n(no inertia)", size=28)
box(ax, 200, 420, 660, 170, "25 MW / 50 MWh grid-forming\nbattery = only inertia", fc="#fdf0e6", ec=ACC, size=28)
box(ax, 1200, 310, 420, 160, "Load", size=34)
arrow(ax, 870, 275, 1190, 370)
arrow(ax, 870, 505, 1190, 410)
bullets(ax, ["Real hourly 2025 solar and wind output, Kutch, Gujarat (Renewables.ninja)",
             "Three chemistries compared: LFP, NMC, NCA",
             "Battery keeps a 12.5 MW reserve for sudden trips"], y=700, dy=95, size=29)
save(fig, "system")

# 4 chain
fig, ax = new("The framework: a chain of four steps", 4)
steps = [("1  Real data", "Kutch solar / wind\n→ how hard the\nbattery works"),
         ("2  Model 1", "random forest\n(Sandia data)\n→ SoH per year"),
         ("3  Model 2", "R(SoH) curve\n(Severson data)\n→ resistance\n→ max power"),
         ("4  Events", "10 MW trip\nevery hour\n→ frequency\nheld or not?")]
for i, (h, t) in enumerate(steps):
    x = 70 + i * 460
    ax.add_patch(FancyBboxPatch((x, 260), 380, 560, boxstyle="round,pad=6,rounding_size=20",
                                fc=LIGHT if i % 2 == 0 else "#fdf0e6", ec=NAVY if i % 2 == 0 else ACC, lw=3))
    ax.text(x + 190, 330, h, ha="center", fontsize=32, weight="bold", color=NAVY)
    ax.text(x + 190, 560, t, ha="center", va="center", fontsize=28, linespacing=1.5)
    if i < 3:
        arrow(ax, x + 392, 540, x + 448, 540)
ax.text(960, 920, "SoH-aware operation uses Models 1 and 2 to protect the battery's reserve", ha="center",
        fontsize=28, color=ACC, weight="bold")
save(fig, "chain")

# 5 models
fig, ax = new("How good are the models?", 5)
ax.text(90, 220, "Model 1: state of health", fontsize=32, weight="bold", color=NAVY)
ax.text(90, 285, "Sandia, 85 cells; inputs: chemistry, temperature,", fontsize=23, color=GREY)
ax.text(90, 330, "depth of discharge, discharge rate, cycles", fontsize=23, color=GREY)
ax2 = fig.add_axes([0.17, 0.13, 0.29, 0.5])
ax2.barh(["straight line\n(baseline)", "random forest\nunseen cells", "random forest\nunseen conditions"],
         [0.55, 0.87, 0.81], color=["#bbbbbb", NAVY, NAVY])
for i, v in enumerate([0.55, 0.87, 0.81]):
    ax2.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=22)
ax2.set_xlim(0, 1.05)
ax2.set_xlabel("R²", fontsize=22)
ax2.tick_params(labelsize=20)
ax.text(1060, 220, "Model 2: resistance from SoH", fontsize=32, weight="bold", color=NAVY)
ax.text(1060, 285, "Severson, 124 LFP cells;", fontsize=23, color=GREY)
ax.text(1060, 330, "tested on 43 cells it never saw", fontsize=23, color=GREY)
ax3 = fig.add_axes([0.64, 0.36, 0.31, 0.27])
ax3.barh(["linear formula", "our quadratic", "random forest"], [0.74, 0.78, 0.79],
         color=["#bbbbbb", ACC, "#bbbbbb"])
for i, v in enumerate([0.74, 0.78, 0.79]):
    ax3.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=22)
ax3.set_xlim(0, 1.05)
ax3.set_xlabel("R²", fontsize=22)
ax3.tick_params(labelsize=20)
box(ax, 1060, 790, 780, 150, "R/R₀ = 1 + 0.393x + 3.02x²,  x = 1 − SoH\nat SoH 0.82: 86% of fresh power",
    fc="#fdf0e6", ec=ACC, size=26)
save(fig, "models")

# 6 figure A
fig, ax = new("Ageing cuts power, and chemistry decides how much", 6)
h = image(ax, REP / "fig_A.png", 60, 180, 1250)
ax.text(1360, 260, "Year 10 power ceiling", fontsize=30, weight="bold", color=NAVY)
for i, (c, v) in enumerate([("LFP", "24.3 MW"), ("NMC", "21.8 MW"), ("NCA", "21.5 MW")]):
    ax.text(1380, 360 + i * 80, c, fontsize=30, weight="bold", color=CHEM[c])
    ax.text(1540, 360 + i * 80, v, fontsize=30)
ax.text(1360, 640, "Needed: 12.5 + 10 = 22.5 MW", fontsize=23, color="#222222")
ax.text(1360, 720, "NCA below from year 3\nNMC below from year 5\nLFP never", fontsize=26, color=ACC,
        weight="bold", va="top", linespacing=1.5)
save(fig, "fig_a")

# 7 figure B
fig, ax = new("Within its limit, an aged battery acts like new", 7)
image(ax, REP / "fig_B.png", 60, 190, 1250)
box(ax, 1370, 260, 480, 230, "Within limit\ndips to 49.81 Hz\nand recovers", size=27)
box(ax, 1370, 560, 480, 260, "At its limit\ncrosses 49.4 Hz\nin < 1 s\nno recovery", fc="#fdf0e6", ec=ACC, size=27)
save(fig, "fig_b")

# 8 figure C
fig, ax = new("How often it fails, and the SoH-aware fix", 8)
image(ax, REP / "fig_C.png", 60, 170, 1150)
rows = [("", "solar", "wind"), ("LFP", "0%", "0%"), ("NMC", "0.45%", "1.14%"), ("NCA", "0.59%", "1.24%"),
        ("SoH-aware", "0%", "0%")]
ax.text(1280, 230, "Trips not covered, year 10", fontsize=28, weight="bold", color=NAVY)
for i, (a, b, c) in enumerate(rows):
    y = 310 + i * 70
    col = CHEM.get(a, ACC if a == "SoH-aware" else "#222222")
    ax.text(1290, y, a, fontsize=27, weight="bold", color=col)
    ax.text(1590, y, b, fontsize=27, ha="center", weight="bold" if i == 0 else "normal")
    ax.text(1770, y, c, fontsize=27, ha="center", weight="bold" if i == 0 else "normal")
box(ax, 1270, 700, 590, 230, "Battery = wallet with an\nemergency fund. SoH-aware:\nspend less as it shrinks,\nkeep the fund whole.",
    fc="#fdf0e6", ec=ACC, size=24)
ax.text(90, 1000, "Wind is about twice as exposed: it produces in twice as many hours", fontsize=26, color="#222222")
save(fig, "fig_c")

# 9 conclusion
fig, ax = new("Conclusion and future scope", 9)
bullets(ax, ["Aged battery gives full synthetic inertia until its power ceiling is too low",
             "Chemistry matters: extra power rating LFP +5%, NMC +15%, NCA +17%",
             "SoH-aware operation removes every failure at almost no cost"], y=230, dy=100, size=30)
ax.text(90, 600, "Future scope", fontsize=32, weight="bold", color=NAVY)
bullets(ax, ["Resistance data for each chemistry",
             "Grid-scale cells and longer tests",
             "Real-time SoH-aware control from battery management data"], y=690, dy=85, size=28)
ax.text(1830, 960, "Thank you", fontsize=40, weight="bold", color=ACC, ha="right")
save(fig, "conclusion")

# 10 credits (shown silently)
fig, ax = new("Team and contributions", 10)
team = [("Akshat Sharma", "Simulation"), ("Rudraksh Sharma", "Modelling (ML)"), ("Suryangshu Das", "Inferences"),
        ("Krishna Gahlod and Sai Charan", "Data Collection and Data Processing")]
for i, (who, role) in enumerate(team):
    y = 290 + i * 150
    ax.add_patch(FancyBboxPatch((120, y - 55), 1710, 110, boxstyle="round,pad=4,rounding_size=16",
                                fc=LIGHT if i % 2 == 0 else "#fdf0e6", ec="none"))
    ax.text(190, y, who, fontsize=31, weight="bold", color=NAVY, va="center")
    ax.text(1060, y, role, fontsize=28, color="#222222", va="center")
ax.text(960, 930, "CM617, IIT Bombay", fontsize=28, color=GREY, ha="center")
save(fig, "credits")
print("saved", sorted(p.name for p in SL.glob("*.png")))
