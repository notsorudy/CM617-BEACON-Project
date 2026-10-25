# How to use this in Gamma

1. In Gamma choose **Create new > Paste in text**.
2. Paste everything below the line "PASTE FROM HERE".
3. Settings: **Presentation**, **16:9**, text content **"Preserve this exact text"** (or "Keep as is"; not "Generate" or "Condense"), card breaks: **use the `---` separators**, so you get exactly 10 cards.
4. Theme: a clean, light, academic theme with navy and orange accents (it should match our figures).
5. After it generates, upload our three figures into cards 6, 7 and 8 where it says IMAGE. The files are in the project's `report` folder: `fig_A.png`, `fig_B.png`, `fig_C.png`. Delete any AI images or charts Gamma added there.
6. Check that every number is unchanged and there are still exactly 10 cards in this order.
7. Export: **PNG (one image per card)** if available, otherwise **PDF**. Send me the files.

PASTE FROM HERE

Instructions for Gamma: Make exactly 10 slides, one per section separated by ---. Keep all text and every number exactly as written; do not add, remove or reword content, and do not add extra slides. Use minimal decoration, no stock photos of people, and no AI-generated charts or numbers. Where a slide says IMAGE, leave a large empty image area for a figure I will upload. Style: clean academic, white background, navy (#1f3b5c) headings, orange (#e07b28) accents.

---

# A State-of-Health-Aware Framework for Synthetic Inertia Delivery in Solar and Wind-Integrated Grids

Akshat Sharma, Rudraksh Sharma, Krishna Gahlod, Suryangshu Das, Sai Charan

CM617, IIT Bombay

---

# The problem: inverters have no inertia, batteries age

- A plant trips, and grid frequency starts to fall
- Spinning generators slow the fall: this is inertia
- Solar and wind connect through inverters: no inertia
- Grid-forming batteries can provide synthetic inertia
- Most studies assume the battery always gives full rated power
- Real batteries age: resistance rises, deliverable power falls

**Question: how does ageing limit synthetic inertia?**

---

# Our system: the battery is the only inertia

- 100 MW plant: solar OR wind (no inertia)
- 25 MW / 50 MWh grid-forming battery: the only source of inertia
- Both supply the load

Details:
- Real hourly 2025 solar and wind output, Kutch, Gujarat (Renewables.ninja)
- Three chemistries compared: LFP, NMC, NCA
- Battery keeps a 12.5 MW reserve for sudden trips

---

# The framework: a chain of four steps

1. **Real data:** Kutch solar or wind output sets how hard the battery works
2. **Model 1:** random forest trained on Sandia cycling data gives state of health (SoH) per year
3. **Model 2:** R(SoH) curve from Severson data gives resistance, and so the maximum power
4. **Events:** a 10 MW trip in every hour of the year: is the frequency held?

SoH-aware operation uses Models 1 and 2 to protect the battery's reserve.

---

# How good are the models?

**Model 1: state of health** (Sandia, 85 cells; inputs: chemistry, temperature, depth of discharge, discharge rate, cycles)
- Random forest, unseen cells: R² = 0.87
- Random forest, unseen test conditions: R² = 0.81
- Straight-line baseline: R² = 0.55

**Model 2: resistance from SoH** (Severson, 124 LFP cells, tested on 43 unseen cells)
- Linear formula: R² = 0.74
- Our quadratic curve: R² = 0.78
- Random forest: R² = 0.79

R/R₀ = 1 + 0.393x + 3.02x², with x = 1 − SoH. At SoH 0.82 the battery keeps 86% of its fresh power.

---

# Ageing cuts power, and chemistry decides how much

IMAGE: fig_A.png (state of health and power ceiling over 15 years)

Year 10 power ceiling:
- LFP: 24.3 MW
- NMC: 21.8 MW
- NCA: 21.5 MW

Needed: 12.5 + 10 = 22.5 MW. NCA falls below from year 3, NMC from year 5, LFP never.

---

# Within its limit, an aged battery acts like new

IMAGE: fig_B.png (frequency in the first 10 seconds after a 10 MW trip)

- Within its power limit: frequency dips to 49.81 Hz and recovers
- At its limit: frequency crosses 49.4 Hz (load shedding) in under 1 second and does not recover

---

# How often it fails, and the SoH-aware fix

IMAGE: fig_C.png (share of trip hours not covered, by year)

Trips not covered at year 10 (solar / wind):
- LFP: 0% / 0%
- NMC: 0.45% / 1.14%
- NCA: 0.59% / 1.24%
- SoH-aware operation: 0% / 0%

Wind is about twice as exposed: it produces in twice as many hours.

The battery is like a wallet with an emergency fund. SoH-aware operation spends less as the wallet shrinks and keeps the fund whole.

---

# Conclusion and future scope

- An aged battery gives full synthetic inertia until its power ceiling is too low
- Chemistry matters: extra power rating needed is LFP +5%, NMC +15%, NCA +17%
- SoH-aware operation removes every failure at almost no cost

**Future scope**
- Resistance data for each chemistry
- Grid-scale cells and longer tests
- Real-time SoH-aware control from battery management data

**Thank you**

---

# Team and contributions

- **Akshat Sharma:** Simulation
- **Rudraksh Sharma:** Modelling (ML)
- **Suryangshu Das:** Inferences
- **Krishna Gahlod and Sai Charan:** Data Collection and Data Processing

CM617, IIT Bombay
