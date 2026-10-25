"""Narration for the 5-minute video, one entry per slide. Text is written as it should be spoken
(chemistries spelled out letter by letter, units in words) for the Windows text-to-speech voice."""

SLIDES = [
    ("title",
     "A state-of-health-aware framework for synthetic inertia delivery in solar and wind integrated grids. "
     "Course project for C M six seventeen, I I T Bombay."),

    ("problem",
     "When a power plant suddenly trips, grid frequency starts to fall. Heavy spinning generators normally slow "
     "that fall. This is called inertia. Solar panels and wind turbines connect through inverters, and have no "
     "inertia. Grid-forming batteries can provide synthetic inertia instead, but most studies assume the battery "
     "always delivers its full rated power. Real batteries age. Their resistance rises, and the power they can "
     "deliver falls."),

    ("system",
     "So we ask: how does battery ageing limit synthetic inertia? We study a system powered only by solar, or only "
     "by wind. A one hundred megawatt plant is paired with a twenty-five megawatt battery, which is the only "
     "source of inertia. We use real hourly solar and wind data for twenty twenty-five at Kutch, Gujarat, and "
     "compare three chemistries: L F P, N M C, and N C A."),

    ("chain",
     "Our framework is a chain of four steps. First, the real solar or wind data sets how hard the battery works. "
     "Second, Model one, a random forest trained on Sandia cycling data, turns this workload into state of health, "
     "year by year. Third, Model two, learned from Severson's cells, turns state of health into resistance, and so "
     "into the maximum power the battery can still deliver. Fourth, in every hour of the year, we apply a sudden "
     "ten megawatt loss, and check whether the aged battery can hold the frequency."),

    ("models",
     "Model one uses chemistry, temperature, depth of discharge, discharge rate, and cycle count. On cells it "
     "never saw, it reaches an R squared of zero point eight seven. For Model two, the common straight-line "
     "formula reaches zero point seven four. Our quadratic curve reaches zero point seven eight, almost as good "
     "as a random forest. At a state of health of zero point eight two, the battery keeps only eighty-six "
     "percent of its fresh power."),

    ("fig_a",
     "Here is what ageing does over fifteen years. L F P ages slowly, and still delivers twenty-four point three "
     "megawatts at year ten. N M C and N C A drop to about twenty-one and a half. The dotted line is what the "
     "battery needs: twelve and a half megawatts for normal work, plus ten for the trip. N C A falls below it in "
     "year three, and N M C in year five."),

    ("fig_b",
     "What happens to frequency? We simulated the first twenty seconds after a trip. Within its power limit, an "
     "aged battery behaves exactly like a new one. Frequency dips to forty-nine point eight hertz, and recovers. "
     "But once the aged battery hits its limit, it loses control. Frequency crosses forty-nine point four hertz, "
     "where load shedding begins, in under one second, and does not recover."),

    ("fig_c",
     "How often does this happen? L F P never fails in fifteen years. N M C and N C A fail in about half a "
     "percent of hours with solar, and over one percent with wind. Wind is about twice as exposed, because it "
     "produces in twice as many hours. Think of the battery as a wallet with an emergency fund. As it ages, the "
     "wallet shrinks, and if the operator keeps spending as before, ageing eats the emergency fund. Our "
     "state-of-health-aware operation cuts everyday use just enough to protect the reserve. This removes every "
     "failure, at almost no cost."),

    ("conclusion",
     "To conclude. An ageing battery gives full synthetic inertia, until its power ceiling falls below what it "
     "is asked to carry. L F P needs about five percent extra power rating, N M C and N C A fifteen to "
     "seventeen percent. Tracking state of health and protecting the reserve solves the problem at almost no "
     "cost. Next steps are resistance data for each chemistry, grid-scale cells, and real-time control. "
     "Thank you."),

    ("credits", ""),                       # shown silently
]
