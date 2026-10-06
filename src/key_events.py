"""Reference dates for notable FFF-relevant events -- plotted as vertical markers in
notebooks/05_llm_scoring_analysis.ipynb so they're maintained in one place instead of
hardcoded in every plotting cell (same pattern as src/crisis_windows.py).

`type` is one of:
    "global_strike" -- worldwide, simultaneously-coordinated FFF strike days
    "strike"         -- national/regional/local FFF strikes and protests (Germany-focused
                        unless noted), including one-off single-city or single-issue actions
    "campaign"       -- non-strike campaigns: petitions, open letters, livestream/education
                        series, lobbying pushes
    "congress"       -- internal movement congresses (coordination, strategy, networking)
    "election"       -- state/local German elections (external, not organized by FFF)
    "eu_election"    -- European Parliament elections (external, not organized by FFF)
    "bt_election"    -- German federal (Bundestag) elections (external, not organized by FFF)
    "COP"            -- UN Climate Change Conferences (external, not organized by FFF)

`highlight` marks the curated subset for the static "hero" plots in
notebooks/05_llm_scoring_analysis.ipynb (src.plot_style.add_event_lines with the default
`styles` only draws events with a type it has a style for) -- True for every global_strike,
eu_election, and bt_election; False for everything else, including state/local "election"
entries. Recompute by hand if the curated criteria change; it's stored rather than derived
at plot time so the "what counts as highlighted" decision lives in one visible place.

Two entries are unverified rather than guessed at: "Sterki bleibt" (2022-02-11) -- kept
verbatim, likely a typo/unfamiliar campaign name, worth checking; and any event still
missing a `topic` description.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KeyEvent:
    date: str  # "YYYY-MM-DD"
    label: str
    topic: str  # one-sentence description of the event
    type: str
    highlight: bool  # curated subset for the static hero plots -- see module docstring


KEY_EVENTS: list[KeyEvent] = [

    # 2019
    KeyEvent(
        "2019-03-15",
        "Global Climate Strike",
        "First worldwide school strike for climate action, demanding immediate and effective climate protection.",
        "global_strike",
        True,
    ),
    KeyEvent(
        "2019-04-08",
        "FFFFordert Campaign",
        "Fridays for Future Germany publishes a list of concrete demands to political decision-makers, calling for immediate climate action.",
        "campaign",
        False,
    ),
    KeyEvent(
        "2019-05-24",
        "Global Climate Strike (ahead of EU election)",
        "Worldwide climate strike ahead of the European Parliament election, calling for stronger climate action.",
        "global_strike",
        True,
    ),
    KeyEvent(
        "2019-05-26",
        "EU Parliament Election",
        "Election to the European Parliament in Germany.",
        "eu_election",
        True,
    ),
    KeyEvent(
        "2019-06-21",
        "Central Strike, Aachen",
        "First major central strike gathering, held in Aachen with participants from 16 countries.",
        "strike",
        False,
    ),
    KeyEvent(
        "2019-07-31",
        "Summer Congress, Dortmund",
        "Summer congress to connect and coordinate Fridays for Future groups across Germany.",
        "congress",
        False,
    ),
    KeyEvent(
        "2019-08-05",
        "Smile for Future Summit, Lausanne",
        "Climate summit in Lausanne, bringing together 450 young activists from 37 countries.",
        "congress",
        False,
    ),
    KeyEvent(
        "2019-09-01",
        "State Elections, Brandenburg & Saxony",
        "State elections in Brandenburg and Saxony, with coal mining as a central campaign topic.",
        "election",
        False,
    ),
    KeyEvent(
        "2019-09-20",
        "Global Climate Strike (#AlleFürsKlima)",
        "Worldwide climate strike demanding ambitious climate action; around 1.4 million people took part in Germany alone.",
        "global_strike",
        True,
    ),
    KeyEvent(
        "2019-09-21",
        "Week4Climate Campaign",
        "Global week of climate action and events.",
        "campaign",
        False,
    ),
    KeyEvent(
        "2019-09-27",
        "Global Climate Strike",
        "International climate strike during the global climate week, demanding stronger climate policies.",
        "global_strike",
        True,
    ),
    KeyEvent(
        "2019-11-25",
        "Public Climate School",
        "Recurring educational campaign of public lectures and workshops on climate science and policy.",
        "campaign",
        False,
    ),
    KeyEvent(
        "2019-11-29",
        "Global Climate Strike (#NeustartKlima)",
        "Global climate strike shortly before COP25, demanding a fundamental change in German climate policy.",
        "global_strike",
        True,
    ),
    KeyEvent(
        "2019-12-02",
        "COP25",
        "UN Climate Change Conference, Madrid, Spain.",
        "COP",
        False,
    ),

    # 2020
    KeyEvent(
        "2020-01-02",
        "Nordkongress, Hamburg",
        "Congress of around 300 activists in Hamburg to plan and coordinate the upcoming year.",
        "congress",
        False,
    ),
    KeyEvent(
        "2020-02-21",
        "Election, Hamburg",
        "Bürgerschaft (state parliament) election in Hamburg.",
        "election",
        False,
    ),
    KeyEvent(
        "2020-03-13",
        "Municipal Election, Bavaria",
        "Local elections in Bavaria; the accompanying protest moved online (\"Netzstreik\") due to the COVID-19 pandemic.",
        "election",
        False,
    ),
    KeyEvent(
        "2020-04-24",
        "Global Climate Strike (#NetzstreikFürsKlima, #FightEveryCrisis)",
        "Large-scale digital global climate strike during the COVID-19 pandemic, keeping pressure on governments for climate action.",
        "global_strike",
        True,
    ),
    KeyEvent(
        "2020-05-01",
        "WirBildenZukunft Campaign",
        "Livestream series on climate, society, and crisis education.",
        "campaign",
        False,
    ),
    KeyEvent(
        "2020-05-25",
        "Public Climate School",
        "Recurring educational campaign of public lectures and workshops on climate science and policy.",
        "campaign",
        False,
    ),
    KeyEvent(
        "2020-06-02",
        "KlimaZielStattLobbyDeal Campaign",
        "Strikes for a fair and sustainable COVID-19 economic recovery package, demanding that state aid come with climate conditions.",
        "strike",
        False,
    ),
    KeyEvent(
        "2020-08-14",
        "Day of Action, Datteln IV",
        "Coordinated strikes in 35 German cities against the Datteln IV coal-fired power plant.",
        "strike",
        False,
    ),
    KeyEvent(
        "2020-09-13",
        "Municipal Elections, North Rhine-Westphalia",
        "Local elections in North Rhine-Westphalia.",
        "election",
        False,
    ),
    KeyEvent(
        "2020-09-25",
        "Global Climate Strike (#KeinGradWeiter)",
        "Sixth global climate strike, demanding climate justice and an end to delaying the energy transition.",
        "global_strike",
        True,
    ),
    KeyEvent(
        "2020-10-04",
        "Save Danni Campaign",
        "Protest to save the Dannenröder Forst from being cleared for a new highway.",
        "strike",
        False,
    ),
    KeyEvent(
        "2020-10-15",
        "1.5° Feasibility Study",
        "Fridays for Future commissions a study assessing the feasibility of the 1.5°C target under current German policy.",
        "campaign",
        False,
    ),
    KeyEvent(
        "2020-11-23",
        "Public Climate School",
        "Recurring educational campaign of public lectures and workshops on climate science and policy.",
        "campaign",
        False,
    ),
    KeyEvent(
        "2020-12-11",
        "FightFor1Point5 Campaign",
        "Action marking the fifth anniversary of the Paris Climate Agreement.",
        "strike",
        False,
    ),

    # 2021
    KeyEvent(
        "2021-01-22",
        "Campaign Against Gas Expansion",
        "Nationwide campaign actions across Germany opposing the expansion of fossil gas infrastructure.",
        "campaign",
        False,
    ),
    KeyEvent(
        "2021-03-14",
        "State Elections, Hesse, Rhineland-Palatinate & Baden-Württemberg",
        "State elections held in three federal states.",
        "election",
        False,
    ),
    KeyEvent(
        "2021-03-19",
        "Global Climate Strike (#AlleFür1Komma5, #NoMoreEmptyPromises)",
        "Global climate strike demanding policies compatible with the 1.5°C target and an end to empty climate promises.",
        "global_strike",
        True,
    ),
    KeyEvent(
        "2021-04-29",
        "Constitutional Court Climate Ruling",
        "The Federal Constitutional Court rules that parts of Germany's Climate Protection Act are unconstitutional for insufficiently limiting future emissions.",
        "campaign",
        False,
    ),
    KeyEvent(
        "2021-05-14",
        "Day of Action Against Coal",
        "Germany-wide day of action against fossil fuels.",
        "campaign",
        False,
    ),
    KeyEvent(
        "2021-05-17",
        "Public Climate School",
        "Recurring educational campaign of public lectures and workshops on climate science and policy.",
        "campaign",
        False,
    ),
    KeyEvent(
        "2021-08-05",
        "Summer Congress & Strikes, Eastern Germany",
        "Climate congress with workshops, discussions, music and networking around climate justice, combined with regional strikes.",
        "strike",
        False,
    ),
    KeyEvent(
        "2021-08-13",
        "Our Future Is Not For Sale",
        "Strike in Frankfurt am Main targeting the finance industry.",
        "strike",
        False,
    ),
    KeyEvent(
        "2021-08-27",
        "Central Strike Across Germany (#NichtWieNRW)",
        "Nationwide central strike directed against Armin Laschet's candidacy.",
        "strike",
        False,
    ),
    KeyEvent(
        "2021-09-24",
        "Global Climate Strike (#AlleFürsKlima)",
        "Global climate strike two days before the German federal election, calling for ambitious climate action.",
        "global_strike",
        True,
    ),
    KeyEvent(
        "2021-09-26",
        "German Federal Election",
        "Election to the German Bundestag.",
        "bt_election",
        True,
    ),
    KeyEvent(
        "2021-09-26",
        "State Election, Mecklenburg-Vorpommern",
        "State election held concurrently with the federal election.",
        "election",
        False,
    ),
    KeyEvent(
        "2021-10-01",
        "100-Day Demands",
        "Fridays for Future presents demands the incoming government coalition should achieve within its first 100 days.",
        "campaign",
        False,
    ),
    KeyEvent(
        "2021-10-22",
        "Central Strike, Berlin",
        "Central nationwide strike held in Berlin.",
        "strike",
        False,
    ),
    KeyEvent(
        "2021-10-31",
        "COP26",
        "UN Climate Change Conference, Glasgow, UK.",
        "COP",
        False,
    ),
    KeyEvent(
        "2021-11-22",
        "Public Climate School",
        "Recurring educational campaign of public lectures and workshops on climate science and policy.",
        "campaign",
        False,
    ),

    # 2022
    KeyEvent(
        "2022-02-11",
        "Sterki bleibt",
        "Protest against forest clearance for highway construction.",
        "strike",
        False,
    ),
    KeyEvent(
        "2022-03-03",
        "Stand with Ukraine",
        "Solidarity strike with Ukraine against Russia's invasion of the country.",
        "strike",
        False,
    ),
    KeyEvent(
        "2022-03-25",
        "Global Climate Strike (#ReichtHaltNicht)",
        "Global climate strike demanding peace and climate justice.",
        "global_strike",
        True,
    ),
    KeyEvent(
        "2022-05-16",
        "Public Climate School",
        "Recurring educational campaign of public lectures and workshops on climate science and policy.",
        "campaign",
        False,
    ),
    KeyEvent(
        "2022-05-21",
        "Central Strike, Cologne",
        "Central strike ahead of the state election in North Rhine-Westphalia.",
        "strike",
        False,
    ),
    KeyEvent(
        "2022-09-23",
        "Global Climate Strike (#PeopleNotProfit)",
        "Worldwide climate strike demanding that people and climate justice take priority over fossil-fuel profits.",
        "global_strike",
        True,
    ),
    KeyEvent(
        "2022-11-06",
        "COP27",
        "UN Climate Change Conference, Sharm El-Sheikh, Egypt.",
        "COP",
        False,
    ),
    KeyEvent(
        "2022-11-07",
        "Public Climate School",
        "Recurring educational campaign of public lectures and workshops on climate science and policy.",
        "campaign",
        False,
    ),

    # 2023
    KeyEvent(
        "2023-01-14",
        "Solidarity with Lützerath",
        "Large climate demonstration opposing the demolition of Lützerath and the expansion of the Garzweiler coal mine.",
        "strike",
        False,
    ),
    KeyEvent(
        "2023-03-03",
        "Global Climate Strike (#TomorrowIsTooLate)",
        "Global climate strike calling for immediate climate action; in Germany the protest was coordinated with a public-transport workers' strike.",
        "global_strike",
        True,
    ),
    KeyEvent(
        "2023-03-27",
        "Strike with ver.di",
        "Joint climate and public-sector strike action with the ver.di trade union.",
        "strike",
        False,
    ),
    KeyEvent(
        "2023-04-30",
        "Protest Against LNG Pipelines, Rügen",
        "Strike opposing liquefied natural gas (LNG) pipeline construction on the island of Rügen.",
        "strike",
        False,
    ),
    KeyEvent(
        "2023-05-08",
        "Public Climate School",
        "Recurring educational campaign of public lectures and workshops on climate science and policy.",
        "campaign",
        False,
    ),
    KeyEvent(
        "2023-05-11",
        "Protest, Heidelberg",
        "\"End Cement\" demonstration in Heidelberg, Germany.",
        "strike",
        False,
    ),
    KeyEvent(
        "2023-05-17",
        "Strategy Congress, Dresden",
        "Congress on movement strategy and organizational structures, held in Dresden.",
        "congress",
        False,
    ),
    KeyEvent(
        "2023-05-28",
        "Protest Against LNG Pipelines, Rügen",
        "Further strike opposing liquefied natural gas (LNG) pipeline construction on the island of Rügen.",
        "strike",
        False,
    ),
    KeyEvent(
        "2023-08-08",
        "Summer Congress, Lüneburg",
        "Congress marking five years since the movement's founding, focused on coordination and exchange.",
        "congress",
        False,
    ),
    KeyEvent(
        "2023-09-15",
        "Global Climate Strike (#EndFossilFuels)",
        "Global climate strike demanding an end to fossil fuels and a just transition.",
        "global_strike",
        True,
    ),
    KeyEvent(
        "2023-10-08",
        "State Elections, Hesse & Bavaria",
        "State elections held in Hesse and Bavaria.",
        "election",
        False,
    ),
    KeyEvent(
        "2023-11-20",
        "Public Climate School",
        "Recurring educational campaign of public lectures and workshops on climate science and policy.",
        "campaign",
        False,
    ),
    KeyEvent(
        "2023-11-24",
        "Protest Ahead of COP28",
        "Demonstration held shortly before the UN Climate Change Conference.",
        "strike",
        False,
    ),
    KeyEvent(
        "2023-11-30",
        "COP28",
        "UN Climate Change Conference, Dubai, UAE.",
        "COP",
        False,
    ),

    # 2024
    KeyEvent(
        "2024-02-01",
        "Protest Against Right-Wing Populism",
        "Nationwide protests against right-wing populism and fascism, and for an open and diverse society.",
        "strike",
        False,
    ),
    KeyEvent(
        "2024-03-01",
        "Climate Strike (#WirFahrenZusammen)",
        "Nationwide climate strike held jointly with public-transport workers, focusing on reliable public transport and better working conditions.",
        "strike",
        False,
    ),
    KeyEvent(
        "2024-05-31",
        "Climate Strike (ahead of EU election)",
        "Climate strike ahead of the European Parliament election, connecting climate protection with democratic participation.",
        "strike",
        False,
    ),
    KeyEvent(
        "2024-06-01",
        "Klimaklage 2.0 Campaign",
        "Campaign opposing the weakening of Germany's Climate Protection Act.",
        "campaign",
        False,
    ),
    KeyEvent(
        "2024-06-09",
        "EU Parliament Election",
        "Election to the European Parliament in Germany.",
        "eu_election",
        True,
    ),
    KeyEvent(
        "2024-07-25",
        "Summer Congress, Halle",
        "Summer congress in Halle with a diverse program of workshops and discussions.",
        "congress",
        False,
    ),
    KeyEvent(
        "2024-08-01",
        "Bo(h)rkum Campaign",
        "Protests against liquefied natural gas (LNG) projects in the German Baltic Sea near Borkum.",
        "strike",
        False,
    ),
    KeyEvent(
        "2024-09-20",
        "Global Climate Strike (#NowForFuture)",
        "Global climate strike demanding a rapid phase-out of coal, oil and gas and an end to fossil-fuel investments.",
        "global_strike",
        True,
    ),
    KeyEvent(
        "2024-10-10",
        "Campaign Against Gas Extraction, Bavaria",
        "Campaign opposing planned natural gas extraction projects in Bavaria.",
        "campaign",
        False,
    ),
    KeyEvent(
        "2024-11-01",
        "Campaign Around COP29",
        "Campaign activities accompanying the UN Climate Change Conference.",
        "campaign",
        False,
    ),
    KeyEvent(
        "2024-11-11",
        "COP29",
        "UN Climate Change Conference, Baku, Azerbaijan.",
        "COP",
        False,
    ),
    KeyEvent(
        "2024-12-10",
        "Protest Against Gas Lobby, Berlin",
        "Protest in Berlin against the fossil-gas industry's political lobbying.",
        "strike",
        False,
    ),

    # 2025
    KeyEvent(
        "2025-02-01",
        "Offener Brief Campaign",
        "Open letter to the German government, signed by over 100,000 citizens, demanding climate protection measures and opposing right-wing parties.",
        "campaign",
        False,
    ),
    KeyEvent(
        "2025-02-14",
        "Climate Strike (#RechtAufZukunft)",
        "Climate strike on climate policy during the German federal election campaign, demanding a safe and liveable future.",
        "strike",
        False,
    ),
    KeyEvent(
        "2025-02-23",
        "German Federal Election",
        "Election to the German Bundestag.",
        "bt_election",
        True,
    ),
    KeyEvent(
        "2025-03-21",
        "Keine Koalition ohne Klima",
        "Strike in Berlin demanding climate action be part of the new government coalition agreement.",
        "strike",
        False,
    ),
    KeyEvent(
        "2025-04-11",
        "International Day of Action",
        "International day of climate action calling for an immediate response to the escalating climate crisis.",
        "campaign",
        False,
    ),
    KeyEvent(
        "2025-05-03",
        "Stop Gas Drilling Campaign",
        "Demonstration against planned fossil-gas exploration in the Ammersee region and elsewhere in Bavaria.",
        "strike",
        False,
    ),
    KeyEvent(
        "2025-05-09",
        "Welt brennt, Zeit rennt! Campaign",
        "Fridays for Future presents demands for the new federal government's first 100 days, calling for accelerated climate action.",
        "campaign",
        False,
    ),
    KeyEvent(
        "2025-08-01",
        "Summer Congress, Munich",
        "Members meet in Munich for workshops and discussions, preparing campaigns for the fossil-gas phase-out.",
        "congress",
        False,
    ),
    KeyEvent(
        "2025-09-04",
        "Klimacamp Borkum",
        "Climate camp opposing planned gas drilling off Borkum, with workshops, demonstrations, concerts and networking.",
        "campaign",
        False,
    ),
    KeyEvent(
        "2025-09-20",
        "Global Climate Strike (#ExitGasEnterFuture)",
        "Worldwide climate demonstrations in around 100 countries opposing fossil-fuel expansion and calling for climate justice.",
        "global_strike",
        True,
    ),
    KeyEvent(
        "2025-11-01",
        "Taten statt Blockaden! Campaign",
        "Petition calling on Chancellor Friedrich Merz to take stronger climate action ahead of COP30.",
        "campaign",
        False,
    ),
    KeyEvent(
        "2025-11-01",
        "COP30 Campaign",
        "Fridays for Future participates in the UN Climate Change Conference in Belém, Brazil, reporting on negotiations and campaigning for climate justice.",
        "campaign",
        False,
    ),
    KeyEvent(
        "2025-11-10",
        "COP30",
        "UN Climate Change Conference, Belém, Brazil.",
        "COP",
        False,
    ),
    KeyEvent(
        "2025-11-14",
        "Global Climate Strike",
        "International climate strike shortly before COP30, calling for climate justice and stronger international climate action.",
        "global_strike",
        True,
    ),

    # 2026
    KeyEvent(
        "2026-02-01",
        "SPD: Jetzt Verantwortung übernehmen Campaign",
        "Mass email campaign calling on the SPD to oppose planned gas drilling off Borkum.",
        "campaign",
        False,
    ),
    KeyEvent(
        "2026-04-18",
        "Defend Renewable Energy Demonstrations",
        "Central demonstrations in Berlin, Hamburg, Cologne and Munich demanding protection of the energy transition and expansion of renewables.",
        "strike",
        False,
    ),
    KeyEvent(
        "2026-04-24",
        "Day of Action",
        "Nationwide climate actions calling for an end to fossil energy and expansion of renewables, targeting politicians and party offices.",
        "campaign",
        False,
    ),
    KeyEvent(
        "2026-05-01",
        "Stop Fossil Heating Campaign",
        "Petition with more than 150,000 signatures opposing the planned expansion of fossil heating and demanding a different heating policy.",
        "campaign",
        False,
    ),
    KeyEvent(
        "2026-05-30",
        "Zukunft statt Gas",
        "Climate movement gathers in Hamm to protest gas-fired power plants and demand a fossil-free future.",
        "strike",
        False,
    ),

]
