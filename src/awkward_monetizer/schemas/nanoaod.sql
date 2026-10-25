-- Schema for the NanoAOD-style 'nanoaod' dataset: event-level MET plus jagged
-- jets and muons exploded into flat tables. Column order matches the DataFrames
-- ingest builds. (A 'tracks' table is planned but not yet loaded.)

CREATE TABLE events (
    event_id     BIGINT PRIMARY KEY,   -- (file_id << 40) | entry  (see keys.py)
    run          INT,
    lumi         INT,                  -- luminosityBlock
    event_number BIGINT,               -- NanoAOD `event` (not unique on its own)
    met_pt       DOUBLE,
    met_phi      DOUBLE,
    -- trigger flags: NULL when the file's menu doesn't contain the path
    hlt_isomu24     BOOLEAN,           -- HLT_IsoMu24
    hlt_isomu27     BOOLEAN,           -- HLT_IsoMu27
    hlt_mu17_mu8_dz BOOLEAN,           -- HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ
    -- stored multiplicities, so object-count cuts never need GROUP BY
    n_jets       INT,
    n_muons      INT
);

CREATE TABLE jets (
    event_id  BIGINT,
    jet_index INT,
    pt        DOUBLE,
    eta       DOUBLE,
    phi       DOUBLE,
    mass      DOUBLE
);
CREATE INDEX jets_event_idx ON jets(event_id);

CREATE TABLE muons (
    event_id   BIGINT,
    muon_index INT,
    pt         DOUBLE,
    eta        DOUBLE,
    phi        DOUBLE,
    mass       DOUBLE,
    charge     INT
);
CREATE INDEX muons_event_idx ON muons(event_id);
