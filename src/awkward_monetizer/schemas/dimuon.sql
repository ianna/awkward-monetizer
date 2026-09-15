-- Schema for the flat CMS dimuon teaching ntuple (the 'dimuon' dataset).
-- Column order matches the DataFrames ingest builds, so COPY INTO (headerless
-- CSV) loads them positionally. event_id is a synthetic row id (0..N-1).

CREATE TABLE events (
    event_id     BIGINT PRIMARY KEY,
    run          INT,
    event_number BIGINT,
    muon_type    VARCHAR(8),
    mass         DOUBLE,
    n_muons      INT
);

CREATE TABLE muons (
    event_id   BIGINT,
    muon_index INT,
    e          DOUBLE,
    px         DOUBLE,
    py         DOUBLE,
    pz         DOUBLE,
    pt         DOUBLE,
    eta        DOUBLE,
    phi        DOUBLE,
    charge     INT
);
CREATE INDEX muons_event_idx ON muons(event_id);
