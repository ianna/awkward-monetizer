-- Schema for the NanoAOD-style 'nanoaod' dataset: event-level MET plus jagged
-- jets and muons exploded into flat tables. Column order matches the DataFrames
-- ingest builds. (A 'tracks' table is planned but not yet loaded.)

CREATE TABLE events (
    event_id BIGINT PRIMARY KEY,
    run      INT,
    met_pt   DOUBLE,
    met_phi  DOUBLE,
    n_jets   INT
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
