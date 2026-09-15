-- Standalone jets table (subset of nanoaod.sql), kept for reference.
CREATE TABLE jets (
    event_id  BIGINT,
    jet_index INT,
    pt        DOUBLE,
    eta       DOUBLE,
    phi       DOUBLE,
    mass      DOUBLE
);
CREATE INDEX jets_event_idx ON jets(event_id);
