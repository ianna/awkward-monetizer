-- SQL user-defined function: dimuon invariant mass from two muon 4-vectors.
-- LANGUAGE SQL (portable — runs on a MonetDB server and on embedded monetdbe),
-- so the physics can be computed inside the database. CREATE OR REPLACE makes
-- re-loading idempotent. NOTE: this is a single statement (the ';' inside the
-- body are part of it) — load it without splitting on ';'.
CREATE OR REPLACE FUNCTION dimuon_mass(
    e1 DOUBLE, px1 DOUBLE, py1 DOUBLE, pz1 DOUBLE,
    e2 DOUBLE, px2 DOUBLE, py2 DOUBLE, pz2 DOUBLE)
RETURNS DOUBLE
BEGIN
    RETURN sqrt((e1 + e2) * (e1 + e2)
                - ((px1 + px2) * (px1 + px2)
                   + (py1 + py2) * (py1 + py2)
                   + (pz1 + pz2) * (pz1 + pz2)));
END;
