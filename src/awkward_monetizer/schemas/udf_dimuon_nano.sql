-- SQL UDF: dimuon invariant mass from two NanoAOD muons in (pt, eta, phi, mass)
-- form. Real NanoAOD stores Muon_pt/eta/phi/mass, not the flat ntuple's
-- Cartesian E/px/py/pz, so this is the cylindrical-coordinate sibling of
-- dimuon_mass. sinh(eta) is computed via exp (MonetDB has no sinh), matching the
-- trijet UDFs. LANGUAGE SQL (portable across the server and embedded monetdbe);
-- single statement -- load with db.apply_sql(split=False).
CREATE OR REPLACE FUNCTION dimuon_mass_nano(
    pt1 DOUBLE, eta1 DOUBLE, phi1 DOUBLE, m1 DOUBLE,
    pt2 DOUBLE, eta2 DOUBLE, phi2 DOUBLE, m2 DOUBLE)
RETURNS DOUBLE
BEGIN
    DECLARE px DOUBLE, py DOUBLE, pz DOUBLE, e DOUBLE;
    SET px = pt1*cos(phi1) + pt2*cos(phi2);
    SET py = pt1*sin(phi1) + pt2*sin(phi2);
    SET pz = pt1*(exp(eta1)-exp(-eta1))/2
           + pt2*(exp(eta2)-exp(-eta2))/2;
    SET e = sqrt(power(pt1*cos(phi1),2) + power(pt1*sin(phi1),2)
                 + power(pt1*(exp(eta1)-exp(-eta1))/2,2) + m1*m1)
          + sqrt(power(pt2*cos(phi2),2) + power(pt2*sin(phi2),2)
                 + power(pt2*(exp(eta2)-exp(-eta2))/2,2) + m2*m2);
    RETURN sqrt(e*e - (px*px + py*py + pz*pz));
END;
