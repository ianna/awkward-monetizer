-- SQL UDFs for the ADL Q6 trijet analysis (NanoAOD jets). Two LANGUAGE SQL
-- functions built from jet (pt, eta, phi, mass); sinh(eta) is computed via exp.
-- Load with db.load_functions() (splits on `END;`, since each body has its own
-- semicolons). These exist to show the *contrast* with `ak.combinations`: the
-- physics is expressible in SQL, but the Q6 selection then needs a 3-way
-- self-join (all jet triples) plus a window function to pick the best per event.

CREATE OR REPLACE FUNCTION trijet_pt(
    pt1 DOUBLE, phi1 DOUBLE, pt2 DOUBLE, phi2 DOUBLE, pt3 DOUBLE, phi3 DOUBLE)
RETURNS DOUBLE
BEGIN
    RETURN sqrt(power(pt1*cos(phi1) + pt2*cos(phi2) + pt3*cos(phi3), 2)
              + power(pt1*sin(phi1) + pt2*sin(phi2) + pt3*sin(phi3), 2));
END;

CREATE OR REPLACE FUNCTION trijet_mass(
    pt1 DOUBLE, eta1 DOUBLE, phi1 DOUBLE, m1 DOUBLE,
    pt2 DOUBLE, eta2 DOUBLE, phi2 DOUBLE, m2 DOUBLE,
    pt3 DOUBLE, eta3 DOUBLE, phi3 DOUBLE, m3 DOUBLE)
RETURNS DOUBLE
BEGIN
    DECLARE px DOUBLE, py DOUBLE, pz DOUBLE, e DOUBLE;
    SET px = pt1*cos(phi1) + pt2*cos(phi2) + pt3*cos(phi3);
    SET py = pt1*sin(phi1) + pt2*sin(phi2) + pt3*sin(phi3);
    SET pz = pt1*(exp(eta1)-exp(-eta1))/2
           + pt2*(exp(eta2)-exp(-eta2))/2
           + pt3*(exp(eta3)-exp(-eta3))/2;
    SET e = sqrt(power(pt1*cos(phi1),2) + power(pt1*sin(phi1),2)
                 + power(pt1*(exp(eta1)-exp(-eta1))/2,2) + m1*m1)
          + sqrt(power(pt2*cos(phi2),2) + power(pt2*sin(phi2),2)
                 + power(pt2*(exp(eta2)-exp(-eta2))/2,2) + m2*m2)
          + sqrt(power(pt3*cos(phi3),2) + power(pt3*sin(phi3),2)
                 + power(pt3*(exp(eta3)-exp(-eta3))/2,2) + m3*m3);
    RETURN sqrt(e*e - (px*px + py*py + pz*pz));
END;
