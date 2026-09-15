# data/

Data files are **not** committed (see `.gitignore`). Put inputs here.

## `cms.root` — dimuon sample
The flat CMS dimuon teaching ntuple (tree `events`, 2,304 events; branches
`Run, Event, Type, M` and two muons as `pt1/pt2`, `Q1/Q2`, …). It's the default
input for the `dimuon` dataset. This is a well-known CMS open-data outreach file
(the "Dimuon" / DoubleMu sample); drop your copy here as `data/cms.root`.

## `nano_synth.root` — synthetic NanoAOD
Generated on demand for the NanoAOD path and the ADL benchmark:

```
awkward-monetizer make-nano            # writes data/nano_synth.root
# or:  python examples/make_sample_nanoaod.py
```

To run the ADL queries on **real** data, point `--from-root` at a CMS Open Data
NanoAOD file instead.
