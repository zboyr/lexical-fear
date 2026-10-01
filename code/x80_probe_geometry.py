"""x80 aggregate: data/results/x80_probe_geometry.json from the per-seed probe runs.

Reads the three primary runs (x80_fit_paired_probes.py on the pre-registered
x71 L2 grid, data/runs/x80/probes/x80_probes_s{42,123,7}.json) and the three
extended-L2-grid sensitivity runs (--l2-ext, data/runs/x80/probes_l2ext/).

  seeds                      each primary run's JSON without its per_state block
  per_state_skill            per seed and state: R/H deviance skill, cos(q_R, q_H), eligibility
  H_caveat                   H is comply2 compliance, not judged harm
  l2_extended_sensitivity    each extended-grid run without per_state, plus its own
                             per_state_skill (adds the selected L2 per head)
  note                       which grid is primary

Extracted from the two inline scripts that produced the released file (the
aggregation, then the extended-grid patch); the logic is unchanged, the
diagnostic printing is dropped.

Usage, from anywhere inside the checkout:
  python code/x80_probe_geometry.py [--probes-dir data/runs/x80/probes]
      [--l2ext-dir data/runs/x80/probes_l2ext] [--out data/results/x80_probe_geometry.json]
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SEEDS = (42, 123, 7)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--probes-dir", default=os.path.join(ROOT, "data", "runs", "x80", "probes"))
    ap.add_argument("--l2ext-dir", default=os.path.join(ROOT, "data", "runs", "x80", "probes_l2ext"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "results", "x80_probe_geometry.json"))
    args = ap.parse_args()

    # Stage 1: primary (pre-registered L2 grid) aggregation.
    out = {}
    for s in SEEDS:
        out[s] = json.load(open(os.path.join(args.probes_dir, f"x80_probes_s{s}.json")))
    g = {"seeds": {s: {k: v for k, v in d.items() if k != "per_state"} for s, d in out.items()},
         "per_state_skill": {s: {k: {"R": out[s]["per_state"][k]["R"]["deviance_skill"],
                                     "H": out[s]["per_state"][k]["H"]["deviance_skill"],
                                     "cos_qR_qH": out[s]["per_state"][k]["cos_qR_qH"],
                                     "eligible": out[s]["per_state"][k]["eligible"]}
                                 for k in out[s]["per_state"]} for s in out},
         "H_caveat": "H is comply2 substantive compliance, not judged harm"}
    # Stage 1 was written to disk and re-read before the patch below; the JSON
    # round trip turns the integer seed keys into strings, as in the released file.
    g = json.loads(json.dumps(g))

    # Stage 2: extended-L2-grid sensitivity patch.
    g["l2_extended_sensitivity"] = {}
    for s in SEEDS:
        d = json.load(open(os.path.join(args.l2ext_dir, f"x80_probes_s{s}.json")))
        g["l2_extended_sensitivity"][s] = {k: v for k, v in d.items() if k != "per_state"}
        g["l2_extended_sensitivity"][s]["per_state_skill"] = {
            k: {"R": d["per_state"][k]["R"]["deviance_skill"],
                "H": d["per_state"][k]["H"]["deviance_skill"],
                "cos_qR_qH": d["per_state"][k]["cos_qR_qH"],
                "l2_R": d["per_state"][k]["R"]["l2"],
                "l2_H": d["per_state"][k]["H"]["l2"]}
            for k in d["per_state"]}
    g["note"] = ("primary = pre-registered x71 L2 grid (saturated at its top value 1e-3 for every "
                 "head/state); l2_extended_sensitivity = grid [1e-3..1] with interior optima")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(g, open(args.out, "w"), indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
