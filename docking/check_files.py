import yaml, os
cfg = yaml.safe_load(open("p4_config.yaml"))
for t, d in cfg["targets"].items():
    for k in ("receptor", "box", "ref_ligand"):
        print(t, k, "OK" if os.path.exists(d[k]) else "MISSING", d[k])