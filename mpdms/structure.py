"""Structure utilities: residue table, secondary structure, RSA, contacts, membrane frame."""
from __future__ import annotations

import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .annot import MAX_ASA

warnings.filterwarnings("ignore", module="Bio")


def load_structure(path: Path, chain: str = "A"):
    from Bio.PDB import MMCIFParser, PDBParser
    path = Path(path)
    parser = MMCIFParser(QUIET=True) if path.suffix in (".cif", ".mmcif") else PDBParser(QUIET=True)
    s = parser.get_structure("s", str(path))
    model = s[0]
    if chain not in model:
        chain = next(iter(model)).id
    return s, model, model[chain]


def residue_table(path: Path, chain: str = "A", offset: int = 0) -> pd.DataFrame:
    """One row per residue: pos, aa, CA xyz, pLDDT (B-factor), SS (H/E/C), RSA, contacts."""
    from Bio.PDB.Polypeptide import three_to_index, index_to_one
    s, model, ch = load_structure(path, chain)
    rows = []
    for res in ch:
        if res.id[0] != " " or "CA" not in res:
            continue
        try:
            aa = index_to_one(three_to_index(res.get_resname()))
        except (KeyError, ValueError):
            aa = "X"
        ca = res["CA"].coord
        rows.append({"pos": res.id[1] + offset, "aa": aa, "x": ca[0], "y": ca[1], "z": ca[2],
                     "plddt": res["CA"].bfactor})
    rt = pd.DataFrame(rows)
    rt["ss"] = secondary_structure(path, ch.id, model, len(rt))
    rt["rsa"] = relative_sasa(s, ch)
    rt["contacts"] = contact_number(ch)
    return rt


def secondary_structure(path: Path, chain: str, model, n: int) -> list[str]:
    """3-state SS per residue. DSSP if available, else pydssp, else phi/psi rules."""
    exe = shutil.which("mkdssp") or shutil.which("dssp")
    if exe:
        try:
            from Bio.PDB.DSSP import DSSP
            d = DSSP(model, str(path), dssp=exe)
            ss = [d[k][2] for k in d.keys() if k[0] == chain]
            if len(ss) == n:
                return [_to3(c) for c in ss]
        except Exception:
            pass
    try:
        import pydssp
        coords = []
        for res in model[chain]:
            if res.id[0] == " " and "CA" in res:
                coords.append([res[a].coord if a in res else res["CA"].coord for a in ("N", "CA", "C", "O")])
        ss = pydssp.assign(np.array(coords), out_type="c3")
        return ["C" if c == "-" else c for c in ss]
    except Exception:
        return _phi_psi_ss(model[chain])


def _to3(c: str) -> str:
    return "H" if c in "HGI" else ("E" if c in "EB" else "C")


def _phi_psi_ss(chain) -> list[str]:
    from Bio.PDB.Polypeptide import PPBuilder
    out = []
    for pp in PPBuilder().build_peptides(chain):
        for phi, psi in pp.get_phi_psi_list():
            if phi is None or psi is None:
                out.append("C"); continue
            phi, psi = np.degrees(phi), np.degrees(psi)
            if -160 < phi < -20 and -120 < psi < 50:
                out.append("H")
            elif -180 < phi < -40 and (psi > 90 or psi < -150):
                out.append("E")
            else:
                out.append("C")
    # smooth: helices need >= 4 consecutive
    s = "".join(out)
    import re
    s = re.sub(r"(?<!H)H{1,3}(?!H)", lambda m: "C" * len(m.group()), s)
    return list(s)


def relative_sasa(structure, chain) -> list[float]:
    """Shrake-Rupley SASA / Tien max ASA. Computed on the isolated chain (no membrane)."""
    from Bio.PDB.SASA import ShrakeRupley
    from Bio.PDB.Polypeptide import three_to_index, index_to_one
    sr = ShrakeRupley()
    sr.compute(chain, level="R")
    out = []
    for res in chain:
        if res.id[0] != " " or "CA" not in res:
            continue
        try:
            aa = index_to_one(three_to_index(res.get_resname()))
            out.append(min(1.0, res.sasa / MAX_ASA[aa]))
        except (KeyError, ValueError):
            out.append(np.nan)
    return out


def contact_number(chain, cutoff: float = 8.0, exclude: int = 4) -> list[int]:
    """Residues with any heavy atom within cutoff, excluding |i-j| <= exclude."""
    from scipy.spatial import cKDTree
    atoms, owner = [], []
    residues = [r for r in chain if r.id[0] == " " and "CA" in r]
    for k, res in enumerate(residues):
        for a in res:
            if a.element != "H":
                atoms.append(a.coord); owner.append(k)
    atoms, owner = np.array(atoms), np.array(owner)
    tree = cKDTree(atoms)
    pairs = tree.query_pairs(cutoff, output_type="ndarray")
    ri, rj = owner[pairs[:, 0]], owner[pairs[:, 1]]
    keep = np.abs(ri - rj) > exclude
    rp = {(min(a, b), max(a, b)) for a, b in zip(ri[keep], rj[keep])}
    cnt = np.zeros(len(residues), int)
    for a, b in rp:
        cnt[a] += 1; cnt[b] += 1
    return cnt.tolist()


def membrane_frame(rt: pd.DataFrame, cfg) -> tuple[pd.DataFrame, dict]:
    """Add columns zdepth (signed, A) and radial (A) given structure.membrane_normal."""
    method = cfg.get_path("structure.membrane_normal", "pca_tm_axes")
    xyz = rt[["x", "y", "z"]].to_numpy(float)
    tm = [s for s in cfg.get_path("topology.segments", []) or [] if s.get("type") == "TM"]
    info = {"method": method}
    if method in ("ppm", "opm"):
        # PPM/OPM output files are already oriented: membrane normal = z, centre z = 0
        normal, centre = np.array([0, 0, 1.0]), np.array([0, 0, float(cfg.get_path("structure.membrane_center_z", 0.0))])
    elif method == "pca_tm_axes" and tm:
        axes = []
        for s in tm:
            m = rt["pos"].between(s["start"], s["end"]).to_numpy()
            if m.sum() < 7:
                continue
            c = xyz[m] - xyz[m].mean(0)
            v = np.linalg.svd(c, full_matrices=False)[2][0]
            # orient every axis N->C then flip in_out/out_in so all point the same way
            if np.dot(v, xyz[m][-1] - xyz[m][0]) < 0:
                v = -v
            if s.get("orientation") == "out_in":
                v = -v
            axes.append(v)
        if not axes:
            return rt.assign(zdepth=np.nan, radial=np.nan), {"method": "none", "reason": "no TM >= 7 residues"}
        normal = np.mean(axes, axis=0)
        normal /= np.linalg.norm(normal)
        tm_mask = np.zeros(len(rt), bool)
        for s in tm:
            tm_mask |= rt["pos"].between(s["start"], s["end"]).to_numpy()
        centre = xyz[tm_mask].mean(0)
        info.update(n_tm_axes=len(axes), note="normal = mean of per-TMD principal axes (sign-aligned by "
                                               "orientation); z = 0 at mean TM CA; +z = lumenal side")
    else:
        return rt.assign(zdepth=np.nan, radial=np.nan), {"method": "none"}
    rel = xyz - centre
    z = rel @ normal
    radial = np.linalg.norm(rel - np.outer(z, normal), axis=1)
    return rt.assign(zdepth=z, radial=radial), info
