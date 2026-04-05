#!/usr/bin/env python3
"""Trajectory and topology manipulation tools.

Provides command-line utilities for:
  - Writing subsets of trajectories (strip water, select atoms)
  - Centering and wrapping atoms around a selection
  - Aligning trajectories to a reference frame
  - Writing new topology files (PSF/PDB) for subsets
  - Striding trajectories

Supports DCD, XTC, and TRR trajectory formats (anything MDAnalysis reads/writes).

Usage
-----
::

    # Strip water from a trajectory, write XTC
    traj-tools write --top sys.psf --traj prod.dcd \\
        --select "not resname TIP3 SOD CLA" --out nowat.xtc

    # Center on protein, wrap all atoms in the box
    traj-tools write --top sys.psf --traj prod.dcd \\
        --select "all" --center "protein" --wrap --out centered.xtc

    # Align on backbone, write only protein
    traj-tools write --top sys.psf --traj prod.dcd \\
        --select "protein" --align "protein and backbone" --out aligned.xtc

    # Write a new topology (PDB) for a subset
    traj-tools topology --top sys.psf --traj prod.dcd \\
        --select "protein or resname POPC POPE POPS" --out nowat.pdb

    # Stride a trajectory
    traj-tools write --top sys.psf --traj prod.dcd \\
        --select "all" --stride 10 --out strided.xtc

    # Combine: strip water + center on protein + align + stride
    traj-tools write --top sys.psf --traj prod.dcd \\
        --select "not resname TIP3 SOD CLA" \\
        --center "protein" --wrap \\
        --align "protein and name CA" \\
        --stride 10 --out final.xtc
"""

import argparse
import os
import sys

import numpy as np
import MDAnalysis as mda
from MDAnalysis import transformations
from tqdm import tqdm


# ── Core functions ───────────────────────────────────────────────────────────

def write_trajectory(top, traj, out_path, select="all", center_sel=None,
                     wrap=False, align_sel=None, ref_frame=0,
                     stride=1, start=None, stop=None):
    """Write a (sub)trajectory with optional centering, wrapping, and alignment.

    Parameters
    ----------
    top        : str — topology file (PSF, PDB, GRO, TPR, etc.)
    traj       : str — trajectory file (DCD, XTC, TRR, etc.)
    out_path   : str — output trajectory path (format inferred from extension)
    select     : str — MDAnalysis selection for atoms to write
    center_sel : str or None — selection to center at the box origin
    wrap       : bool — wrap all atoms into the primary unit cell after centering
    align_sel  : str or None — selection for RMSD alignment to ref_frame
    ref_frame  : int — reference frame for alignment (default 0)
    stride     : int — write every Nth frame
    start      : int or None — first frame index
    stop       : int or None — last frame index (exclusive)
    """
    print(f"Loading: {top} + {traj}")
    u = mda.Universe(top, traj)

    # Build the atom group to write
    ag = u.select_atoms(select)
    print(f"  Selection: '{select}' -> {ag.n_atoms} atoms")

    if ag.n_atoms == 0:
        print("  ERROR: selection matched zero atoms.")
        return

    # Build on-the-fly transformations
    workflow = []

    if center_sel:
        center_ag = u.select_atoms(center_sel)
        print(f"  Centering on: '{center_sel}' ({center_ag.n_atoms} atoms)")
        if center_ag.n_atoms == 0:
            print("  WARNING: center selection matched zero atoms, skipping centering.")
        else:
            workflow.append(transformations.center_in_box(center_ag, wrap=wrap))

    if wrap and not center_sel:
        # Wrap without explicit centering — wrap around box center
        workflow.append(transformations.wrap(ag))

    if align_sel:
        ref = mda.Universe(top, traj)
        ref.trajectory[ref_frame]
        align_ag_mobile = u.select_atoms(align_sel)
        align_ag_ref = ref.select_atoms(align_sel)
        print(f"  Aligning on: '{align_sel}' ({align_ag_mobile.n_atoms} atoms) "
              f"to frame {ref_frame}")
        if align_ag_mobile.n_atoms == 0:
            print("  WARNING: align selection matched zero atoms, skipping alignment.")
        else:
            workflow.append(
                transformations.fit_rot_trans(align_ag_mobile, align_ag_ref)
            )

    if workflow:
        u.trajectory.add_transformations(*workflow)

    # Determine frame range
    total = len(u.trajectory)
    _start = start if start is not None else 0
    _stop = stop if stop is not None else total
    frame_slice = slice(_start, _stop, stride)
    n_frames = len(range(*frame_slice.indices(total)))

    print(f"  Writing {n_frames} frames to: {out_path}")

    # Also write a reference PDB topology for convenience
    pdb_path = os.path.splitext(out_path)[0] + ".pdb"
    u.trajectory[_start]
    ag.write(pdb_path)
    print(f"  Reference topology: {pdb_path}")

    with mda.Writer(out_path, ag.n_atoms) as W:
        for ts in tqdm(u.trajectory[frame_slice], total=n_frames,
                       desc="  Writing"):
            W.write(ag)

    print(f"  Done: {out_path} ({n_frames} frames, {ag.n_atoms} atoms)")


def write_topology(top, traj, out_path, select="all", frame=0):
    """Write a topology file (PDB/GRO) for a subset of atoms.

    Parameters
    ----------
    top      : str — input topology file
    traj     : str or None — trajectory file (optional, for coordinates)
    out_path : str — output topology path (PDB, GRO, etc.)
    select   : str — MDAnalysis selection
    frame    : int — frame to use for coordinates
    """
    if traj:
        u = mda.Universe(top, traj)
    else:
        u = mda.Universe(top)

    ag = u.select_atoms(select)
    print(f"  Selection: '{select}' -> {ag.n_atoms} atoms")

    if ag.n_atoms == 0:
        print("  ERROR: selection matched zero atoms.")
        return

    if traj:
        u.trajectory[frame]

    ag.write(out_path)
    print(f"  Wrote topology: {out_path} ({ag.n_atoms} atoms, frame {frame})")


# ── PSF writing ──────────────────────────────────────────────────────────────

def write_psf_subset(top, out_path, select="all", traj=None):
    """Write a new PSF topology for a subset of atoms.

    MDAnalysis can write PSF files if the input topology is PSF.
    The new PSF will contain only the selected atoms with bond/angle
    information preserved for those atoms.

    A trajectory is required because MDAnalysis needs coordinates loaded
    to write. If no trajectory is provided, provide a PDB or use the
    ``topology`` subcommand with a PDB instead.

    Parameters
    ----------
    top      : str — input PSF file
    out_path : str — output PSF path
    select   : str — MDAnalysis selection
    traj     : str or None — trajectory for coordinates
    """
    if traj:
        u = mda.Universe(top, traj)
    else:
        # Try loading PSF alone — needs coordinates for writing
        # Fall back to creating dummy coords if needed
        try:
            u = mda.Universe(top)
            # Check if we can access trajectory
            _ = u.trajectory
        except (AttributeError, Exception):
            print("  PSF requires a trajectory (--traj) or PDB for coordinates.")
            print("  Use: traj-tools topology --top sys.psf --traj prod.dcd "
                  "--select '...' --out subset.psf")
            return

    ag = u.select_atoms(select)
    print(f"  Selection: '{select}' -> {ag.n_atoms} atoms")

    if ag.n_atoms == 0:
        print("  ERROR: selection matched zero atoms.")
        return

    # MDAnalysis can't write PSF directly — use parmed if available
    try:
        import parmed
        parm = parmed.load_file(top)
        # Build index set of atoms to keep
        keep_indices = set(ag.indices)
        remove_indices = [i for i in range(len(parm.atoms)) if i not in keep_indices]
        parm.strip('@' + ','.join(str(i + 1) for i in sorted(remove_indices)))
        parm.save(out_path, overwrite=True)
        print(f"  Wrote PSF (via parmed): {out_path} ({ag.n_atoms} atoms)")
    except ImportError:
        # Fallback: write a PDB instead
        pdb_fallback = os.path.splitext(out_path)[0] + ".pdb"
        if traj:
            u.trajectory[0]
        ag.write(pdb_fallback)
        print(f"  parmed not installed — cannot write PSF directly.")
        print(f"  Wrote PDB instead: {pdb_fallback} ({ag.n_atoms} atoms)")
        print(f"  Install parmed for PSF support: pip install parmed")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="traj-tools",
        description="Trajectory and topology manipulation tools",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── write subcommand ─────────────────────────────────────────────────
    write_p = subparsers.add_parser(
        "write", help="Write a (sub)trajectory with optional transformations")
    write_p.add_argument("--top", required=True, help="Topology file (PSF, PDB, GRO, TPR)")
    write_p.add_argument("--traj", required=True, help="Trajectory file (DCD, XTC, TRR)")
    write_p.add_argument("--out", required=True, help="Output trajectory path")
    write_p.add_argument("--select", default="all",
                         help="Atom selection to write (default: all)")
    write_p.add_argument("--center", default=None,
                         help="Selection to center at box origin")
    write_p.add_argument("--wrap", action="store_true",
                         help="Wrap atoms into primary unit cell")
    write_p.add_argument("--align", default=None,
                         help="Selection for RMSD alignment to --ref-frame")
    write_p.add_argument("--ref-frame", type=int, default=0,
                         help="Reference frame for alignment (default: 0)")
    write_p.add_argument("--stride", type=int, default=1,
                         help="Write every Nth frame")
    write_p.add_argument("--start", type=int, default=None,
                         help="First frame index")
    write_p.add_argument("--stop", type=int, default=None,
                         help="Last frame index (exclusive)")

    # ── topology subcommand ──────────────────────────────────────────────
    topo_p = subparsers.add_parser(
        "topology", help="Write a new topology file for a subset of atoms")
    topo_p.add_argument("--top", required=True, help="Input topology file")
    topo_p.add_argument("--traj", default=None, help="Trajectory (optional, for coordinates)")
    topo_p.add_argument("--out", required=True, help="Output topology path (PDB, GRO, PSF)")
    topo_p.add_argument("--select", default="all",
                        help="Atom selection (default: all)")
    topo_p.add_argument("--frame", type=int, default=0,
                        help="Frame for coordinates (default: 0)")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == "write":
        write_trajectory(
            top=args.top, traj=args.traj, out_path=args.out,
            select=args.select, center_sel=args.center,
            wrap=args.wrap, align_sel=args.align,
            ref_frame=args.ref_frame, stride=args.stride,
            start=args.start, stop=args.stop,
        )

    elif args.command == "topology":
        out_ext = os.path.splitext(args.out)[1].lower()
        if out_ext == ".psf":
            write_psf_subset(args.top, args.out, select=args.select, traj=args.traj)
        else:
            write_topology(args.top, args.traj, args.out,
                           select=args.select, frame=args.frame)


if __name__ == "__main__":
    main()
