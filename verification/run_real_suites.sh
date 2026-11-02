#!/usr/bin/env bash
# Run the CosmoViewer test suites against real Gadget-4 output.
#
#   verification/run_real_suites.sh <data_root> <scratch> [immersive|explorer|both] [extra pytest args...]
#
# <data_root> must hold PLAIN COPIES (not symlinks) of
#     DM-L50-N128/output            snapshot_000..002.hdf5, fof_tab_000..002.hdf5, energy.txt
#     CollidingGalaxiesSFR/output   snapshot_NNN.hdf5 (any subset of the run), energy.txt
# because the apps write cosmoviewer_sets/ next to the snapshots and
# selection_sets.candidate_dirs() resolves symlinks: a folder of symlinks would
# make them write into the original output folder.
#
# <scratch> receives an isolated HOME (so ~/.cosmoviewer caches, set fallbacks and
# Explorer instance files of other runs are neither used nor touched) and the logs.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="$(cd "$1" && pwd)"; SCRATCH="$2"; WHICH="${3:-both}"; shift 3 2>/dev/null || shift $#
mkdir -p "$SCRATCH/home" "$SCRATCH/logs"
export QT_QPA_PLATFORM=offscreen
export HOME="$SCRATCH/home"
G4="$DATA/DM-L50-N128/output"
CG="$DATA/CollidingGalaxiesSFR/output"
for f in "$G4"/snapshot_000.hdf5 "$CG"; do
  [ -e "$f" ] || { echo "missing $f" >&2; exit 2; }
  [ -L "$f" ] && { echo "$f is a symlink; use plain copies" >&2; exit 2; }
done
stamp="$(date +%Y%m%d-%H%M%S)"
echo "commit: $(git -C "$ROOT" rev-parse --short HEAD)  data: $DATA  home: $HOME"

if [ "$WHICH" = immersive ] || [ "$WHICH" = both ]; then
  log="$SCRATCH/logs/immersive-$stamp.log"; SECONDS=0
  ( cd "$ROOT/Cosmoviewer Immersive" && \
    COSMOVIEWER_G4_OUTPUT="$G4" COSMOVIEWER_CG_OUTPUT="$CG" \
    COSMOVIEWER_DM_SNAPSHOT="$G4/snapshot_000.hdf5" \
    python3 -m pytest tests -q -p no:cacheprovider -rfEs --durations=25 "$@" ) \
    > "$log" 2>&1
  echo "immersive exit $? after ${SECONDS}s -> $log"; tail -n 5 "$log"
fi
if [ "$WHICH" = explorer ] || [ "$WHICH" = both ]; then
  log="$SCRATCH/logs/explorer-$stamp.log"; SECONDS=0
  ( cd "$ROOT/Dataset Explorer" && \
    COSMOVIEWER_DM_OUTPUT="$G4" COSMOVIEWER_SFR_OUTPUT="$CG" \
    python3 -m pytest -q -p no:cacheprovider -rfEs --durations=25 "$@" ) \
    > "$log" 2>&1
  echo "explorer exit $? after ${SECONDS}s -> $log"; tail -n 5 "$log"
fi
