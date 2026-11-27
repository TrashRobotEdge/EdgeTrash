#!/usr/bin/env bash
# Run on the VM / laptop. Packs what a Pi needs into ONE file for a USB stick (or scp):
# the repo as a git bundle (commits included, so results keep their commit hash), the model
# files from models/files/ and the replayed bench frames listed in bench/configs/*.yaml.
#
#   bash setup/bundle.sh                 # -> edgetrash-bundle.tar in the repo root
# On the Pi:
#   tar xf edgetrash-bundle.tar -C ~ && bash ~/edgetrash-bundle/install.sh   # then pick the config
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$REPO/edgetrash-bundle.tar}"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
B="$STAGE/edgetrash-bundle"
mkdir -p "$B/files" "$B/frames"

cd "$REPO"
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "[!!] uncommitted changes are NOT in the bundle: commit first if the Pi needs them"
fi
git bundle create -q "$B/repo.bundle" HEAD "$(git branch --show-current)"

shopt -s nullglob
for f in models/files/*.{onnx,tflite,hef,blob,so,npy}; do cp "$f" "$B/files/"; done
for list in $(grep -h '^frames:' bench/configs/*.yaml | awk '{print $2}' | sort -u); do
  [[ -f "$list" ]] || continue
  while read -r img; do
    [[ -n "$img" && -f "$img" ]] && mkdir -p "$B/frames/$(dirname "$img")" && cp "$img" "$B/frames/$img"
  done < "$list"
done

cat > "$B/install.sh" <<'EOF'
#!/usr/bin/env bash
# On the Pi: clone/update ~/EdgeTrash from the bundle, copy models + frames, run setup.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
command -v git >/dev/null || { echo "[XX] git missing: sudo apt install git, then re-run this script"; exit 1; }
DEST="${EDGETRASH_DIR:-$HOME/EdgeTrash}"
(cd "$HERE" && sha256sum -c --quiet SHA256SUMS) || { echo "[XX] bundle corrupted: copy it again"; exit 1; }
if [[ -d "$DEST/.git" ]]; then
  git -C "$DEST" fetch -q "$HERE/repo.bundle" "+HEAD:refs/remotes/bundle/head"
  git -C "$DEST" checkout -q -B bundle bundle/head
else
  git clone -q "$HERE/repo.bundle" "$DEST"
fi
mkdir -p "$DEST/models/files"
cp "$HERE"/files/* "$DEST/models/files/" 2>/dev/null || true
cp -r "$HERE/frames/." "$DEST/"
echo "[ok] $DEST at commit $(git -C "$DEST" rev-parse --short HEAD)"
exec bash "$DEST/setup/setup.sh" "$@"
EOF
(cd "$B" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
tar cf "$OUT" -C "$STAGE" edgetrash-bundle
echo "[ok] $OUT ($(du -h "$OUT" | cut -f1)): $(ls "$B/files" | wc -l) model files, $(find "$B/frames" -type f | wc -l) frames"
