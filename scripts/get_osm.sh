#!/bin/bash
cd "$(dirname "$0")/.."
OUT=data/raw/manhattan.osm
UA="RapidRoute-student-project/1.0 (KJSSE ML lab)"
MIRRORS=(
  https://overpass-api.de/api/interpreter
  https://overpass.private.coffee/api/interpreter
  https://maps.mail.ru/osm/tools/overpass/api/interpreter
  https://overpass.kumi.systems/api/interpreter
)
for attempt in 1 2 3 4; do
  for url in "${MIRRORS[@]}"; do
    echo "[try $attempt] $url"
    curl -sS -L -A "$UA" --max-time 300 -o "$OUT" --data-urlencode data@src/data/manhattan_drive.overpassql "$url"
    size=$(wc -c < "$OUT" | tr -d ' ')
    if [ "$size" -gt 1000000 ] && head -c 200 "$OUT" | grep -q "<osm"; then
      echo "[ok] $OUT ($((size / 1048576)) MB) from $url"
      exit 0
    fi
    echo "  got $size bytes: $(head -c 80 "$OUT" | tr '\n' ' ')"
  done
  sleep $((attempt * 20))
done
echo "[fail] all mirrors busy, try again in a few minutes"
exit 1
