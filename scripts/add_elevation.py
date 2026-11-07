#!/usr/bin/env python3
"""Add `elevation_meters` to every location in data/modern.jsonl.

The elevation is sampled from AWS Terrain Tiles at each location's `lonlat`
(see terrain_tiles.py). Only that one key changes: the script first checks that
it can re-serialize every existing line byte for byte, and refuses to write
otherwise, so the diff of data/modern.jsonl is exactly the new field.

    python3 scripts/add_elevation.py            # fill in / refresh the field
    python3 scripts/add_elevation.py --check    # exit 1 if the file is out of date

Downloaded tiles are cached in .terrain-cache/ (about 30 MB at zoom 12), so a
second run needs no network.
"""

import argparse
import json
import os
import sys

from terrain_tiles import DEFAULT_ZOOM, MAX_ZOOM, TILE_SIZE, TerrainTiles, lonlat_to_global_pixel

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIELD = "elevation_meters"


def dumps(record):
    """The exact serialization used by the dataset's .jsonl files."""
    return json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def parse_lonlat(value):
    longitude, latitude = value.split(",")
    return float(longitude), float(latitude)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", default=os.path.join(REPO_ROOT, "data", "modern.jsonl"),
                        help="the modern.jsonl file to update (default: %(default)s)")
    parser.add_argument("--zoom", type=int, default=DEFAULT_ZOOM,
                        help="tile zoom level, 0-%d (default: %%(default)s)" % MAX_ZOOM)
    parser.add_argument("--cache-dir", default=os.path.join(REPO_ROOT, ".terrain-cache"),
                        help="where to keep downloaded tiles (default: %(default)s)")
    parser.add_argument("--workers", type=int, default=8, help="parallel downloads (default: %(default)s)")
    parser.add_argument("--check", action="store_true",
                        help="don't write; exit 1 if any elevation is missing or different")
    args = parser.parse_args()

    with open(args.data, encoding="utf-8", newline="") as handle:
        original = handle.read()
    lines = original.split("\n")  # keeps the file's trailing-newline state intact

    records = []  # one dict per line; None for the empty string a trailing newline leaves
    for number, line in enumerate(lines, 1):
        if not line:
            records.append(None)
            continue
        record = json.loads(line)
        # Guard: if re-serializing an untouched line doesn't reproduce it, rewriting
        # the file would reformat the whole dataset, not just add a field.
        if dumps(record) != line:
            sys.exit("line %d of %s does not round-trip through json.dumps; refusing to "
                     "rewrite the file and change its formatting" % (number, args.data))
        records.append(record)

    tiles = TerrainTiles(zoom=args.zoom, cache_dir=args.cache_dir)
    located = [record for record in records if record]
    points = [parse_lonlat(record["lonlat"]) for record in located]

    def progress(done, total):
        if done % 50 == 0 or done == total:
            print("  downloaded %d / %d tiles" % (done, total), file=sys.stderr, flush=True)

    print("Sampling %d locations from zoom-%d AWS Terrain Tiles..." % (len(points), args.zoom),
          file=sys.stderr)
    tile_count = tiles.prefetch(points, workers=args.workers, progress=progress)
    print("  %d tiles ready; decoding and sampling" % tile_count, file=sys.stderr, flush=True)

    # Sample in tile order so the small in-memory cache of decoded tiles stays effective.
    def tile_of(record):
        x, y = lonlat_to_global_pixel(*parse_lonlat(record["lonlat"]), args.zoom)
        return int(x // TILE_SIZE), int(y // TILE_SIZE)

    changed = 0
    for record in sorted(located, key=tile_of):
        elevation = round(tiles.elevation(*parse_lonlat(record["lonlat"])))
        if record.get(FIELD) != elevation:
            changed += 1
            record[FIELD] = elevation

    output = "\n".join(dumps(record) if record else "" for record in records)
    print("%d of %d locations %s." % (changed, len(points),
                                       "need updating" if args.check else "updated"), file=sys.stderr)

    if args.check:
        sys.exit(1 if changed else 0)
    if output != original:
        temporary = args.data + ".tmp"
        with open(temporary, "w", encoding="utf-8", newline="") as handle:
            handle.write(output)
        os.replace(temporary, args.data)
        print("Wrote %s" % args.data, file=sys.stderr)
    else:
        print("%s is already up to date." % args.data, file=sys.stderr)


if __name__ == "__main__":
    main()
