"""Offline tests for terrain_tiles.py. Run from the repository root:

    python3 -m unittest discover -s scripts -v

Set TERRAIN_TILES_ONLINE=1 to also run a smoke test against the real AWS bucket.
"""

import os
import random
import shutil
import struct
import tempfile
import unittest
import zlib

import terrain_tiles as tt


# --------------------------------------------------------------------------
# A tiny PNG encoder, so the tests need no image library and no network
# --------------------------------------------------------------------------

def _chunk(ctype, body):
    return (struct.pack(">I", len(body)) + ctype + body
            + struct.pack(">I", zlib.crc32(ctype + body) & 0xFFFFFFFF))


def _paeth(a, b, c):
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    return b if pb <= pc else c


def encode_png(pixels, filter_types, channels=3):
    """Encode rows of (r, g, b[, a]) tuples; filter_types[y] picks each row's filter."""
    height, width = len(pixels), len(pixels[0])
    stride = width * channels
    out = bytearray()
    prev = bytearray(stride)
    for y, row in enumerate(pixels):
        line = bytearray(v for pixel in row for v in pixel)
        f = filter_types[y % len(filter_types)]
        filtered = bytearray(stride)
        for i in range(stride):
            a = line[i - channels] if i >= channels else 0
            b = prev[i]
            c = prev[i - channels] if i >= channels else 0
            predictor = {0: 0, 1: a, 2: b, 3: (a + b) >> 1, 4: _paeth(a, b, c)}[f]
            filtered[i] = (line[i] - predictor) & 255
        out.append(f)
        out += filtered
        prev = line
    header = struct.pack(">IIBBBBB", width, height, 8, 2 if channels == 3 else 6, 0, 0, 0)
    return (tt.PNG_SIGNATURE + _chunk(b"IHDR", header)
            + _chunk(b"IDAT", zlib.compress(bytes(out))) + _chunk(b"IEND", b""))


def encode_elevation(elevation):
    """Terrarium-encode an elevation (whole meters) as an (r, g, b) pixel."""
    value = int(round((elevation + 32768) * 256))
    return value >> 16 & 255, value >> 8 & 255, value & 255


def tile_png(elevation_at):
    """A 256 x 256 tile whose pixel (x, y) holds `elevation_at(x, y)`."""
    pixels = [[encode_elevation(elevation_at(x, y)) for x in range(256)] for y in range(256)]
    return encode_png(pixels, [0, 1, 2, 3, 4])


# --------------------------------------------------------------------------

class TerrariumTests(unittest.TestCase):
    def test_known_values(self):
        self.assertEqual(tt.terrarium_to_elevation(128, 0, 0), 0)         # sea level
        self.assertEqual(tt.terrarium_to_elevation(128, 10, 128), 10.5)    # B is 1/256 m
        self.assertEqual(tt.terrarium_to_elevation(127, 255, 128), -0.5)
        self.assertEqual(tt.terrarium_to_elevation(125, 0, 0), -768)       # roughly the Dead Sea shore
        self.assertEqual(tt.terrarium_to_elevation(0, 0, 0), -32768)

    def test_encoder_round_trip(self):
        for elevation in (-430, -1, 0, 1, 712, 2814, 8848):
            self.assertEqual(tt.terrarium_to_elevation(*encode_elevation(elevation)), elevation)


class TileMathTests(unittest.TestCase):
    def test_origin_and_corners(self):
        self.assertEqual(tt.lonlat_to_global_pixel(0, 0, 0), (128, 128))
        x, y = tt.lonlat_to_global_pixel(-180, tt.MAX_LATITUDE, 3)
        self.assertAlmostEqual(x, 0)
        self.assertAlmostEqual(y, 0, places=6)
        x, y = tt.lonlat_to_global_pixel(180, -tt.MAX_LATITUDE, 3)
        self.assertAlmostEqual(x, 256 * 8)
        self.assertAlmostEqual(y, 256 * 8, places=6)

    def test_jerusalem_tile(self):
        x, y = tt.lonlat_to_global_pixel(35.2137, 31.7683, 12)
        self.assertEqual((int(x // 256), int(y // 256)), (2448, 1666))  # matches the live AWS tile

    def test_latitude_is_clamped(self):
        self.assertEqual(tt.lonlat_to_global_pixel(0, 90, 2),
                         tt.lonlat_to_global_pixel(0, tt.MAX_LATITUDE, 2))
        self.assertEqual(tt.lonlat_to_global_pixel(0, -90, 2),
                         tt.lonlat_to_global_pixel(0, -tt.MAX_LATITUDE, 2))


class PngTests(unittest.TestCase):
    def test_every_filter_type_round_trips(self):
        rng = random.Random(7)
        for channels in (3, 4):
            pixels = [[tuple(rng.randrange(256) for _ in range(channels)) for _ in range(23)]
                      for _ in range(11)]
            for f in range(5):
                with self.subTest(channels=channels, filter=f):
                    image = tt.decode_png(encode_png(pixels, [f], channels))
                    self.assertEqual((image.width, image.height), (23, 11))
                    for y in range(11):
                        for x in range(23):
                            self.assertEqual(image.rgb(x, y), pixels[y][x][:3])

    def test_mixed_filters_in_one_image(self):
        rng = random.Random(3)
        pixels = [[(rng.randrange(256), rng.randrange(256), rng.randrange(256))
                   for _ in range(40)] for _ in range(30)]
        image = tt.decode_png(encode_png(pixels, [0, 1, 2, 3, 4]))
        self.assertTrue(all(image.rgb(x, y) == pixels[y][x] for y in range(30) for x in range(40)))

    def test_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            tt.decode_png(b"not a png at all")
        good = encode_png([[(1, 2, 3)]], [0])
        for cut in (20, len(good) - 20):  # mid-header, and mid-pixel-data
            with self.subTest(cut=cut), self.assertRaises(ValueError):
                tt.decode_png(good[:cut])
        interlaced = bytearray(good)
        interlaced[8 + 8 + 12] = 1  # IHDR interlace byte
        with self.assertRaises(ValueError):
            tt.decode_png(bytes(interlaced))


class SamplingTests(unittest.TestCase):
    ZOOM = 2  # 4 x 4 tiles; small enough to reason about by hand

    def make(self, elevation_at, **kwargs):
        """TerrainTiles whose elevation is a function of the global pixel (gx, gy)."""
        requested = []

        def fetcher(z, x, y):
            requested.append((z, x, y))
            return tile_png(lambda px, py: elevation_at(x * 256 + px, y * 256 + py))

        tiles = tt.TerrainTiles(zoom=self.ZOOM, fetcher=fetcher, **kwargs)
        tiles.requested = requested
        return tiles

    def pixel_center_lonlat(self, gx, gy):
        """The longitude and latitude of the center of global pixel (gx, gy)."""
        import math
        size = 256 * 2 ** self.ZOOM
        lon = (gx + 0.5) / size * 360 - 180
        lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (gy + 0.5) / size))))
        return lon, lat

    def test_pixel_center_returns_that_pixel(self):
        tiles = self.make(lambda gx, gy: (gx * 7 + gy * 3) % 900)
        for gx, gy in ((10, 10), (255, 255), (256, 256), (700, 300)):
            lon, lat = self.pixel_center_lonlat(gx, gy)
            self.assertAlmostEqual(tiles.elevation(lon, lat), (gx * 7 + gy * 3) % 900, places=3)

    def test_interpolates_between_pixels(self):
        tiles = self.make(lambda gx, gy: 100 * gx)  # 100 m per pixel eastward
        lon, lat = self.pixel_center_lonlat(40, 50)
        one_pixel = 360 / (256 * 2 ** self.ZOOM)
        self.assertAlmostEqual(tiles.elevation(lon + 0.25 * one_pixel, lat), 4000 + 25, places=2)
        self.assertAlmostEqual(tiles.elevation(lon + 0.5 * one_pixel, lat), 4050, places=2)

    def test_interpolates_across_tile_boundary(self):
        tiles = self.make(lambda gx, gy: 10 * gx)
        lon, lat = self.pixel_center_lonlat(255, 100)   # last pixel of tile 0
        one_pixel = 360 / (256 * 2 ** self.ZOOM)
        self.assertAlmostEqual(tiles.elevation(lon + 0.5 * one_pixel, lat), 2555, places=2)
        self.assertEqual({(x, y) for _, x, y in tiles.requested}, {(0, 0), (1, 0)})

    def test_wraps_at_antimeridian(self):
        size = 256 * 2 ** self.ZOOM
        tiles = self.make(lambda gx, gy: 1000 if gx == 0 else 0)
        lon, lat = self.pixel_center_lonlat(size - 1, 100)
        one_pixel = 360 / size
        # halfway between the last and the first pixel column of the world
        self.assertAlmostEqual(tiles.elevation(lon + 0.5 * one_pixel, lat), 500, places=2)

    def test_needed_tiles_matches_what_elevation_requests(self):
        points = [(35.2137, 31.7683), (-179.9999, 10), (179.9999, -10), (0, 0), (35.5, 32.9)]
        tiles = self.make(lambda gx, gy: 5)
        expected = tiles.needed_tiles(points)
        for lon, lat in points:
            tiles.elevation(lon, lat)
        self.assertEqual({(x, y) for _, x, y in tiles.requested}, expected)

    def test_each_tile_is_fetched_once_and_cached_on_disk(self):
        cache = tempfile.mkdtemp()
        try:
            first = self.make(lambda gx, gy: gx, cache_dir=cache)
            lon, lat = self.pixel_center_lonlat(300, 300)
            first.elevation(lon, lat)
            first.elevation(lon, lat)
            self.assertEqual(len(first.requested), 1)
            self.assertTrue(os.path.exists(os.path.join(cache, str(self.ZOOM), "1", "1.png")))

            second = self.make(lambda gx, gy: gx, cache_dir=cache)   # a fresh run
            self.assertAlmostEqual(second.elevation(lon, lat), 300, places=3)
            self.assertEqual(second.requested, [])                    # served from disk
        finally:
            shutil.rmtree(cache)

    def test_prefetch_downloads_in_parallel_without_decoding(self):
        points = [self.pixel_center_lonlat(gx, 100) for gx in (10, 300, 600, 900)]
        tiles = self.make(lambda gx, gy: 1)
        count = tiles.prefetch(points, workers=4)
        self.assertEqual(count, len(tiles.requested))
        self.assertEqual(len(tiles.requested), 4)
        for lon, lat in points:
            tiles.elevation(lon, lat)
        self.assertEqual(len(tiles.requested), 4)  # nothing fetched twice

    def test_decoded_tile_cache_is_bounded(self):
        tiles = self.make(lambda gx, gy: 1, decoded_tiles=2)
        for gx in (10, 300, 600, 900):
            tiles.elevation(*self.pixel_center_lonlat(gx, 100))
        self.assertLessEqual(len(tiles._decoded), 2)

    def test_zoom_is_validated(self):
        with self.assertRaises(ValueError):
            tt.TerrainTiles(zoom=16)


@unittest.skipUnless(os.environ.get("TERRAIN_TILES_ONLINE"), "set TERRAIN_TILES_ONLINE=1 to hit AWS")
class OnlineSmokeTest(unittest.TestCase):
    def test_known_places(self):
        tiles = tt.TerrainTiles(zoom=12)
        for name, lon, lat, low, high in (
            ("Jerusalem", 35.2137, 31.7683, 700, 800),
            ("Dead Sea shore", 35.4, 31.5, -440, -380),
            ("Mount Hermon", 35.85, 33.416, 2700, 2830),
        ):
            with self.subTest(name):
                self.assertTrue(low <= tiles.elevation(lon, lat) <= high, name)


if __name__ == "__main__":
    unittest.main()
