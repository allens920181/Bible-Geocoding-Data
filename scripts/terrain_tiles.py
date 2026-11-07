"""Sample elevations from AWS Terrain Tiles (Terrarium encoding).

AWS Terrain Tiles are a public dataset (https://registry.opendata.aws/terrain-tiles/)
of Web Mercator PNG tiles in which each pixel's RGB value encodes elevation:

    elevation_meters = (R * 256 + G + B / 256) - 32768

This module uses only the Python standard library. It converts a longitude and
latitude to a pixel in the tile pyramid, downloads (and optionally caches) the
tile, decodes the PNG, and bilinearly interpolates between the four nearest
pixel centers.

    >>> tiles = TerrainTiles(zoom=12)
    >>> tiles.elevation(35.2137, 31.7683)   # Jerusalem, roughly 750 m
"""

import math
import os
import struct
import time
import urllib.error
import urllib.request
import zlib
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

TILE_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
TILE_SIZE = 256
MAX_ZOOM = 15  # the highest zoom level the dataset provides
DEFAULT_ZOOM = 12  # about 33 m per pixel at 31 degrees north, close to SRTM's 30 m
MAX_LATITUDE = 85.0511287798066  # Web Mercator's northern/southern limit

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


# --------------------------------------------------------------------------
# Terrarium encoding and tile math
# --------------------------------------------------------------------------

def terrarium_to_elevation(r, g, b):
    """Decode one Terrarium pixel to meters above (or below) sea level."""
    return r * 256 + g + b / 256 - 32768


def lonlat_to_global_pixel(lon, lat, zoom):
    """Return the (x, y) position of a point in the zoom level's global pixel grid.

    The grid is `256 * 2**zoom` pixels on each side. Pixel (i, j) covers the
    square [i, i + 1) x [j, j + 1), so its center is at (i + 0.5, j + 0.5).
    """
    size = TILE_SIZE * 2 ** zoom
    lat = max(-MAX_LATITUDE, min(MAX_LATITUDE, lat))
    x = (lon + 180.0) / 360.0 * size
    y = (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * size
    return x, y


# --------------------------------------------------------------------------
# Minimal PNG decoder (8-bit RGB / RGBA, non-interlaced: what AWS serves)
# --------------------------------------------------------------------------

class PngImage:
    """A decoded PNG: `rows[y]` is a bytearray of `width * channels` bytes."""

    def __init__(self, width, height, channels, rows):
        self.width = width
        self.height = height
        self.channels = channels
        self.rows = rows

    def rgb(self, x, y):
        i = x * self.channels
        row = self.rows[y]
        return row[i], row[i + 1], row[i + 2]


def decode_png(data):
    """Decode an 8-bit, non-interlaced RGB or RGBA PNG; raise ValueError otherwise."""
    if data[:8] != PNG_SIGNATURE:
        raise ValueError("not a PNG file")

    header = None
    idat = []
    pos = 8
    while pos + 8 <= len(data):
        length, ctype = struct.unpack(">I4s", data[pos:pos + 8])
        body = data[pos + 8:pos + 8 + length]
        if len(body) != length:
            raise ValueError("truncated PNG")
        if ctype == b"IHDR":
            if length != 13:
                raise ValueError("malformed PNG header")
            header = struct.unpack(">IIBBBBB", body)
        elif ctype == b"IDAT":
            idat.append(body)
        elif ctype == b"IEND":
            break
        pos += length + 12  # length + type + body + CRC
    if header is None or not idat:
        raise ValueError("truncated PNG")

    width, height, bit_depth, color_type, _compression, _filter, interlace = header
    if bit_depth != 8 or color_type not in (2, 6) or interlace != 0:
        raise ValueError(
            "unsupported PNG (bit depth %d, color type %d, interlace %d); "
            "need 8-bit non-interlaced RGB or RGBA" % (bit_depth, color_type, interlace)
        )
    channels = 3 if color_type == 2 else 4

    try:
        raw = zlib.decompress(b"".join(idat))
    except zlib.error as error:
        raise ValueError("corrupt PNG pixel data: %s" % error) from None
    stride = width * channels
    if len(raw) != height * (stride + 1):
        raise ValueError("PNG pixel data has the wrong length")

    rows = []
    prev = bytearray(stride)
    for y in range(height):
        start = y * (stride + 1)
        filter_type = raw[start]
        line = bytearray(raw[start + 1:start + 1 + stride])
        if filter_type == 0:  # None
            pass
        elif filter_type == 1:  # Sub
            for i in range(channels, stride):
                line[i] = (line[i] + line[i - channels]) & 255
        elif filter_type == 2:  # Up
            for i in range(stride):
                line[i] = (line[i] + prev[i]) & 255
        elif filter_type == 3:  # Average
            for i in range(stride):
                left = line[i - channels] if i >= channels else 0
                line[i] = (line[i] + ((left + prev[i]) >> 1)) & 255
        elif filter_type == 4:  # Paeth
            for i in range(stride):
                a = line[i - channels] if i >= channels else 0
                b = prev[i]
                c = prev[i - channels] if i >= channels else 0
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                if pa <= pb and pa <= pc:
                    predictor = a
                elif pb <= pc:
                    predictor = b
                else:
                    predictor = c
                line[i] = (line[i] + predictor) & 255
        else:
            raise ValueError("unknown PNG filter type %d" % filter_type)
        rows.append(line)
        prev = line
    return PngImage(width, height, channels, rows)


# --------------------------------------------------------------------------
# Tile access and elevation sampling
# --------------------------------------------------------------------------

def download_tile(z, x, y, timeout=30, attempts=4):
    """Fetch one tile's PNG bytes, retrying transient failures with backoff."""
    url = TILE_URL.format(z=z, x=x, y=y)
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            if error.code < 500 and error.code != 429:
                raise  # 403/404 and the like won't improve on retry
            failure = error
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            failure = error
        if attempt == attempts - 1:
            raise failure
        time.sleep(2 ** attempt)


class TerrainTiles:
    """Look up elevations at a fixed zoom level.

    `cache_dir`: if given, downloaded PNGs are stored there as `{z}/{x}/{y}.png`
    so later runs need no network. `fetcher(z, x, y) -> bytes` replaces the
    download (useful for tests). `decoded_tiles` bounds how many decoded tiles
    stay in memory (each is roughly 200 KB).
    """

    def __init__(self, zoom=DEFAULT_ZOOM, cache_dir=None, fetcher=None, decoded_tiles=48):
        if not 0 <= zoom <= MAX_ZOOM:
            raise ValueError("zoom must be between 0 and %d" % MAX_ZOOM)
        self.zoom = zoom
        self.cache_dir = cache_dir
        self._fetcher = fetcher or download_tile
        self._decoded_limit = decoded_tiles
        self._decoded = OrderedDict()
        self._raw = {}  # in-memory PNG bytes, used only when there is no cache_dir

    # -- tile loading ------------------------------------------------------

    def _cache_path(self, x, y):
        return os.path.join(self.cache_dir, str(self.zoom), str(x), "%d.png" % y)

    def _png_bytes(self, x, y):
        if self.cache_dir:
            path = self._cache_path(x, y)
            if os.path.exists(path):
                with open(path, "rb") as handle:
                    return handle.read()
            data = self._fetcher(self.zoom, x, y)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            temporary = "%s.%d.tmp" % (path, os.getpid())
            with open(temporary, "wb") as handle:
                handle.write(data)
            os.replace(temporary, path)  # never leave a half-written tile in the cache
            return data
        key = (x, y)
        if key not in self._raw:
            self._raw[key] = self._fetcher(self.zoom, x, y)
        return self._raw[key]

    def _tile(self, x, y):
        key = (x, y)
        tile = self._decoded.get(key)
        if tile is not None:
            self._decoded.move_to_end(key)
            return tile
        tile = decode_png(self._png_bytes(x, y))
        if tile.width != TILE_SIZE or tile.height != TILE_SIZE:
            raise ValueError("tile %d/%d/%d is not %d x %d" % (self.zoom, x, y, TILE_SIZE, TILE_SIZE))
        self._decoded[key] = tile
        if len(self._decoded) > self._decoded_limit:
            self._decoded.popitem(last=False)
        return tile

    # -- sampling ----------------------------------------------------------

    def _pixel_elevation(self, gx, gy):
        """Elevation of the global pixel (gx, gy); x wraps around, y is clamped."""
        size = TILE_SIZE * 2 ** self.zoom
        gx %= size
        gy = max(0, min(size - 1, gy))
        tile = self._tile(gx // TILE_SIZE, gy // TILE_SIZE)
        return terrarium_to_elevation(*tile.rgb(gx % TILE_SIZE, gy % TILE_SIZE))

    def _corner_pixels(self, lon, lat):
        """Return the four neighbouring pixel indices and the bilinear weights."""
        px, py = lonlat_to_global_pixel(lon, lat, self.zoom)
        fx, fy = px - 0.5, py - 0.5  # distance measured between pixel centers
        x0, y0 = math.floor(fx), math.floor(fy)
        return x0, y0, fx - x0, fy - y0

    def elevation(self, lon, lat):
        """Bilinearly interpolated elevation in meters at a longitude and latitude."""
        x0, y0, tx, ty = self._corner_pixels(lon, lat)
        top = (self._pixel_elevation(x0, y0) * (1 - tx)
               + self._pixel_elevation(x0 + 1, y0) * tx)
        bottom = (self._pixel_elevation(x0, y0 + 1) * (1 - tx)
                  + self._pixel_elevation(x0 + 1, y0 + 1) * tx)
        return top * (1 - ty) + bottom * ty

    def needed_tiles(self, points):
        """The set of (x, y) tiles that `elevation` will touch for these (lon, lat) points."""
        size = TILE_SIZE * 2 ** self.zoom
        tiles = set()
        for lon, lat in points:
            x0, y0, _, _ = self._corner_pixels(lon, lat)
            for gx in (x0, x0 + 1):
                for gy in (y0, y0 + 1):
                    tiles.add(((gx % size) // TILE_SIZE,
                               max(0, min(size - 1, gy)) // TILE_SIZE))
        return tiles

    def prefetch(self, points, workers=8, progress=None):
        """Download every tile the points need, in parallel.

        Only the network part runs concurrently; decoding still happens lazily
        in `elevation`. `progress(done, total)` is called as tiles arrive.
        """
        tiles = sorted(self.needed_tiles(points))
        done = 0

        def fetch(tile):
            self._png_bytes(*tile)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            for _ in pool.map(fetch, tiles):
                done += 1
                if progress:
                    progress(done, len(tiles))
        return len(tiles)
