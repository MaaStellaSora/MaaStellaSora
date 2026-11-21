"""NumPy preprocessing for Maa's uint8 BGR screenshots and OCR crops."""

from dataclasses import dataclass

import numpy as np


_COLOR_SHIFT = 12
_COLOR_ROUND = 1 << (_COLOR_SHIFT - 1)
_CHANNEL_VALUES = np.arange(1, 256)
_SATURATION_DIVISOR = np.r_[0, np.rint((255 << _COLOR_SHIFT) / _CHANNEL_VALUES)].astype(np.int32)
_HUE_DIVISOR = np.r_[0, np.rint((180 << _COLOR_SHIFT) / (6 * _CHANNEL_VALUES))].astype(np.int32)


def bgr_to_gray(image):
    blue, green, red = image[..., :3].astype(np.int32).transpose(2, 0, 1)
    return ((3735 * blue + 19235 * green + 9798 * red + (1 << 14)) >> 15).astype(np.uint8)


def bgr_to_hsv(image):
    """Return hue in 0–179, saturation and value in 0–255."""
    blue, green, red = image[..., :3].astype(np.int32).transpose(2, 0, 1)
    value = np.maximum(np.maximum(blue, green), red)
    spread = value - np.minimum(np.minimum(blue, green), red)
    saturation = (spread * _SATURATION_DIVISOR[value] + _COLOR_ROUND) >> _COLOR_SHIFT
    hue = np.where(value == red, green - blue,
                   np.where(value == green, blue - red + 2 * spread, red - green + 4 * spread))
    hue = ((hue * _HUE_DIVISOR[spread] + _COLOR_ROUND) >> _COLOR_SHIFT) % 180
    return np.stack((hue, saturation, value), axis=-1).astype(np.uint8)


def gray_to_bgr(image):
    return np.repeat(image[..., None], 3, axis=-1)


def in_range(image, lower, upper):
    matches = (image >= lower) & (image <= upper)
    return matches.all(axis=-1) if image.ndim == 3 else matches


def normalize_gray(gray):
    low, high = int(gray.min()), int(gray.max())
    if high == low:
        return np.zeros_like(gray)
    return np.rint((gray.astype(np.float32) - low) * (255 / (high - low))).astype(np.uint8)


def _sample_positions(start, length, scale):
    return ((np.arange(start, start + length) + .5) / scale - .5).astype(np.float32)


def _resize_rows(image, length, scale, cubic, *, start=0, source_start=0):
    source_length = image.shape[0]
    if scale == 1:
        first = start - source_start
        return image[first:first + length]

    image = np.ascontiguousarray(image)
    positions = _sample_positions(start, length, scale)
    first = np.floor(positions).astype(np.intp)
    fraction = positions - first.astype(np.float32)
    if cubic:
        # Bicubic convolution with a = -0.75 and replicated edge pixels.
        rest = 1 - fraction
        weights = (
            -.75 * fraction * rest**2,
            1 + fraction**2 * (1.25 * fraction - 2.25),
            1 + rest**2 * (1.25 * rest - 2.25),
            -.75 * rest * fraction**2,
        )
        offsets = (-1, 0, 1, 2)
    else:
        weights = (1 - fraction, fraction)
        offsets = (0, 1)

    shape = (length,) + (1,) * (image.ndim - 1)
    result = None
    for offset, weight in zip(offsets, weights):
        indices = np.clip(first + offset - source_start, 0, source_length - 1)
        term = image[indices].astype(np.float32, copy=False)
        term *= weight.reshape(shape)
        if result is None:
            result = term
        else:
            result += term
    return result


def _resize_uint8(resized):
    np.rint(resized, out=resized)
    np.clip(resized, 0, 255, out=resized)
    return resized.astype(np.uint8, order="C")


def _resize(image, size, scale, cubic):
    height, width = image.shape[:2]
    if size is None:
        target_width, target_height = round(width * scale), round(height * scale)
        scale_x = scale_y = scale
    else:
        target_width, target_height = size
        scale_x, scale_y = target_width / width, target_height / height

    if scale_x == scale_y == 1:
        return np.ascontiguousarray(image)
    resized = _resize_rows(image.swapaxes(0, 1), target_width, scale_x, cubic).swapaxes(0, 1)
    resized = _resize_rows(resized, target_height, scale_y, cubic)
    return _resize_uint8(resized)


def resize_cubic(image, size=None, *, scale=None):
    return _resize(image, size, scale, cubic=True)


def resize_cubic_for_roi(image, roi, *, scale):
    """Keep the scaled frame geometry for Maa and compute only the matching ROI."""
    if scale == 1:
        return image

    height, width = image.shape[:2]
    target_width, target_height = round(width * scale), round(height * scale)
    x, y, roi_width, roi_height = roi
    right, bottom = x + roi_width, y + roi_height
    if not (0 <= x < right <= target_width and 0 <= y < bottom <= target_height):
        return resize_cubic(image, scale=scale)

    # Sample at the same positions as a full resize, including the neighboring
    # source rows the bicubic kernel uses at either end of the ROI.
    rows = np.floor(_sample_positions(y, roi_height, scale)).astype(np.intp)
    source_top = max(0, rows[0] - 1)
    source_bottom = min(height, rows[-1] + 3)
    source = image[source_top:source_bottom]
    resized = _resize_rows(source.swapaxes(0, 1), roi_width, scale, cubic=True, start=x).swapaxes(0, 1)
    resized = _resize_rows(resized, roi_height, scale, cubic=True, start=y, source_start=source_top)

    frame = np.zeros((target_height, target_width, *image.shape[2:]), dtype=np.uint8)
    frame[y:bottom, x:right] = _resize_uint8(resized)
    return frame


def resize_linear(image, *, scale):
    return _resize(image, None, scale, cubic=False)


@dataclass(frozen=True)
class Component:
    label: int
    left: int
    top: int
    width: int
    height: int
    area: int
    center_x: float
    center_y: float


def connected_components(mask):
    """Label the 8-connected foreground using horizontal runs."""
    labels = np.zeros(mask.shape, dtype=np.int32)
    padded = np.pad(mask.astype(bool, copy=False), ((0, 0), (1, 1)))
    edges = padded[:, 1:] != padded[:, :-1]
    parents, runs, previous = [], [], []

    def root(index):
        while index != parents[index]:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for y, row in enumerate(edges):
        current = []
        previous_start = 0
        for left, right in np.flatnonzero(row).reshape(-1, 2).tolist():
            index = len(parents)
            parents.append(index)
            runs.append((y, left, right))
            current.append((left, right, index))
            # The endpoints also meet diagonally because right is exclusive.
            while previous_start < len(previous) and previous[previous_start][1] < left:
                previous_start += 1
            for previous_index in range(previous_start, len(previous)):
                prev_left, _, prev_label = previous[previous_index]
                if prev_left > right:
                    break
                first, second = root(index), root(prev_label)
                if first != second:
                    parents[max(first, second)] = min(first, second)
        previous = current

    component_stats = {}
    for index, (y, left, right) in enumerate(runs):
        representative = root(index)
        if representative not in component_stats:
            component_stats[representative] = [len(component_stats) + 1, left, y, right, y, 0, 0, 0]
        stat = component_stats[representative]
        label = stat[0]
        length = right - left
        labels[y, left:right] = label
        stat[1] = min(stat[1], left)
        stat[3] = max(stat[3], right)
        stat[4] = y
        stat[5] += length
        stat[6] += (left + right - 1) * length
        stat[7] += y * length

    components = [Component(label, left, top, right - left, bottom - top + 1, area,
                            twice_sum_x / (2 * area), sum_y / area)
                  for label, left, top, right, bottom, area, twice_sum_x, sum_y in component_stats.values()]
    return labels, components


def close_mask(mask, size):
    """Fill small holes in a binary mask using a square structuring element."""
    before, after = size // 2, (size - 1) // 2
    for erode in (False, True):
        for axis in (0, 1):
            padding = ((before, after), (0, 0)) if axis == 0 else ((0, 0), (before, after))
            padded = np.pad(mask, padding, constant_values=erode)
            windows = np.lib.stride_tricks.sliding_window_view(padded, size, axis=axis)
            mask = windows.all(axis=-1) if erode else windows.any(axis=-1)
    return mask


def match_vertical_strip(image, strip):
    """Return the highest zero-mean normalized correlation and its row."""
    height = strip.shape[0]
    centered = strip.astype(np.float64) - strip.mean()
    norm_squared = np.square(centered).sum()
    if norm_squared == 0:
        return 1.0, 0

    # Correlate each column independently, then sum before the inverse FFT.
    rows = image.shape[0] - height + 1
    fft_size = 1 << (image.shape[0] + height - 2).bit_length()
    samples = np.fft.rfft(image, n=fft_size, axis=0)
    template = np.fft.rfft(centered, n=fft_size, axis=0)
    numerator = np.fft.irfft((samples * template.conj()).sum(axis=1), n=fft_size)[:rows]

    row_sums = np.r_[0, image.sum(axis=1, dtype=np.float64)].cumsum()
    row_squares = np.r_[0, np.square(image, dtype=np.float64).sum(axis=1)].cumsum()
    sums = row_sums[height:] - row_sums[:-height]
    variances = row_squares[height:] - row_squares[:-height] - sums**2 / strip.size
    denominator = np.sqrt(norm_squared * np.maximum(variances, 0))
    scores = np.divide(numerator, denominator, out=np.zeros(rows), where=denominator > 0)
    np.clip(scores, -1.0, 1.0, out=scores)
    y = int(scores.argmax())
    return float(scores[y]), y


def laplacian_variance(gray):
    padded = np.pad(gray.astype(np.int16), 1, mode="reflect")
    laplacian = padded[1:-1, :-2] + padded[1:-1, 2:] + padded[:-2, 1:-1] + padded[2:, 1:-1]
    return float((laplacian - 4 * padded[1:-1, 1:-1]).var(dtype=np.float32))
