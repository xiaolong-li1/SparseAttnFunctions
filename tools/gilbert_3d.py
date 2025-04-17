#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2018 Jakub Červený

"""
Generalized 3D Hilbert ('Gilbert') curve generator and tensor rearranger.

This module provides:
- gilbert3d: yields discrete 3D coordinates filling a cuboid of given dimensions.
- GilbertRearranger: a PyTorch-based utility to reorder sequence data along a 3D Gilbert curve.

Usage:
    order_gen = GilbertRearranger(width, height, depth, text_length)
    q_r, k_r, v_r = order_gen.rearrange(q, k, v)
    output = model(q_r, k_r, v_r)
    restored = order_gen.reversed_rearrange(output)
"""

import torch


def gilbert3d(width: int, height: int, depth: int):
    """
    Generate a 3D Gilbert (generalized Hilbert) space-filling curve.

    Yields:
        Tuples of (x, y, z) coordinates that traverse a cuboid of size
        (width x height x depth) in a continuous, locality-preserving order.

    Even dimensions are recommended for optimal pathing.
    """
    # Choose the longest axis as the primary direction
    if width >= height and width >= depth:
        yield from _generate3d(0, 0, 0,
                                width, 0, 0,
                                0, height, 0,
                                0, 0, depth)
    elif height >= width and height >= depth:
        yield from _generate3d(0, 0, 0,
                                0, height, 0,
                                width, 0, 0,
                                0, 0, depth)
    else:
        yield from _generate3d(0, 0, 0,
                                0, 0, depth,
                                width, 0, 0,
                                0, height, 0)


def _sgn(x: int) -> int:
    """
    Sign function: returns -1 for negatives, +1 for positives, and 0 for zero.
    """
    return -1 if x < 0 else (1 if x > 0 else 0)


def _generate3d(x: int, y: int, z: int,
                ax: int, ay: int, az: int,
                bx: int, by: int, bz: int,
                cx: int, cy: int, cz: int):
    """
    Recursively generate coordinates for a 3D Gilbert curve.

    Parameters:
        x, y, z: Starting coordinates.
        ax, ay, az: Vector for the 'x' axis steps.
        bx, by, bz: Vector for the 'y' axis steps.
        cx, cy, cz: Vector for the 'z' axis steps.
    """
    # Calculate sizes along each axis
    w = abs(ax + ay + az)
    h = abs(bx + by + bz)
    d = abs(cx + cy + cz)

    # Unit direction vectors
    dax, day, daz = _sgn(ax), _sgn(ay), _sgn(az)
    dbx, dby, dbz = _sgn(bx), _sgn(by), _sgn(bz)
    dcx, dcy, dcz = _sgn(cx), _sgn(cy), _sgn(cz)

    # Base cases: 1D line along each axis
    if h == 1 and d == 1:
        for _ in range(w):
            yield (x, y, z)
            x, y, z = x + dax, y + day, z + daz
        return
    if w == 1 and d == 1:
        for _ in range(h):
            yield (x, y, z)
            x, y, z = x + dbx, y + dby, z + dbz
        return
    if w == 1 and h == 1:
        for _ in range(d):
            yield (x, y, z)
            x, y, z = x + dcx, y + dcy, z + dcz
        return

    # Half steps for recursive splitting
    ax2, ay2, az2 = ax // 2, ay // 2, az // 2
    bx2, by2, bz2 = bx // 2, by // 2, bz // 2
    cx2, cy2, cz2 = cx // 2, cy // 2, cz // 2

    w2, h2, d2 = abs(ax2 + ay2 + az2), abs(bx2 + by2 + bz2), abs(cx2 + cy2 + cz2)

    # Adjust to prefer even sub-steps
    if (w2 % 2) and (w > 2):
        ax2, ay2, az2 = ax2 + dax, ay2 + day, az2 + daz
    if (h2 % 2) and (h > 2):
        bx2, by2, bz2 = bx2 + dbx, by2 + dby, bz2 + dbz
    if (d2 % 2) and (d > 2):
        cx2, cy2, cz2 = cx2 + dcx, cy2 + dcy, cz2 + dcz

    # Recursive splitting cases
    if 2 * w > 3 * h and 2 * w > 3 * d:
        # Split along major axis only
        yield from _generate3d(x, y, z,
                                ax2, ay2, az2,
                                bx, by, bz,
                                cx, cy, cz)
        yield from _generate3d(x + ax2, y + ay2, z + az2,
                                ax - ax2, ay - ay2, az - az2,
                                bx, by, bz,
                                cx, cy, cz)
    elif 3 * h > 4 * d:
        # Do not split depth
        yield from _generate3d(x, y, z,
                                bx2, by2, bz2,
                                cx, cy, cz,
                                ax2, ay2, az2)
        yield from _generate3d(x + bx2, y + by2, z + bz2,
                                ax, ay, az,
                                bx - bx2, by - by2, bz - bz2,
                                cx, cy, cz)
        yield from _generate3d(
            x + (ax - dax) + (bx2 - dbx),
            y + (ay - day) + (by2 - dby),
            z + (az - daz) + (bz2 - dbz),
            -bx2, -by2, -bz2,
            cx, cy, cz,
            -(ax - ax2), -(ay - ay2), -(az - az2)
        )
    elif 3 * d > 4 * h:
        # Do not split height
        yield from _generate3d(x, y, z,
                                cx2, cy2, cz2,
                                ax2, ay2, az2,
                                bx, by, bz)
        yield from _generate3d(x + cx2, y + cy2, z + cz2,
                                ax, ay, az,
                                bx, by, bz,
                                cx - cx2, cy - cy2, cz - cz2)
        yield from _generate3d(
            x + (ax - dax) + (cx2 - dcx),
            y + (ay - day) + (cy2 - dcy),
            z + (az - daz) + (cz2 - dcz),
            -cx2, -cy2, -cz2,
            -(ax - ax2), -(ay - ay2), -(az - az2),
            bx, by, bz
        )
    else:
        # Regular case: split along all axes
        yield from _generate3d(x, y, z,
                                bx2, by2, bz2,
                                cx2, cy2, cz2,
                                ax2, ay2, az2)
        yield from _generate3d(x + bx2, y + by2, z + bz2,
                                cx, cy, cz,
                                ax2, ay2, az2,
                                bx - bx2, by - by2, bz - bz2)
        yield from _generate3d(
            x + (bx2 - dbx) + (cx - dcx),
            y + (by2 - dby) + (cy - dcy),
            z + (bz2 - dbz) + (cz - dcz),
            ax, ay, az,
            -bx2, -by2, -bz2,
            -(cx - cx2), -(cy - cy2), -(cz - cz2)
        )
        yield from _generate3d(
            x + (ax - dax) + bx2 + (cx - dcx),
            y + (ay - day) + by2 + (cy - dcy),
            z + (az - daz) + bz2 + (cz - dcz),
            -cx, -cy, -cz,
            -(ax - ax2), -(ay - ay2), -(az - az2),
            bx - bx2, by - by2, bz - bz2
        )
        yield from _generate3d(
            x + (ax - dax) + (bx2 - dbx),
            y + (ay - day) + (by2 - dby),
            z + (az - daz) + (bz2 - dbz),
            -bx2, -by2, -bz2,
            cx2, cy2, cz2,
            -(ax - ax2), -(ay - ay2), -(az - az2)
        )


class GilbertRearranger:
    """
    A sequence rearranger based on the 3D Gilbert curve.

    This utility reorders the video portion of Q, K, V tensors along the
    Gilbert curve while preserving a specified text prefix.
    """
    def __init__(self,
                 width: int,
                 height: int,
                 depth: int,
                 text_length: int = 224):
        """
        Initialize the rearranger with grid dimensions and text segment length.

        Args:
            width, height, depth: Dimensions of the 3D grid for video frames.
            text_length: Number of initial tokens to keep in original order.
        """
        self.width = width
        self.height = height
        self.depth = depth
        self.total_elements = width * height * depth
        self.text_length = text_length

        coord_to_index = self._gilbert3d_with_index(width,
                                                   height,
                                                   depth)
        # Build mappings between original and Gilbert orders
        orig2gil = [0] * self.total_elements
        gil2orig = [0] * self.total_elements
        for gil_idx, orig_idx in coord_to_index.items():
            orig2gil[orig_idx] = gil_idx
            gil2orig[gil_idx] = orig_idx

        # Move tensors to CUDA for indexing
        self.orig_to_gilbert = torch.tensor(orig2gil,
                                            dtype=torch.long,
                                            device='cuda')
        self.gilbert_to_orig = torch.tensor(gil2orig,
                                            dtype=torch.long,
                                            device='cuda')

    def _gilbert3d_with_index(self,
                              width: int,
                              height: int,
                              depth: int) -> dict:
        """
        Build a mapping from linear indices to Gilbert curve order indices.
        """
        mapping = {}
        index = 0

        def linear_idx(x, y, z):
            return x + width * (y + height * z)

        for x, y, z in gilbert3d(width, height, depth):
            mapping[linear_idx(x, y, z)] = index
            index += 1
        return mapping

    def rearrange(self, q: torch.Tensor,
                  k: torch.Tensor,
                  v: torch.Tensor) -> tuple:
        """
        Rearrange the video segments of Q, K, V tensors along the Gilbert path.

        Args:
            q, k, v: Tensors of shape (..., seq_len, dim).

        Returns:
            A tuple of rearranged (q, k, v) with text prefix appended back.
        """
        seq_dim = -2
        # Split into text and video parts
        txt_q, vid_q = q[..., :self.text_length, :], q[..., self.text_length:, :]
        txt_k, vid_k = k[..., :self.text_length, :], k[..., self.text_length:, :]
        txt_v, vid_v = v[..., :self.text_length, :], v[..., self.text_length:, :]

        # Apply Gilbert ordering to video segments
        qv = vid_q.index_select(seq_dim, self.orig_to_gilbert)
        kv = vid_k.index_select(seq_dim, self.orig_to_gilbert)
        vv = vid_v.index_select(seq_dim, self.orig_to_gilbert)

        # Concatenate reordered video with original text prefix
        return (
            torch.cat((qv, txt_q), dim=seq_dim),
            torch.cat((kv, txt_k), dim=seq_dim),
            torch.cat((vv, txt_v), dim=seq_dim)
        )

    def reversed_rearrange(self, out: torch.Tensor) -> torch.Tensor:
        """
        Restore the original ordering of the video portion in the output tensor.

        Args:
            out: Tensor of shape (..., seq_len, dim), with video first then text.

        Returns:
            Tensor with text prefix first and video restored to linear order.
        """
        seq_dim = -2
        vid_out, txt_out = out[..., :-self.text_length, :], out[..., -self.text_length:, :]
        # Inverse indexing to restore original spatial order
        restored_vid = vid_out.index_select(seq_dim, self.gilbert_to_orig)
        return torch.cat((txt_out, restored_vid), dim=seq_dim)
