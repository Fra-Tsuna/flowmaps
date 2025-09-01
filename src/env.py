import numpy as np
import random
import math
import copy
import re

from src.utils.lookup import FURNITURE_TYPES, OBJECT_TYPES

import io
from PIL import Image
import cairosvg
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class FlowSimEnvironment:
    """
    Map with randomly placed tables and objects placed on them,
    with cycling of object positions across tables of the same category.
    Object size remains constant across relocations.
    Includes bounds checking for all bounding boxes, preserving explicit int() casts.
    """

    # Furniture size scales relative to map dimensions
    TABLE_MIN_SCALE = 0.1   # minimum table dimension as fraction of map size
    TABLE_MAX_SCALE = 0.25  # maximum table dimension as fraction of map size

    # Object sizing
    OBJECT_SCALE = 0.4      # initial object size as fraction of table dimension
    OBJECT_MIN_SIZE = 5     # minimum object size in pixels

    def __init__(self, height, width, max_tables=10, min_each=3, stochastic=False, dtype=np.uint8):
        self.height = height
        self.width = width
        self.max_tables = max_tables
        self.min_each = min_each
        self.stochastic = stochastic
        # initialize map
        self.map = np.ones((3, height, width), dtype=dtype) * 255
        # static furniture metadata
        self.furniture_info = []  # list of {id, label, color, bbox}
        # dynamic object metadata
        self.object_info = []     # list of {id, label, shape, color, size, bbox, on_furniture_id}

        # place tables
        self._init_tables(self.min_each)
        remaining = self.max_tables - len(self.furniture_info)
        if remaining > 0:
            self._place_random_furniture(remaining)
        # save furniture-only background
        self.base_map = self.map.copy()

        # place objects initially
        self._place_objects_on_furniture()
        # save initial object info
        self.initial_object_info = copy.deepcopy(self.object_info)

        # pre-compute global ordering per object type for timestamp lookup
        self._compute_furniture_orders()

    def _clamp_bbox(self, bbox):
        """Clamp bbox to be within map bounds with explicit int() casts."""
        y0, x0, y1, x1 = bbox
        y0 = int(max(0, min(self.height-1, y0)))
        x0 = int(max(0, min(self.width-1, x0)))
        y1 = int(max(0, min(self.height-1, y1)))
        x1 = int(max(0, min(self.width-1, x1)))
        assert y0 < y1 and x0 < x1, f"Invalid bbox after clamping: {(y0, x0, y1, x1)}"
        return (y0, x0, y1, x1)

    def _init_tables(self, count):
        placed = []
        fid = 0
        for label, color in FURNITURE_TYPES.items():
            for _ in range(count):
                fh = random.randint(int(self.TABLE_MIN_SCALE * self.height), int(self.TABLE_MAX_SCALE * self.height))
                fw = random.randint(int(self.TABLE_MIN_SCALE * self.width),  int(self.TABLE_MAX_SCALE * self.width))
                for _ in range(100):
                    y0 = random.randint(0, self.height - fh)
                    x0 = random.randint(0, self.width - fw)
                    bbox = (y0, x0, y0+fh, x0+fw)
                    bbox = self._clamp_bbox(bbox)
                    if not self._overlaps_any(bbox, placed):
                        self.map[:, bbox[0]:bbox[2], bbox[1]:bbox[3]] = np.array(color)[:, None, None]
                        placed.append(bbox)
                        self.furniture_info.append({"id": fid, "label": label, "color": color, "bbox": bbox})
                        fid += 1
                        break

    def _place_random_furniture(self, count):
        placed = [f['bbox'] for f in self.furniture_info]
        start_id = len(self.furniture_info)
        for fid in range(start_id, start_id + count):
            label = random.choice(list(FURNITURE_TYPES.keys()))
            color = FURNITURE_TYPES[label]
            fh = random.randint(int(self.TABLE_MIN_SCALE * self.height), int(self.TABLE_MAX_SCALE * self.height))
            fw = random.randint(int(self.TABLE_MIN_SCALE * self.width),  int(self.TABLE_MAX_SCALE * self.width))
            for _ in range(100):
                y0 = random.randint(0, self.height - fh)
                x0 = random.randint(0, self.width - fw)
                bbox = (y0, x0, y0+fh, x0+fw)
                bbox = self._clamp_bbox(bbox)
                if not self._overlaps_any(bbox, placed):
                    self.map[:, bbox[0]:bbox[2], bbox[1]:bbox[3]] = np.array(color)[:, None, None]
                    placed.append(bbox)
                    self.furniture_info.append({"id": fid, "label": label, "color": color, "bbox": bbox})
                    break

    def _place_objects_on_furniture(self):
        oid = 0
        for obj_label, props in OBJECT_TYPES.items():
            candidates = [f for f in self.furniture_info if f['label'] == props['furniture']]
            if not candidates:
                continue

            def angle_of(f):
                y0, x0, y1, x1 = f['bbox']
                cy, cx = (y0+y1)/2.0, (x0+x1)/2.0
                dy, dx = cy - (self.height/2.0), cx - (self.width/2.0)
                return math.atan2(dx, -dy) % (2*math.pi)

            # choose starting furniture by shape
            if props['shape'] == 'circle':
                # circle: start at the highest angle for CCW stepping
                f = max(candidates, key=angle_of)
            elif props['shape'] == 'triangle':
                # triangle: start at the lowest y-center
                f = min(candidates, key=lambda f: ((f['bbox'][0] + f['bbox'][2]) / 2.0))
            elif props['shape'] == 'square':
                # square: start at the lowest angle for CW stepping
                f = min(candidates, key=angle_of)
            else:
                raise ValueError(f"Unknown object shape: {props['shape']}")

            y0, x0, y1, x1 = f['bbox']
            cy, cx = y0 + (y1 - y0) // 2, x0 + (x1 - x0) // 2
            raw = int(min(y1 - y0, x1 - x0) * self.OBJECT_SCALE)
            size = max(self.OBJECT_MIN_SIZE, raw)
            color = props['color']
            shape = props['shape']

            # draw shape and compute raw bbox
            if shape == 'square':
                half = size // 2
                yy0, yy1 = cy - half, cy + half
                xx0, xx1 = cx - half, cx + half
                bbox = self._clamp_bbox((yy0, xx0, yy1, xx1))
                self.map[:, bbox[0]:bbox[2], bbox[1]:bbox[3]] = np.array(color)[:, None, None]

            elif shape == 'circle':
                yy, xx = np.ogrid[:self.height, :self.width]
                mask = (yy - cy)**2 + (xx - cx)**2 <= (size//2)**2
                ys, xs = np.where(mask)
                if ys.size and xs.size:
                    raw_bbox = (ys.min(), xs.min(), ys.max()+1, xs.max()+1)
                else:
                    raw_bbox = (cy-size//2, cx-size//2, cy+size//2, cx+size//2)
                bbox = self._clamp_bbox(raw_bbox)
                for c in range(3): self.map[c][mask] = color[c]

            elif shape == 'triangle':
                p_top = (cx, cy - size//2)
                p_left = (cx - size//2, cy + size//2)
                p_right = (cx + size//2, cy + size//2)
                yy, xx = np.meshgrid(np.arange(self.height), np.arange(self.width), indexing='ij')
                def area(pa, pb, pc):
                    return abs((pa[0]*(pb[1]-pc[1]) + pb[0]*(pc[1]-pa[1]) + pc[0]*(pa[1]-pb[1]))/2.0)
                A = area(p_top, p_left, p_right)
                mask = np.zeros((self.height, self.width), dtype=bool)
                for i in range(self.height):
                    for j in range(self.width):
                        P = (j, i)
                        if abs((area(P, p_left, p_right)
                                + area(p_top, P, p_right)
                                + area(p_top, p_left, P)) - A) < 1e-3:
                            mask[i, j] = True
                ys, xs = np.where(mask)
                if ys.size and xs.size:
                    raw_bbox = (ys.min(), xs.min(), ys.max()+1, xs.max()+1)
                else:
                    raw_bbox = (cy, cx, cy+1, cx+1)
                bbox = self._clamp_bbox(raw_bbox)
                for c in range(3): self.map[c][mask] = color[c]

            else:
                raise ValueError(f"Unknown object shape: {shape}")

            self.object_info.append({
                "id": oid,
                "label": obj_label,
                "shape": shape,
                "color": color,
                "size": size,
                "bbox": bbox,
                "on_furniture_id": f['id']
            })
            oid += 1

    def _compute_furniture_orders(self):
        """
        Pre-compute, for each object type, the global ordering of furniture IDs
        so that get_positions_at_timestamp can fetch efficiently.
        """
        self.furniture_orders = {}
        cy0, cx0 = self.height/2.0, self.width/2.0

        for obj_label, props in OBJECT_TYPES.items():
            furns = [f for f in self.furniture_info if f['label'] == props['furniture']]

            def angle_of(f):
                y0, x0, y1, x1 = f['bbox']
                cy, cx = (y0+y1)/2.0, (x0+x1)/2.0
                dy, dx = cy - cy0, cx - cx0
                return math.atan2(dx, -dy) % (2*math.pi)

            if props['shape'] == 'circle':
                # reverse angle-sorted for forward CCW stepping
                ordered = sorted(furns, key=angle_of)[::-1]
            elif props['shape'] == 'triangle':
                # ascending y-center
                ordered = sorted(furns, key=lambda f: ((f['bbox'][0] + f['bbox'][2]) / 2.0))
            elif props['shape'] == 'square':
                # ascending angle for CW stepping
                ordered = sorted(furns, key=angle_of)
            else:
                raise ValueError(f"Unknown object shape: {props['shape']}")

            # store only the list of furniture dicts
            self.furniture_orders[obj_label] = ordered

    def get_positions_at_timestamp(self, timestamp):
        """
        Compute object positions at a given timestamp
        """
        results = []
        for init_obj in self.initial_object_info:
            obj_label = init_obj['label']
            ordered = self.furniture_orders[obj_label]
            ids = [f['id'] for f in ordered]
            idx0 = ids.index(init_obj['on_furniture_id'])
            
            new_idx = idx0 + timestamp
            if self.stochastic:
                p = random.random()
                if p < 0.1: # 10% chance of staying on the same furniture
                    new_idx -= 1
                elif p < 0.2 + 0.1: # 20% chance of skipping one furniture
                    new_idx += 1
                else: # 70% chance of moving to the next furniture
                    new_idx += 0
            new_f = ordered[new_idx % len(ordered)]

            # compute new bbox exactly as in placement logic
            y0, x0, y1, x1 = new_f['bbox']
            cy, cx = y0 + (y1 - y0)//2, x0 + (x1 - x0)//2
            size = init_obj['size']
            shape = init_obj['shape']
            color = init_obj['color']

            if shape == 'square':
                half = size // 2
                raw_bb = (cy-half, cx-half, cy+half, cx+half)

            elif shape == 'circle':
                yy, xx = np.ogrid[:self.height, :self.width]
                mask = (yy-cy)**2 + (xx-cx)**2 <= (size//2)**2
                ys, xs = np.where(mask)
                if ys.size and xs.size:
                    raw_bb = (ys.min(), xs.min(), ys.max()+1, xs.max()+1)
                else:
                    raw_bb = (cy-size//2, cx-size//2, cy+size//2, cx+size//2)

            elif shape == 'triangle':
                p_top = (cx, cy - size//2)
                p_left = (cx - size//2, cy + size//2)
                p_right = (cx + size//2, cy + size//2)
                yy, xx = np.meshgrid(np.arange(self.height), np.arange(self.width), indexing='ij')
                def area(pa,pb,pc):
                    return abs((pa[0]*(pb[1]-pc[1]) + pb[0]*(pc[1]-pa[1]) + pc[0]*(pa[1]-pb[1]))/2.0)
                A = area(p_top, p_left, p_right)
                mask = np.zeros((self.height, self.width), dtype=bool)
                for i in range(self.height):
                    for j in range(self.width):
                        P = (j, i)
                        if abs((area(P, p_left, p_right)
                                + area(p_top, P, p_right)
                                + area(p_top, p_left, P)) - A) < 1e-3:
                            mask[i, j] = True
                ys, xs = np.where(mask)
                if ys.size and xs.size:
                    raw_bb = (ys.min(), xs.min(), ys.max()+1, xs.max()+1)
                else:
                    raw_bb = (cy, cx, cy+1, cx+1)

            else:
                raise ValueError(f"Unknown object shape: {shape}")

            bbox = self._clamp_bbox(raw_bb)
            results.append({
                "id": init_obj['id'],
                "label": obj_label,
                "shape": shape,
                "color": color,
                "size": size,
                "bbox": tuple(int(v) for v in bbox),
                "on_furniture_id": new_f['id']
            })

        return results

    def query(self, timestamp):
        """
        Query the environment for object positions at a single timestamp,
        returning (image, furniture_info, object_list) only at timestamp.
        """
        
        # compute object positions
        objs = self.get_positions_at_timestamp(timestamp)
        
        # render map at timestamp
        img = self.base_map.copy()
        for obj in objs:
            # find furniture bbox
            f = next(f for f in self.furniture_info if f['id'] == obj['on_furniture_id'])
            y0, x0, y1, x1 = f['bbox']
            cy, cx = y0 + (y1 - y0)//2, x0 + (x1 - x0)//2
            size = obj['size']
            color = obj['color']
            shape = obj['shape']

            if shape == 'square':
                half = size // 2
                raw_bb = (cy-half, cx-half, cy+half, cx+half)
                bb = self._clamp_bbox(raw_bb)
                img[:, bb[0]:bb[2], bb[1]:bb[3]] = np.array(color)[:, None, None]

            elif shape == 'circle':
                yy, xx = np.ogrid[:self.height, :self.width]
                mask = (yy-cy)**2 + (xx-cx)**2 <= (size//2)**2
                for c in range(3):
                    img[c][mask] = color[c]
                ys, xs = np.where(mask)
                if ys.size and xs.size:
                    obj['bbox'] = tuple(int(v) for v in self._clamp_bbox((ys.min(), xs.min(), ys.max()+1, xs.max()+1)))
                else:
                    obj['bbox'] = tuple(int(v) for v in self._clamp_bbox((cy-size//2, cx-size//2, cy+size//2, cx+size//2)))

            elif shape == 'triangle':
                p_top = (cx, cy - size//2)
                p_left = (cx - size//2, cy + size//2)
                p_right = (cx + size//2, cy + size//2)
                yy, xx = np.meshgrid(np.arange(self.height), np.arange(self.width), indexing='ij')
                def area(pa,pb,pc):
                    return abs((pa[0]*(pb[1]-pc[1]) + pb[0]*(pc[1]-pa[1]) + pc[0]*(pa[1]-pb[1]))/2.0)
                A = area(p_top, p_left, p_right)
                mask = np.zeros((self.height, self.width), dtype=bool)
                for i in range(self.height):
                    for j in range(self.width):
                        P = (j, i)
                        if abs((area(P, p_left, p_right)
                                + area(p_top, P, p_right)
                                + area(p_top, p_left, P)) - A) < 1e-3:
                            mask[i, j] = True
                for c in range(3):
                    img[c][mask] = color[c]
                ys, xs = np.where(mask)
                if ys.size and xs.size:
                    obj['bbox'] = tuple(int(v) for v in self._clamp_bbox((ys.min(), xs.min(), ys.max()+1, xs.max()+1)))
                else:
                    obj['bbox'] = tuple(int(v) for v in self._clamp_bbox((cy, cx, cy+1, cx+1)))

        # attach timestamp and return info
        for o in objs:
            o["timestamp"] = timestamp
        return img, list(self.furniture_info), objs

    def _get_rendered_canvas(self, objs, icon_dir, display_scale):
        """
        Builds the RGBA canvas for a given timestamp, composited from SVGs
        that are each rasterized to exactly the bbox size (w*h) by disabling
        aspect-ratio preservation in the SVG itself.
        """
        H, W = self.height, self.width
        H2, W2 = H * display_scale, W * display_scale
        canvas = np.zeros((H2, W2, 4), dtype=np.float32)

        def paste_svg_stretch(svg_path, top, left, box_w, box_h):
            # 1) Read the raw SVG
            with open(svg_path, 'r', encoding='utf-8') as f:
                svg_text = f.read()

            # 2) Inject preserveAspectRatio="none" into the <svg> tag
            svg_text = re.sub(
                r'<svg(\s)',
                r'<svg preserveAspectRatio="none"\1',
                svg_text,
                count=1
            )

            # 3) Rasterize from the altered SVG string at the exact box size
            png_bytes = cairosvg.svg2png(
                bytestring=svg_text.encode('utf-8'),
                output_width=int(box_w),
                output_height=int(box_h)
            )

            # 4) Alpha-composite onto our float canvas
            img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
            arr = np.array(img).astype(np.float32) / 255.0  # (h,w,4)
            h, w, _ = arr.shape
            dst = canvas[top:top+h, left:left+w]
            rgb, alpha = arr[..., :3], arr[..., 3:4]
            canvas[top:top+h, left:left+w, :3] = rgb*alpha + dst[..., :3]*(1-alpha)
            canvas[top:top+h, left:left+w, 3:4] = alpha + dst[..., 3:4]*(1-alpha)

        # Furniture layer
        for f in self.furniture_info:
            y0, x0, y1, x1 = f['bbox']
            bw = (x1 - x0) * display_scale
            bh = (y1 - y0) * display_scale
            paste_svg_stretch(
                f"{icon_dir}/{f['label']}.svg",
                int(y0 * display_scale),
                int(x0 * display_scale),
                bw, bh
            )

        # Object layer
        for obj in objs:
            y0, x0, y1, x1 = obj['bbox']
            bw = (x1 - x0) * display_scale
            bh = (y1 - y0) * display_scale
            paste_svg_stretch(
                f"{icon_dir}/{obj['label']}.svg",
                int(y0 * display_scale),
                int(x0 * display_scale),
                bw, bh
            )

        return (canvas * 255).clip(0,255).astype(np.uint8)
    
    def cycle(self, n, icon_dir=None, display_scale=4):
        """
        Returns (images, furniture_info, timeline).
        If icon_dir is provided, images are RGBA high-res SVG composites.
        Otherwise, images are the original 3x256x256 maps.
        """
        images = []
        timeline = []

        for t in range(n):
            img, _, objs = self.query(t)

            # Store image (raw or rendered)
            if icon_dir:
                canvas = self._get_rendered_canvas(objs, icon_dir, display_scale)
                images.append(canvas)
            else:
                images.append(img)

            # Store a deep copy of the same objs list
            timeline.append([dict(o) for o in objs])

        return images, list(self.furniture_info), timeline
    
    def visualize_cycle(self,
                        n,
                        icon_dir=None,
                        display_scale=4,
                        cols=5,
                        show_labels=True):
        """
        Runs cycle(n, icon_dir, display_scale), then shows each t=0..n-1
        with its image and overlaid bboxes from the returned timeline.

        - icon_dir=None => uses raw 256×256 maps
        - icon_dir=...  => uses high-res SVG renders at `display_scale`
        - cols           => how many columns in the grid
        """
        # 1) get images, furniture_info, and timeline
        images, furn_info, timeline = self.cycle(n, icon_dir=icon_dir, display_scale=display_scale)

        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        axes = axes.flatten()

        for t in range(n):
            ax = axes[t]
            img = images[t]

            # 2) display the image, handling both raw and RGBA
            if icon_dir is None:
                # raw: shape (3,256,256)
                ax.imshow(img.transpose(1,2,0).astype(np.uint8), origin='upper')
                scale = 1
                w_px, h_px = self.width, self.height
            else:
                # rendered: shape (H2,W2,4)
                ax.imshow(img, origin='upper')
                scale = display_scale
                w_px, h_px = self.width*scale, self.height*scale

            ax.set_xlim(0, w_px)
            ax.set_ylim(h_px, 0)
            ax.set_title(f"t={t}")
            
            # 3) overlay boxes from timeline[t]
            for o in timeline[t]:
                y0,x0,y1,x1 = o['bbox']
                rect = patches.Rectangle(
                    (x0*scale, y0*scale),
                    (x1-x0)*scale, (y1-y0)*scale,
                    edgecolor='blue', facecolor='none', lw=2
                )
                ax.add_patch(rect)
                if show_labels:
                    ax.text(
                        x0*scale, y1*scale + 3,
                        f"O{o['id']}:{o['label']}",
                        color='blue', fontsize=6, va='bottom'
                    )

            # 4) optionally overlay furniture boxes in red
            for f in furn_info:
                y0,x0,y1,x1 = f['bbox']
                rect = patches.Rectangle(
                    (x0*scale, y0*scale),
                    (x1-x0)*scale, (y1-y0)*scale,
                    edgecolor='red', facecolor='none', lw=2
                )
                ax.add_patch(rect)
                if show_labels:
                    ax.text(
                        x0*scale, y0*scale - 3,
                        f"F{f['id']}:{f['label']}",
                        color='red', fontsize=6, va='top'
                    )

            ax.set_axis_off()

        # turn off any extra axes
        for ax in axes[n:]:
            ax.set_visible(False)

        plt.tight_layout()
        plt.show()
    
    def visualize_furniture_ordering(self):
        """
        Plot the global ordering of all furniture for each object type
        in a single figure, with white sequence labels.
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        # show base map
        ax.imshow(self.base_map.transpose(1,2,0).astype(np.uint8))

        # annotate each furniture for each object type
        for obj_label, ordered in self.furniture_orders.items():
            for seq, f in enumerate(ordered):
                y0, x0, y1, x1 = f['bbox']
                cy, cx = (y0 + y1) / 2.0, (x0 + x1) / 2.0
                ax.text(
                    cx, cy, 
                    f"{obj_label[:3]}#{seq}\nID:{f['id']}",
                    fontsize=12, fontweight='bold',
                    color='white', ha='center', va='center',
                    bbox=dict(facecolor='black', alpha=0.5)
                )

        ax.set_title('Furniture Visit Order per Object Type', color='white', fontsize=16)
        ax.axis('off')
        plt.tight_layout()
        plt.show(block=False)

    def _overlaps_any(self, bbox, boxes):
        y0, x0, y1, x1 = bbox
        for by0, bx0, by1, bx1 in boxes:
            if not (x1 <= bx0 or x0 >= bx1 or y1 <= by0 or y0 >= by1):
                return True
        return False

    def render_with_svgs(self, icon_dir, display_scale=4):
        """
        Display the 256x256 environment using SVG icons.

        icon_dir: path to folder containing '<label>.svg' files
        display_scale: integer factor to upscale 256 (256*scale)
        """

        H, W = self.height, self.width
        H2, W2 = H * display_scale, W * display_scale
        # start transparent float canvas
        canvas = np.zeros((H2, W2, 4), dtype=np.float32)

        def paste_svg_stretch(svg_path, top, left, box_w, box_h):
            # read SVG text
            with open(svg_path, 'r', encoding='utf-8') as f:
                svg_text = f.read()
            # inject preserveAspectRatio="none"
            svg_text = re.sub(
                r'<svg(\s)',
                r'<svg preserveAspectRatio="none"\1',
                svg_text,
                count=1
            )
            # rasterize to exact bbox size
            png = cairosvg.svg2png(
                bytestring=svg_text.encode('utf-8'),
                output_width=int(box_w),
                output_height=int(box_h)
            )
            # composite into canvas
            img = Image.open(io.BytesIO(png)).convert("RGBA")
            arr = np.array(img).astype(np.float32) / 255.0  # (h,w,4)
            h, w, _ = arr.shape
            dst = canvas[top:top+h, left:left+w]
            rgb, alpha = arr[..., :3], arr[..., 3:4]
            canvas[top:top+h, left:left+w, :3] = rgb*alpha + dst[..., :3]*(1-alpha)
            canvas[top:top+h, left:left+w, 3:4] = alpha + dst[..., 3:4]*(1-alpha)

        # furniture layer
        for f in self.furniture_info:
            y0, x0, y1, x1 = f['bbox']
            bw = (x1 - x0) * display_scale
            bh = (y1 - y0) * display_scale
            paste_svg_stretch(
                f"{icon_dir}/{f['label']}.svg",
                int(y0 * display_scale),
                int(x0 * display_scale),
                bw, bh
            )

        # object layer (stretched as well)
        _, _, objs = self.query(0)
        for obj in objs:
            y0, x0, y1, x1 = obj['bbox']
            bw = (x1 - x0) * display_scale
            bh = (y1 - y0) * display_scale
            paste_svg_stretch(
                f"{icon_dir}/{obj['label']}.svg",
                int(y0 * display_scale),
                int(x0 * display_scale),
                bw, bh
            )

        # convert to uint8 for display
        canvas_uint8 = (canvas * 255).clip(0,255).astype(np.uint8)

        # show with axes on
        fig, ax = plt.subplots(figsize=(6, 6), facecolor='none')
        ax.imshow(
            canvas_uint8,
            origin="upper",
            extent=(0, self.width, self.height, 0)
        )
        ax.set_xlim(0, self.width)
        ax.set_ylim(self.height, 0)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.tight_layout()
        plt.show()

    def visualize_bboxes(self,
                             icon_dir,
                             display_scale=4,
                             timestamp=0,
                             show_labels=True):
        """
        Show side-by-side subplots:
          - Left:  raw 256x256 map with bounding boxes
          - Right: SVG-rendered high-res map with bounding boxes

        icon_dir:       path to folder with '<label>.svg' files
        display_scale:  upscaling factor for the rendered view
        timestamp:      which timestep to visualize
        show_labels:    annotate IDs and labels
        """
        # Prepare raw data
        raw_img, furn_info, obj_info = self.query(timestamp)
        raw_rgb = raw_img.transpose(1,2,0).astype(np.uint8)

        # Prepare rendered canvas
        rendered = self._get_rendered_canvas(obj_info, icon_dir, display_scale)

        # Set up subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # --- Left: raw ---
        ax1.imshow(raw_rgb, origin='upper')
        ax1.set_title(f"Raw 256×256 @ t={timestamp}")
        ax1.set_xlim(0, self.width)
        ax1.set_ylim(self.height, 0)
        # draw boxes
        for f in furn_info:
            y0, x0, y1, x1 = f['bbox']
            rect = patches.Rectangle(
                (x0, y0), x1-x0, y1-y0,
                linewidth=2, edgecolor='red', facecolor='none'
            )
            ax1.add_patch(rect)
            if show_labels:
                ax1.text(x0, y0-2, f"F{f['id']}:{f['label']}",
                         color='red', fontsize=8, va='bottom')

        for o in obj_info:
            y0, x0, y1, x1 = o['bbox']
            rect = patches.Rectangle(
                (x0, y0), x1-x0, y1-y0,
                linewidth=1.5, edgecolor='blue', facecolor='none'
            )
            ax1.add_patch(rect)
            if show_labels:
                ax1.text(x0, y1+2, f"O{o['id']}:{o['label']}",
                         color='blue', fontsize=7, va='top')

        # --- Right: rendered ---
        ax2.imshow(rendered, origin='upper')
        ax2.set_title(f"Rendered SVG ×{display_scale} @ t={timestamp}")
        ax2.set_xlim(0, self.width*display_scale)
        ax2.set_ylim(self.height*display_scale, 0)
        # draw boxes scaled
        for f in furn_info:
            y0, x0, y1, x1 = f['bbox']
            rect = patches.Rectangle(
                (x0*display_scale, y0*display_scale),
                (x1-x0)*display_scale, (y1-y0)*display_scale,
                linewidth=2, edgecolor='red', facecolor='none'
            )
            ax2.add_patch(rect)
            if show_labels:
                ax2.text(x0*display_scale, y0*display_scale-5,
                         f"F{f['id']}:{f['label']}",
                         color='red', fontsize=8, va='bottom')

        for o in obj_info:
            y0, x0, y1, x1 = o['bbox']
            rect = patches.Rectangle(
                (x0*display_scale, y0*display_scale),
                (x1-x0)*display_scale, (y1-y0)*display_scale,
                linewidth=1.5, edgecolor='blue', facecolor='none'
            )
            ax2.add_patch(rect)
            if show_labels:
                ax2.text(x0*display_scale, y1*display_scale+5,
                         f"O{o['id']}:{o['label']}",
                         color='blue', fontsize=7, va='top')

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    from collections import Counter, defaultdict

    # Create environment (stochastic mode on for randomness tests)
    env = FlowSimEnvironment(256, 256, max_tables=10, min_each=3, stochastic=True)

    # Compare rendered SVG bounding boxes vs. rendered images:
    env.visualize_cycle(n=10, icon_dir=None)
    env.visualize_cycle(n=10, icon_dir="assets/icons", display_scale=4)

    # Sanity check with the bounding boxes
    env.visualize_bboxes(
        icon_dir="assets/icons",
        display_scale=4,
        timestamp=9,
        show_labels=True,
    )

    # Visualize furniture visit ordering
    env.visualize_furniture_ordering()

    # Cycle for 10 timestep
    imgs, furn, objs_timeline = env.cycle(10, icon_dir="assets/icons", display_scale=4)
    print("Furniture info:", furn)
    print("Objects timeline:", objs_timeline)

    # Display the 10 masp
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(imgs[i].astype(np.uint8))
        ax.set_title(f"Cycle={i}")
        ax.axis("on")
    plt.tight_layout()

    # Render the environment at timestamp 0 using SVG icons
    env.render_with_svgs(icon_dir="assets/icons", display_scale=4)

    # Stochastic behavior test at t=7 over 100 runs
    runs = []
    for _ in range(100):
        _, _, objs = env.query(7)
        runs.append(objs)

    # Count how often each object lands on each furniture
    counts = defaultdict(Counter)
    for obj_list in runs:
        for obj in obj_list:
            counts[obj['label']][obj['on_furniture_id']] += 1

    # Print normalized counts
    for label, ctr in counts.items():
        normalized = {fid: round(cnt / 100.0, 3) for fid, cnt in ctr.items()}
        print(f"Object '{label}' visit distribution at t=7:", normalized)

    # Block until all figures are closed
    plt.show(block=True)
