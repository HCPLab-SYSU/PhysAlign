"""Deterministic full-frame locator views. One neutral alias per attached view."""
from pathlib import Path
import PIL
from PIL import Image, ImageDraw, ImageFont
from reference_checks import canonical_bytes, require
from reference_geometry import visual_representations, source_points, affine_points
from reference_integrity import active_origin, origin_fingerprint
from io_utils import sha_bytes, safe_path


def asset_record(path, root):
    path = Path(path)
    with Image.open(path) as img:
        img.load()
        require(getattr(img, 'n_frames', 1) == 1, 'MULTIFRAME_IMAGE_UNSUPPORTED')
        require(img.getexif().get(274, 1) in (None, 1), 'EXIF_ORIENTATION_REQUIRES_EXPLICIT_REPAIR')
        w, h = img.size
        pixels = canonical_bytes({'mode': img.mode, 'size': [w, h]}) + b'\0' + img.tobytes()
        return {'relative_path': path.relative_to(root).as_posix(), 'width': w, 'height': h,
                'mode': img.mode, 'bytes_sha256': sha_bytes(path.read_bytes()),
                'pixels_sha256': sha_bytes(pixels), 'orientation_policy': 'stored_pixels'}


def render(env, view, relative_dir, max_views=128):
    root = Path(env['asset_root'])
    reps = visual_representations(view)
    require(len(reps) <= max_views, 'DISPLAY_BUDGET_EXCEEDED', str(len(reps)))
    outdir = safe_path(root, relative_dir)
    outdir.mkdir(parents=True, exist_ok=False)
    displays = []
    for n, (alias, index, loc) in enumerate(reps, 1):
        iid = loc['image_id']
        with Image.open(safe_path(root, env['assets'][iid]['relative_path'])) as original:
            original.load()
            pad = 32
            canvas = Image.new('RGB', (original.width + 2*pad, original.height + 2*pad), 'white')
            # Preserve original stored pixels and alpha appearance; never crop by gold.
            rgba = original.convert('RGBA')
            canvas.paste(rgba, (pad, pad), rgba)
        matrix = [1, 0, pad, 0, 1, pad]
        points = affine_points(source_points(loc, env['assets'][iid]), matrix)
        draw = ImageDraw.Draw(canvas)
        seq = [tuple(p) for p in points]
        if len(seq) == 1:
            x, y = seq[0]
            draw.ellipse((x-4, y-4, x+4, y+4), outline='#1254ce', width=2)
        else:
            closed = loc['geometry']['type'] in ('bbox', 'circle', 'polygon')
            draw.line(seq + [seq[0]] if closed else seq, fill='#1254ce', width=2)
        draw.text((8, 8), alias, font=ImageFont.load_default(size=16), fill='#1254ce')
        outid = f'V{n}'
        path = outdir / (outid + '.png')
        canvas.save(path, format='PNG')
        env['assets'][outid] = asset_record(path, root)
        displays.append({'view_id': outid, 'source_image_id': iid,
                         'source_version': env['assets'][iid]['bytes_sha256'], 'output_asset_id': outid,
                         'source_to_display': matrix, 'marks': [{'alias': alias, 'locator_index': index,
                         'origin_fingerprint': origin_fingerprint(active_origin(loc, view['context'], env['assets'])),
                         'points_px': points}]})
    env['displays'] = displays
    env['renderer_profile'] = {'id': 'full_frame_single_locator_bbox_v1', 'pillow': PIL.__version__,
                               'padding_px': 32, 'scale': 1, 'max_views': max_views,
                               'geometry_policy': 'explicit_source_bbox_only_no_keypoint_order_inference'}
