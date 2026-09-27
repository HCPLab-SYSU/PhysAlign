"""Appendix E.1: fixed geometry and alias/order controls, independent of outcomes."""

from copy import deepcopy
from fractions import Fraction
from io import BytesIO
from math import sqrt
import random
import re

from .dataset import ImageData, MARKERS, require, sections
from .storage import canonical, digest, fingerprint, loads

RENDERER_VERSION = "original_pixels_single_box_header_v1"


def compose(parts):
    return "\n".join(f"[{name}]\n{parts[name]}" for name in MARKERS)


def original_ids(user):
    section = sections(user)["Images and locator views"]
    match = re.search(r"(?m)^Original image IDs: (.+)$", section)
    require(match is not None, "Original image IDs must be explicitly exported")
    ids = [s.strip() for s in match[1].split(",")]
    require(all(ids) and len(ids) == len(set(ids)), "Invalid original image list")
    return ids


def no_image_input(inp, *, remove_geometry=False):
    out = deepcopy(inp)
    out["attachments"] = []
    if remove_geometry:
        parts = sections(out["messages"]["user"])
        e = loads(parts["Evidence locations and candidates"])
        e["anchors"] = {a: loc if loc["kind"] == "text" else {"kind": "withheld_visual_anchor"}
                        for a, loc in e["anchors"].items()}
        e["candidates"] = [{"alias": c["alias"],
                            "locators": [loc for loc in c["locators"] if loc["kind"] == "text"]}
                           for c in e["candidates"]]
        parts["Evidence locations and candidates"] = canonical(e)
        parts["Images and locator views"] = "Image pixels and visual geometry are withheld in this condition."
        out["messages"]["user"] = compose(parts)
    return out


def permutation(aliases, *, seed, probe_id, index, mode):
    require(mode in {"alias", "order", "both"}, "Unknown permutation mode")
    rng = random.Random(int(fingerprint([seed, probe_id, index, mode]), 16))
    def shuffled(xs):
        result = list(xs)
        rng.shuffle(result)
        if len(result) > 1 and result == list(xs):
            result = result[1:] + result[:1]
        return result
    names = shuffled(aliases) if mode in {"alias", "both"} else list(aliases)
    order = shuffled(range(len(aliases))) if mode in {"order", "both"} else list(range(len(aliases)))
    return dict(zip(aliases, names)), order


def _replace_aliases(text, mapping):
    if not mapping:
        return text
    regex = r"(?<![\w])(?:" + "|".join(re.escape(a) for a in sorted(mapping, key=len, reverse=True)) + r")(?![\w])"
    return re.sub(regex, lambda m: mapping[m[0]], text)


def permute_input(inp, mapping, order):
    out = deepcopy(inp)
    parts = sections(out["messages"]["user"])
    evidence = loads(parts["Evidence locations and candidates"])
    candidates = evidence["candidates"]
    aliases = [c["alias"] for c in candidates]
    require(not set(aliases) & set(evidence["anchors"]), 'Candidate IDs and reading-anchor IDs must be disjoint for permutation controls')
    require(set(mapping) == set(mapping.values()) == set(aliases), "Alias permutation must be a bijection")
    require(sorted(order) == list(range(len(aliases))), "Invalid candidate order permutation")
    evidence["candidates"] = [{**candidates[i], "alias": mapping[candidates[i]["alias"]]} for i in order]
    parts["Evidence locations and candidates"] = canonical(evidence)
    # Original physics names and literal OCR transcriptions are NOT public aliases.
    for key in ("Local question", "Output format", "Images and locator views"):
        parts[key] = _replace_aliases(parts[key], mapping)
    out["messages"]["system"] = _replace_aliases(out["messages"]["system"], mapping)
    out["messages"]["user"] = compose(parts)
    return out


def render_control_input(inp, originals, store_image):
    """Use the SAME renderer for identity reference and all perturbations.

    Never edit a burned-in label on an unknown exported overlay. Rebuild from
    unchanged original pixels, with exactly one locator per view. The original
    main experiment still uses its original exported views.
    """
    from PIL import Image, ImageDraw, ImageFont, __version__ as pillow_version
    out = deepcopy(inp)
    parts = sections(out["messages"]["user"])
    e = loads(parts["Evidence locations and candidates"])
    by_id = {a.asset_id: a for a in originals}
    attachments, legend = [], []
    font = ImageFont.load_default(size=14)
    def view(alias, loc, view_id):
        if loc["kind"] != "visual":
            return
        source = by_id[loc["image_id"]]
        with Image.open(BytesIO(source.data)) as image:
            image.load()
            original = image.convert("RGB")
        header = 32
        canvas = Image.new("RGB", (original.width, original.height + header), "white")
        canvas.paste(original, (0, header))
        draw = ImageDraw.Draw(canvas)
        require(draw.textbbox((8, 6), alias, font=font)[2] < canvas.width, "Alias does not fit the frozen header")
        draw.text((8, 6), alias, fill=(0, 70, 255), font=font)
        b = loc["geometry"]["bbox_1000"]
        box = [b[0] * original.width / 1000, b[1] * original.height / 1000 + header,
               b[2] * original.width / 1000, b[3] * original.height / 1000 + header]
        draw.rectangle(box, outline=(0, 70, 255), width=2)
        buffer = BytesIO()
        canvas.save(buffer, format="PNG")
        data = buffer.getvalue()
        asset = ImageData(view_id, "image/png", canvas.width, canvas.height, digest(data), data)
        attachments.append(store_image(asset))
        legend.append(f"{view_id}: {source.asset_id}; {alias}")
    for i, (alias, loc) in enumerate(e["anchors"].items()):
        view(alias, loc, f"RView{i}")
    for i, c in enumerate(e["candidates"]):
        for j, loc in enumerate(c["locators"]):
            view(c["alias"], loc, f"CView{i}_{j}")
    attachments.extend(store_image(a) for a in originals)
    parts["Images and locator views"] = "\n".join(legend + ["Original image IDs: " + ", ".join(a.asset_id for a in originals)])
    out["messages"]["user"] = compose(parts)
    out["attachments"] = attachments
    return out, {"version": RENDERER_VERSION, "pillow": pillow_version,
                 "header_pixels": 32, "box_width": 2, "font": "Pillow.load_default(size=14)", "resize": False}


def boundary_distance_squared(a, b, width, height):
    """Rectangle-set Euclidean separation: overlapping/touching boxes give zero.

    x/y are converted to original pixel units BEFORE combining them. Squared
    rational distances preserve exact ties, including in non-square images.
    """
    a, b = [Fraction(str(x)) for x in a], [Fraction(str(x)) for x in b]
    dx = max(a[0] - b[2], b[0] - a[2], 0) * width / 1000
    dy = max(a[1] - b[3], b[1] - a[3], 0) * height / 1000
    return dx * dx + dy * dy


def nearest_region(inp, images, declaration):
    """Only public geometry and an a-priori ownership applicability declaration."""
    e = loads(sections(inp["messages"]["user"])["Evidence locations and candidates"])
    anchor = e["anchors"][declaration["anchor_id"]]
    require(anchor["kind"] == "visual", "Nearest-region anchor must be visual")
    image = next(a for a in images if a.asset_id == anchor["image_id"])
    distances = []
    for index, candidate in enumerate(e["candidates"]):
        locs = [loc for loc in candidate["locators"] if loc["kind"] == "visual" and loc["image_id"] == anchor["image_id"]]
        require(bool(locs), "Every nearest-region candidate needs a comparable region on the anchor image")
        distance = min(boundary_distance_squared(anchor["geometry"]["bbox_1000"],
                                                loc["geometry"]["bbox_1000"], image.width, image.height) for loc in locs)
        distances.append((distance, index, candidate["alias"]))
    distance, _, alias = min(distances)
    return {"response": canonical({declaration["field"]: alias}), "selected_alias": alias,
            "distance_pixels": sqrt(float(distance)), "candidate_count": len(distances),
            "all_distances_pixels": {a: sqrt(float(d)) for d, _, a in distances},
            "distance_rule": "bbox_separation_euclidean_original_pixels_overlap_zero",
            "tie_break": "frozen_candidate_index", "uses_private_labels": False}
