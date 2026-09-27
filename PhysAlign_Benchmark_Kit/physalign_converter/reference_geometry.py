"""Explicit source-pixel -> display-pixel transforms, not alias text matching."""
from __future__ import annotations
import math
from reference_checks import require, finite_number, digest, validate_geometry
from reference_integrity import active_origin, origin_fingerprint


def source_points(locator, asset):
    require(locator['kind']=='visual','VISUAL_LOCATOR_REQUIRED')
    g=locator['geometry'];w,h=asset['width'],asset['height']
    validate_geometry(g,w,h)
    if g['type']=='bbox':
        x1,y1,x2,y2=g['bbox_1000'];pts=[[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
    elif g['type']=='point': pts=[g['xy_1000']]
    elif g['type'] in ('polyline','polygon'): pts=g['points_1000']
    else:
        cx=g['center_1000'][0]*w/1000;cy=g['center_1000'][1]*h/1000
        r=g['radius_1000_min_dim']*min(w,h)/1000
        return [[cx+r*math.cos(k*math.tau/32),cy+r*math.sin(k*math.tau/32)] for k in range(32)]
    return [[x*w/1000,y*h/1000] for x,y in pts]


def affine_points(points, matrix):
    require(isinstance(matrix,list) and len(matrix)==6 and all(finite_number(x) for x in matrix),'TRANSFORM_INVALID')
    a,b,c,d,e,f=matrix
    require(abs(a*e-b*d)>1e-12,'TRANSFORM_SINGULAR')
    return [[a*x+b*y+c,d*x+e*y+f] for x,y in points]


def visual_representations(view):
    reps=[]
    for alias,loc in view['anchors'].items():
        if loc['kind']=='visual': reps.append((alias,0,loc))
    for alias,locs in view['reference_views'].items():
        reps.extend((alias,i,loc) for i,loc in enumerate(locs) if loc['kind']=='visual')
    for candidate in view['candidates']:
        reps.extend((candidate['alias'],i,loc) for i,loc in enumerate(candidate.get('locators',[])) if loc['kind']=='visual')
    return reps


def validate_render_manifest(view, assets, displays, *, pixel_tolerance=1e-6):
    expected={(alias,i):loc for alias,i,loc in visual_representations(view)}
    covered=set();output_ids=set()
    for display in displays:
        require(set(display)=={'view_id','source_image_id','source_version','output_asset_id','source_to_display','marks'},'RENDER_MANIFEST_FIELDS')
        iid=display['source_image_id'];outid=display['output_asset_id']
        require(iid in view['context']['image_ids'] and iid in assets and outid in assets,'RENDER_ASSET_MISSING')
        require(outid not in output_ids and outid not in view['context']['image_ids'],'DUPLICATE_DISPLAY_ASSET')
        output_ids.add(outid)
        require(display['source_version']==assets[iid]['bytes_sha256'],'RENDER_SOURCE_VERSION_MISMATCH')
        require(isinstance(display['marks'],list) and bool(display['marks']),'DISPLAY_WITHOUT_MARKS')
        for mark in display['marks']:
            require(set(mark)=={'alias','locator_index','origin_fingerprint','points_px'},'RENDER_MARK_FIELDS')
            require(type(mark['locator_index']) is int and mark['locator_index']>=0,'RENDER_LOCATOR_INDEX')
            pair=(mark['alias'],mark['locator_index'])
            require(pair in expected and pair not in covered,'RENDER_MARK_REFERENCE_INVALID')
            loc=expected[pair]
            require(loc['image_id']==iid,'RENDER_SOURCE_IMAGE_MISMATCH')
            require(mark['origin_fingerprint']==origin_fingerprint(active_origin(loc,view['context'],assets)),'RENDER_MARK_ORIGIN_MISMATCH')
            points=affine_points(source_points(loc,assets[iid]),display['source_to_display'])
            actual=mark['points_px']
            require(isinstance(actual,list) and len(actual)==len(points) and all(isinstance(p,list) and len(p)==2 for p in actual),'RENDER_POINTS_INVALID')
            for x,y in actual:
                require(finite_number(x) and finite_number(y) and 0<=x<=assets[outid]['width'] and 0<=y<=assets[outid]['height'],'RENDER_POINT_OUTSIDE_VIEW')
            require(all(abs(p[k]-q[k])<=pixel_tolerance for p,q in zip(points,actual) for k in (0,1)),'RENDER_TRANSFORM_MISMATCH')
            covered.add(pair)
    require(covered==set(expected),'RENDER_COVERAGE_MISSING')
    return set(view['context']['image_ids'])|output_ids

