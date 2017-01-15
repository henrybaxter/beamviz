import argparse
import sys
import time
import hashlib
import statistics

import numpy as np

from . import egsinp
from . import cgs


def main():
    args = parse_args()
    prev_hsh = None
    while True:
        # watch file hash to avoid causing OpenSCAD to re-render continuously
        hsh = hashlib.sha256(open(args.egsinp).read().encode('utf-8')).hexdigest()
        if hsh == prev_hsh:
            time.sleep(.2)
            continue
        prev_hsh = hsh

        print('Parsing...')
        try:
            render(args.egsinp, args.target_diameter)
        except egsinp.ParseError:
            pass
        except Exception as e:
            print('Could not render: {}'.format(e))

        if not args.watch:
            sys.exit()


def render(path, target_diameter):

    scene = cgs.CGSElement()
    blocks = get_blocks(path)
    collimator_stats(blocks)

    # assume phantom is centered at zfocus of first block
    phantom = cgs.CGSSphere(target_diameter)
    phantom.translate((0, 0, blocks[0]['zfocus']))
    scene.add(phantom)

    collimator = build_collimator(blocks)
    scene.add(collimator)

    output_path = path.replace('.egsinp', '.scad')
    print('Rendering collimator to {} with target size {:.2f} cm'.format(output_path, target_diameter))
    open(output_path, 'w').write(scene.render())
    return output_path


def polygon_area(corners):
    n = len(corners)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]
    area = abs(area) / 2.0
    return area


def block_stats(block):
    max_x = 0
    max_y = 0
    min_x = 0
    min_y = 0
    areas = []
    for region in block['regions']:
        area = polygon_area([(p['x'], p['y']) for p in region['points']])
        max_x = max(max_x, max(p['x'] for p in region['points']))
        max_y = max(max_y, max(p['y'] for p in region['points']))
        min_x = min(min_x, min(p['x'] for p in region['points']))
        min_y = min(min_y, min(p['y'] for p in region['points']))
        areas.append(area)
    print('\tNumber of regions: {}'.format(len(areas)))
    print('\tAverage region area: {:.3f} cm^2'.format(statistics.mean(areas)))
    print('\tTotal area: {:.2f} cm^2'.format(sum(areas)))
    print('\tX extents: [{:.2f}, {:.2f}]'.format(min_x, max_x))
    print('\tY extents: [{:.2f}, {:.2f}]'.format(min_y, max_y))
    print('\tZ focus:', block['zfocus'])


def collimator_stats(blocks):
    print('First block:')
    block_stats(blocks[0])
    print('Last block:')
    block_stats(blocks[-1])
    print('Total blocks: {}'.format(len(blocks)))


def build_collimator(blocks):
    element = cgs.CGSElement()
    for block in blocks:
        for region in block['regions']:
            points = [(p['x'], p['y']) for p in region['points']]
            points, faces = solid_region(points, block['zmin'], block['zmax'], block['zfocus'])
            element.primitives.append(('polyhedron', points, faces))
    return element


def solid_region(planar_points, zmin, zmax, zfocus):
    far_points = []
    near_points = []
    # i specify points CCW in my egsinp
    # and OpenSCAD expects them CW (when looking at the face)
    # so reverse them
    for x, y in reversed(planar_points):
        v0 = np.array([x, y, zmin])
        vz = np.array([0, 0, zfocus])
        v = vz - v0
        t = (zmax - zmin) / v[2]
        v1 = v0 + t * v
        near_points.append(v0.tolist())
        far_points.append(v1.tolist())
    points = near_points + far_points
    faces = [
        list(range(len(near_points))),
        list(range(len(near_points), len(near_points) + len(far_points))),
    ]
    for i in range(len(far_points)):
        i, (i + 1) % len(far_points)
        faces.append([
            len(near_points) + i,
            len(near_points) + (i + 1) % len(far_points),
            (i + 1) % len(near_points),
            i
        ])
    return (points, faces)


def get_blocks(path):
    cms = egsinp.parse_egsinp(open(path).read())['cms']
    # in case they have other geometry before the collimator, strip it out
    blocks = [cm for cm in cms if cm['type'] == 'BLOCK']
    if not cms:
        print('Could not find any BLOCK CMs in {}'.format(path))
        sys.exit(1)
    return blocks


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('egsinp')
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--target-diameter', type=float, default=1.0)
    return parser.parse_args()


if __name__ == '__main__':
    main()
