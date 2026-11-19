"""Manual draft reference overlays for human review; never scores predictions."""

import json
from pathlib import Path

import cv2
import numpy as np

from src.tasks.court_detection.geometry.homography import COURT_HOMOGRAPHY_EDGES, court_template_xy

OUT = Path(__file__).resolve().parent
# Approximate visible ground-line intersections read from the original frame-0
# images, not model coordinates. These require human correction/confirmation.
ANCHORS = {
    0: {0:(993,430), 1:(1342,437), 2:(174,582), 3:(973,698),
        5:(242,592), 8:(892,453), 9:(1278,457), 10:(591,520), 11:(1058,555)},
    1: {0:(469,461), 1:(927,463), 2:(626,804), 3:(1617,658),
        5:(820,777), 10:(675,606), 11:(1201,572), 13:(969,586)},
    2: {0:(1033,449), 1:(1417,450), 2:(274,605), 3:(1071,709),
        5:(335,613), 7:(920,688), 10:(553,546), 11:(1143,569), 13:(868,555)},
    3: {2:(1075,543), 5:(1221,517), 8:(749,338), 9:(1073,308),
        10:(965,416), 11:(1457,349), 12:(932,322), 13:(1230,380)},
}


def label(img, message):
    result = img.copy()
    cv2.rectangle(result, (0,0), (1920,60), (0,0,0), -1)
    cv2.putText(result, message, (10,40), cv2.FONT_HERSHEY_SIMPLEX, .9, (255,255,255),2,cv2.LINE_AA)
    return result


def main():
    refs = OUT / 'reference_draft'
    refs.mkdir(exist_ok=True)
    template = court_template_xy(14).astype(float)
    records = []
    for cam, anchors in ANCHORS.items():
        ids = list(anchors)
        observed = np.array(list(anchors.values()), float)
        matrix, _ = cv2.findHomography(template[ids], observed, method=0)
        assert matrix is not None
        projected = cv2.perspectiveTransform(template[None],matrix)[0]
        record = {'camera': f'cam{cam}', 'source_clip': 'video_000/clip_000', 'frame_index':0,
                  'kind':'manual_visible_line_intersection_draft', 'human_reviewed':False,
                  'approved_for_evaluation':False, 'anchor_indices':ids,
                  'anchor_image_xy':observed.tolist(), 'matrix_court_metres_to_image_pixels':matrix.tolist(),
                  'draft_projected_xy':projected.tolist(), 'source_size_hw':[1080,1920],
                  'note':'Approximate manual marks, no model prediction fit. No error metrics or tuning before human review.'}
        records.append(record)
        rows = []
        for clip in [0,34,68]:
            image = cv2.imread(str(OUT/'frames'/f'clip_{clip:03d}_cam{cam}.jpg'))
            overlay = image.copy()
            for a,b in COURT_HOMOGRAPHY_EDGES:
                cv2.line(overlay,tuple(np.rint(projected[a]).astype(int)),tuple(np.rint(projected[b]).astype(int)),(255,170,20),2,cv2.LINE_AA)
            if clip == 0:
                for i, p in anchors.items():
                    cv2.drawMarker(overlay,p,(0,255,255),cv2.MARKER_CROSS,16,2)
                    cv2.putText(overlay,str(i),(p[0]+8,p[1]-9),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,255,255),2,cv2.LINE_AA)
            overlay = label(overlay, f'cam{cam} clip_{clip:03d}: UNAPPROVED MANUAL DRAFT / blue=ground lines / yellow=anchors')
            cv2.imwrite(str(refs/f'cam{cam}_clip_{clip:03d}_overlay.jpg'),overlay,[cv2.IMWRITE_JPEG_QUALITY,95])
            plain = label(image, f'cam{cam} clip_{clip:03d}: original frame 0')
            rows.append(np.hstack([cv2.resize(plain,(960,540)),cv2.resize(overlay,(960,540))]))
        cv2.imwrite(str(refs/f'cam{cam}_review.jpg'),np.vstack(rows),[cv2.IMWRITE_JPEG_QUALITY,94])
    (refs/'references.json').write_text(json.dumps({'schema':'court_reference_draft_v1',
        'approved_for_evaluation':False,'cameras':records},indent=2,allow_nan=False))


if __name__ == '__main__':
    main()
