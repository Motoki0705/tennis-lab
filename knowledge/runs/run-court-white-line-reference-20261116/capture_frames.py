"""Decode predeclared original frames on CPU; no model or prediction inputs."""
from pathlib import Path
import hashlib
import json
import cv2

BASE = Path(__file__).resolve().parent
DATASET = Path('data/tennis_multiview/processed/meiji_4cam/dataset')
CLIPS = (0, 17, 34, 51, 68)


def main():
    cv2.setNumThreads(1)
    manifest_path=DATASET/'dataset.json';manifest=json.loads(manifest_path.read_text())
    records=[]
    for number in CLIPS:
        clip=next(c for c in manifest['clips'] if c['clip_id']==f'video_000/clip_{number:03d}')
        for camera in range(4):
            video=DATASET/clip['path']/'media'/f'cam{camera}.mp4'
            capture=cv2.VideoCapture(str(video));assert capture.isOpened(),video
            count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT));assert count==clip['num_frames'],(video,count)
            for index in (0,count//2,count-1):
                destination=BASE/'frames'/f'cam{camera}'/f'clip_{number:03d}_frame_{index:06d}.png'
                destination.parent.mkdir(parents=True,exist_ok=True)
                capture.set(cv2.CAP_PROP_POS_FRAMES,index);ok,frame=capture.read()
                assert ok and frame.shape==(1080,1920,3),(video,index)
                assert abs(capture.get(cv2.CAP_PROP_POS_FRAMES)-(index+1))<.5
                assert cv2.imwrite(str(destination),frame,[cv2.IMWRITE_PNG_COMPRESSION,3])
                records.append({'camera':f'cam{camera}','clip_id':clip['clip_id'],'frame_index':index,
                    'frame_count':count,'video':str(video),'path':str(destination.relative_to(BASE)),
                    'bgr_sha256':hashlib.sha256(frame.tobytes()).hexdigest(),
                    'png_sha256':hashlib.sha256(destination.read_bytes()).hexdigest()})
            capture.release();print(clip['clip_id'],f'cam{camera}',flush=True)
    (BASE/'frames.json').write_text(json.dumps({'dataset_manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        'sampling':'clips 0,17,34,51,68; frame 0,floor(N/2),N-1; fixed before fit','size_hw':[1080,1920],'records':records},indent=2))


if __name__=='__main__':main()
