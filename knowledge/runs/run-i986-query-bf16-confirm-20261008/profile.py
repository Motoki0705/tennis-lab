import cProfile, json, pstats, time
from pathlib import Path
import torch, cv2
from src.tasks.ball_detection.data.coordinate_dataset import CoordinateWindowDataset
import runpy
chosen_indices=runpy.run_path("tests/benchmarks/ball_mdd_query_gpu.py")["chosen_indices"]
root=Path('/home/kamimura/projects/tennis-lab/outputs/ball_detection/analyze/conv2d-query-only-gpu/20261008-input-profile-v1')
root.mkdir(exist_ok=False)
(root/'profile.py').write_text(Path(__file__).read_text())
torch.set_num_threads(2)
cv2.setNumThreads(1)
d=CoordinateWindowDataset(Path('/home/kamimura/projects/tennis-lab/outputs/ball_detection/precompute/mdd_coordinates_36/20261008-fps124-v1/mdd_only_windows.json'),split='train',requires_pose=False,mdd_a=.2,mdd_b=.15)
indices=chosen_indices(d,9)
report={}
for name in ('cold','warm'):
 profiler=cProfile.Profile()
 start=time.perf_counter()
 profiler.enable()
 for i in indices: sample=d[i]
 profiler.disable()
 elapsed=time.perf_counter()-start
 profiler.dump_stats(str(root/f'{name}.prof'))
 stats=pstats.Stats(profiler)
 rows=[]
 for (file,line,function), (primitive,calls,self_seconds,total_seconds,_) in stats.stats.items():
  rows.append(dict(file=file,line=line,function=function,calls=calls,self_seconds=self_seconds,total_seconds=total_seconds))
 rows.sort(key=lambda r:r['total_seconds'],reverse=True)
 report[name]=dict(windows=9,seconds=elapsed,top=rows[:25])
 print(name, elapsed, [(r['function'],round(r['total_seconds'],3)) for r in rows[:12]],flush=True)
(root/'profile.json').write_text(json.dumps(report,indent=2)+'\n')
