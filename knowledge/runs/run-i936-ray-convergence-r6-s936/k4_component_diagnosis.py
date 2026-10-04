import argparse,importlib.util,json,pathlib,yaml
parser=argparse.ArgumentParser();parser.add_argument("--output",type=pathlib.Path,required=True);args=parser.parse_args()
if args.output.exists():raise FileExistsError(args.output)
p=pathlib.Path(__file__).with_name('cost_k4.py');spec=importlib.util.spec_from_file_location('cost',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
from src.utils.geometry.probabilistic_triangulation import solver
original=solver.single_view_moments
source=pathlib.Path('/home/kamimura/projects/tennis-lab/data/ball_refiner/synthetic-3d-i936-preflight-r5');manifest=json.loads((source/'manifest.json').read_text());settings=yaml.safe_load(pathlib.Path('src/tasks/ball_refiner/refiner_3d/dataset_plan.yaml').read_text())['degradation']
record=next(r for r in manifest['rallies'] if r['rally_id']=='test-00003')
real=m.triangulate_converged
holder=[]
def wrapped(*args,**kwargs):
 r=real(*args,**kwargs);holder.append(r);return r
m.triangulate_converged=wrapped
reports=[]
for factor in [1,4]:
 solver.single_view_moments=lambda camera,means,cov,prior,n:original(camera,means,cov,prior,n*factor)
 row=m.one((source/(record['rally_id']+'.npz'),record,settings,0));r=holder[-1]
 row['single_order_multiplier']=factor
 row['nonconverged_components']=[{'index':i,'active':r.posterior.camera_subsets[i].tolist(),'weight':float(r.posterior.distribution.weights[i]),'changes':r.component_changes[i].tolist()} for i in range(len(r.component_converged)) if not r.component_converged[i]]
 reports.append(row);print(json.dumps(row,indent=2),flush=True)
args.output.write_text(json.dumps(reports,indent=2))
