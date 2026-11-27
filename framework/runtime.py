"""Command-line lifecycle for experiments using the BASGA framework."""
from dataclasses import dataclass
from pathlib import Path
import argparse,ast,hashlib,importlib.metadata,math
from .config import effective_config,scientific_config
from .io import read_json,write_json,sha256_file


@dataclass
class Experiment:
    experiment_id: str
    kind: str
    root: Path
    config: dict
    visual_defaults: dict
    prepare: object
    verify: object
    render: object
    tag: str
    control: object = None
    comparison: object = None




def render_saved(spec,run,args):
    spec.render(run,args)
    actual=read_json(Path(run)/'metrics.json')
    write_json(Path(run)/'figures/visualization_parameters.json',{
        'experiment_id':spec.experiment_id,'source_run':str(run),'actual_seed':actual['seed'],
        'script_sha256':sha256_file(spec.root/'run.py'),
        'defaults':spec.visual_defaults,
        'arguments':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}})


def data_ast_signature(path):
    tree=ast.parse(Path(path).read_text(encoding='utf-8'))
    selected=[]
    for node in tree.body:
        names=[]
        if isinstance(node,(ast.FunctionDef,ast.ClassDef)): names=[node.name]
        elif isinstance(node,(ast.Assign,ast.AnnAssign)):
            targets=node.targets if isinstance(node,ast.Assign) else [node.target]
            names=[n.id for target in targets for n in ast.walk(target) if isinstance(n,ast.Name)]
        elif isinstance(node,(ast.Import,ast.ImportFrom)): names=[n.asname or n.name.split('.')[0] for n in node.names]
        if any(name.startswith('data_') for name in names): selected.append(node)
    return hashlib.sha256(ast.dump(ast.Module(body=selected,type_ignores=[]),include_attributes=False).encode()).hexdigest()


def preflight(cfg):
    from .evaluation import backend_environment
    result=backend_environment(cfg)
    if cfg['evaluation']['method']!='none':
        version=importlib.metadata.version('pynndescent')
        if version!='0.5.13':
            raise RuntimeError(f'pynndescent={version}; run python -m pip install --no-deps pynndescent==0.5.13 before training')
        result['pynndescent']=version
    return result


def _summary(root):
    import pandas as pd
    rows=[]
    for path in sorted(root.glob('*/*/baseline_boundary/*_seed*/metrics.json')):
        m=read_json(path)
        if m.get('status')!='complete': continue
        row={key:m.get(key) for key in ['experiment_id','task','graph_mode','seed','epochs','lr','intra_weight','inter_weight',
            'n_spots','n_genes','n_clusters','ARI','NMI','HS','n_evaluated','reconstruction_MSE_signed',
            'reconstruction_MSE_clipped','negative_reconstruction_fraction','config_id','protocol_id']}
        row.update(run_directory=str(path.parent),figure_directory=str(path.parent/'figures'))
        rows.append(row)
    if rows:
        temp=root/'all_runs.partial.csv'; pd.DataFrame(rows).to_csv(temp,index=False);temp.replace(root/'all_runs.csv')


def main(spec):
    parser=argparse.ArgumentParser(description=f'{spec.experiment_id}: BASGA training and visualization')
    parser.add_argument('--seed',type=int,default=10)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--data-dir',type=Path,default=spec.root/'data')
    parser.add_argument('--output-root',type=Path,help='Parent output directory; experiment ID is always appended to prevent collisions')
    parser.add_argument('--rscript',default='Rscript')
    parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--smoke',action='store_true',help='3 epochs using the original per-experiment smoke data scope, full model dimensions')
    parser.add_argument('--plot-only',action='store_true')
    parser.add_argument('--run-dir',type=Path,help='A completed run for --plot-only; otherwise latest run of this seed is used')
    parser.add_argument('--dpi',type=int,default=spec.visual_defaults.get('dpi',600))
    for key in ['point_size','spatial_size','umap_size','label_size','rotation_deg']:
        if key in spec.visual_defaults:
            parser.add_argument('--'+key.replace('_','-'),type=float,default=spec.visual_defaults[key])
    if 'background' in spec.visual_defaults:
        parser.add_argument('--background',choices=['image','white'],default=spec.visual_defaults['background'])
    if spec.kind=='slide': parser.add_argument('--skip-markers',action='store_true')
    if spec.kind=='hippo': parser.add_argument('--modes',nargs='+',choices=['2d','3d'],default=['2d','3d'])
    args=parser.parse_args()
    if args.seed<0 or args.dpi<72 or (args.run_dir and not args.plot_only): parser.error('Invalid seed/dpi or --run-dir without --plot-only')
    for key in ['point_size','spatial_size','umap_size','label_size','rotation_deg']:
        value=getattr(args,key,None)
        if value is not None and (not math.isfinite(value) or (key!='rotation_deg' and value<=0)):
            parser.error(f'Invalid {key}')
    if args.check_only and args.plot_only: parser.error('Use --check-only or --plot-only, not both')
    output=(args.output_root.resolve()/spec.experiment_id) if args.output_root else spec.root/'outputs'
    cache=spec.root/'cache'
    output.mkdir(parents=True,exist_ok=True);cache.mkdir(exist_ok=True)
    split='smoke' if args.smoke else 'audit'
    latest_path=output/f'latest_{split}_seed{args.seed}.json'
    from filelock import FileLock
    from .training import run_experiment
    with FileLock(str(output/'experiment.lock'),timeout=0):
        if args.plot_only:
            paths={'single':str(args.run_dir.resolve())} if args.run_dir else read_json(latest_path)['runs']
            for run in paths.values(): render_saved(spec,Path(run),args)
            if spec.comparison and '2d' in paths and '3d' in paths:
                spec.comparison(paths['2d'],paths['3d'],output/f'{split}_comparison_seed{args.seed}',args.dpi)
            print(f'REPLOTTED {spec.experiment_id}; original model outputs retained')
            return
        cfg=effective_config(spec.config,'baseline_boundary')
        cfg['experiment']={'id':spec.experiment_id,'kind':spec.kind,'task':'denoising' if spec.kind=='denoise' else 'domain'}
        cfg['data']['directory']=str(args.data_dir.resolve())
        cfg['data']['cache_dir']=str(cache.resolve());cfg['output']['directory']=str(output.resolve())
        cfg['evaluation']['rscript']=args.rscript
        if args.smoke:
            cfg['smoke']=True;cfg['training'].update(epochs=3,checkpoint_every=1,snapshot_epochs=[3])
        write_json(output/'environment_check.json',preflight(cfg))
        signature=data_ast_signature(spec.root/'run.py')
        write_json(output/'entrypoint.json',{'script':str(spec.root/'run.py'),'script_sha256':sha256_file(spec.root/'run.py'),
            'data_ast_sha256':signature,'visualization_parameters':vars(args)|{'data_dir':str(args.data_dir),'output_root':str(args.output_root),'run_dir':str(args.run_dir)}})
        modes=args.modes if spec.kind=='hippo' else ['single']
        if len(set(modes))!=len(modes): parser.error('Duplicate graph modes')
        completed={}
        for mode in modes:
            if spec.kind=='hippo': cfg['graph']['mode']=mode
            spec.verify(cfg)
            with FileLock(str(cache/'prepare.lock'),timeout=0):
                a,report=spec.prepare(cfg);del a
                write_json(output/f'data_report_{mode}.json',report)
                if args.check_only:
                    print(f"READY {spec.experiment_id}/{mode}: {report['n_spots']} spots, "
                          f"{report.get('n_hvg',report.get('n_genes_input'))} genes, "
                          f"{report['spatial_graph']['directed_nonself_edges']} non-self edges")
                    continue
                tag='hippo_'+mode if spec.kind=='hippo' else spec.tag
                run=run_experiment(cfg,args.seed,device_request=args.device,tag=tag,split=split,
                    skip_completed=True,resume=True,prepare_fn=spec.prepare,data_code_signature=signature)
            if spec.control: spec.control(run)
            render_saved(spec,run,args)
            completed[mode]=str(run)
        if args.check_only: return
        previous=read_json(latest_path).get('runs',{}) if latest_path.is_file() else {}
        previous.update(completed)
        write_json(latest_path,{'experiment_id':spec.experiment_id,'seed':args.seed,'split':split,'runs':previous,
            'figure_directories':{k:str(Path(v)/'figures') for k,v in previous.items()}})
        if spec.comparison and '2d' in previous and '3d' in previous:
            spec.comparison(previous['2d'],previous['3d'],output/f'{split}_comparison_seed{args.seed}',args.dpi)
        _summary(output)
        print(f'SUMMARY: {output/"all_runs.csv"}')
        print(f'LATEST: {latest_path}')
