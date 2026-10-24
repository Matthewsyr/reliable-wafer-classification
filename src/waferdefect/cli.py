"""Command line entry points. No subcommand automatically scores final test."""
import argparse
import json
from .data import prepare
from .experiment import train, freeze, evaluate


def main(argv=None):
    parser=argparse.ArgumentParser(prog='waferdefect',description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare',help='Audit maps and create development-first arrays')
    p.add_argument('--output',required=True)
    p.add_argument('--raw-pickle')
    p.add_argument('--raw-index',help='Verified offset/hash CSV for bounded-memory access')
    p.add_argument('--metadata',help='CSV or CSV.GZ with source_row,label,lot,wafer_index,source_split')
    p.add_argument('--maps-npz',help='Numeric-only named maps; no object-pickle arrays')
    p.add_argument('--representation',choices=['rf','cnn','both'],default='both')
    p.add_argument('--existing-split')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--partition',choices=['development','final-test'],default='development')
    p.add_argument('--freeze',help='Required before final-test preprocessing')
    p.add_argument('--synthetic-fixture',action='store_true',help='Mark fixture outputs as not experimental results')
    p.add_argument('--smoke-run',action='store_true',help='Mark a small real-data engineering check, not scientific performance')
    p.add_argument('--limit',type=int,help='Development-only smoke limit; never an official experiment')
    p.add_argument('--include-linearity',action='store_true')
    t=sub.add_parser('train',help='Train RF or the historical CNN on development roles only')
    t.add_argument('--data',required=True);t.add_argument('--config',required=True);t.add_argument('--output',required=True)
    t.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    f=sub.add_parser('freeze',help='Freeze model/input and optionally a validation-selected Scratch rule')
    f.add_argument('--run',required=True);f.add_argument('--data',required=True);f.add_argument('--output',required=True)
    f.add_argument('--test-source-sha256',required=True)
    f.add_argument('--precision-floor',type=float)
    f.add_argument('--validation-role',choices=['tune','validation'],default='tune')
    f.add_argument('--acknowledge-previous-test-exposure',action='store_true')
    f.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    e=sub.add_parser('evaluate',help='Evaluate one declared role; default is development tune')
    e.add_argument('--run',required=True);e.add_argument('--data',required=True);e.add_argument('--output',required=True)
    e.add_argument('--role',choices=['train','tune','validation','audit'],default='tune')
    e.add_argument('--partition',choices=['development','final-test'],default='development')
    e.add_argument('--freeze');e.add_argument('--once',action='store_true')
    e.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    args=vars(parser.parse_args(argv));command=args.pop('command')
    if command=='prepare':result=prepare(**args)
    elif command=='train':
        args['config_path']=args.pop('config');result=train(**args)
    elif command=='freeze':
        args['run_dir']=args.pop('run');result=freeze(**args)
    else:
        args['run_dir']=args.pop('run');args['freeze_path']=args.pop('freeze');result=evaluate(**args)
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
    return 0

