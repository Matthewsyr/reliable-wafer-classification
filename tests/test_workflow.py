"""Small synthetic end-to-end fixtures, never scientific performance evidence."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from waferdefect.data import prepare, load_prepared
from waferdefect.experiment import train, evaluate, freeze
from waferdefect.io import write_csv, write_json, read_json, sha256_file
from waferdefect.labels import CLASSES


def fixture(folder):
    rows,split,maps=[],[],{}
    i=0
    for role,count in (('train',3),('tune',2),('audit',1),('test',1)):
        for label in CLASSES:
            for _ in range(count):
                a=np.ones((9,13),dtype=np.uint8)
                for bit in range(16):a.flat[bit]=2 if (i+1)&(1<<bit) else 1
                maps[f'row_{i}']=a
                rows.append({'source_row':i,'label':label,'lot':f'lot_{i}','wafer_index':1,'source_split':'test' if role=='test' else 'train'})
                if role!='test':split.append({'source_row':i,'role':role})
                i+=1
    np.savez(folder/'maps.npz',**maps);write_csv(folder/'metadata.csv',rows);write_csv(folder/'split.csv',split)
    return rows


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        fixture(self.root)

    def tearDown(self):self.temp.cleanup()

    def prepare(self,representation='rf'):
        return prepare(self.root/'data',maps_npz=self.root/'maps.npz',metadata=self.root/'metadata.csv',
                       existing_split=self.root/'split.csv',representation=representation,synthetic_fixture=True,include_linearity=True)

    def test_rf_train_evaluate_and_once_freeze_guard(self):
        self.prepare()
        write_json(self.root/'rf.json',{'kind':'rf','seed':42,'rf':{'n_estimators':4,'n_jobs':1},'feature_set':'spatial71+linearity'})
        run=train(self.root/'data',self.root/'rf.json',self.root/'run')
        self.assertTrue(run['synthetic_fixture']);self.assertFalse(run['original_test_used'])
        result=evaluate(self.root/'run',self.root/'data',self.root/'evaluation')
        self.assertEqual(result['metrics']['n'],18);self.assertEqual(result['role'],'tune')
        with self.assertRaises(ValueError):
            evaluate(self.root/'run',self.root/'data',self.root/'forbidden',partition='final-test')
        freeze(self.root/'run',self.root/'data',self.root/'freeze.json',test_source_sha256=sha256_file(self.root/'maps.npz'))
        prepare(self.root/'test',maps_npz=self.root/'maps.npz',metadata=self.root/'metadata.csv',representation='rf',
                partition='final-test',freeze=self.root/'freeze.json',synthetic_fixture=True,include_linearity=True)
        first=evaluate(self.root/'run',self.root/'test',self.root/'once',partition='final-test',freeze_path=self.root/'freeze.json',once=True)
        self.assertEqual(first['metrics']['n'],9)
        with self.assertRaises(FileExistsError):
            evaluate(self.root/'run',self.root/'test',self.root/'second',partition='final-test',freeze_path=self.root/'freeze.json',once=True)
        self.assertEqual(read_json(self.root/'run/final_test_once_receipt.json')['status'],'completed')

    def test_tampered_prepared_data_and_model_are_rejected(self):
        self.prepare()
        with (self.root/'data/targets.npy').open('ab') as handle:handle.write(b'tampered')
        with self.assertRaises(ValueError):load_prepared(self.root/'data')

    def test_tampered_frozen_model_is_rejected(self):
        self.prepare()
        write_json(self.root/'rf.json',{'kind':'rf','seed':42,'rf':{'n_estimators':2,'n_jobs':1}})
        train(self.root/'data',self.root/'rf.json',self.root/'run')
        freeze(self.root/'run',self.root/'data',self.root/'freeze.json',test_source_sha256=sha256_file(self.root/'maps.npz'))
        with (self.root/'run/model.joblib').open('ab') as handle:handle.write(b'tampered')
        with self.assertRaises(ValueError):evaluate(self.root/'run',self.root/'data',self.root/'rejected',freeze_path=self.root/'freeze.json')

    def test_conflicting_duplicate_maps_are_retained_and_logged(self):
        from waferdefect.io import read_csv
        rows=read_csv(self.root/'metadata.csv')
        # Same content, different true wafer identities and conflicting labels.
        # Both occur in the train role, so the preserved hash component is valid.
        with np.load(self.root/'maps.npz',allow_pickle=False) as archive:maps={k:archive[k] for k in archive.files}
        maps['row_3']=maps['row_0'].copy()
        np.savez(self.root/'maps.npz',**maps)
        result=self.prepare()
        audit=read_json(self.root/'data/preparation_audit.json')
        self.assertEqual(result['rows'],54)
        self.assertEqual(audit['conflicting_label_groups'],1)
        group=read_json(self.root/'data/duplicate_groups.json')[0]
        self.assertEqual(group['source_rows'],[0,3])

    def test_invalid_labeled_map_logged_not_silently_skipped(self):
        np.savez(self.root/'bad.npz',row_0=np.array([[1,4]],dtype=np.uint8))
        write_csv(self.root/'bad.csv',[{'source_row':0,'label':'none','lot':'lot','wafer_index':1,'source_split':'train'}])
        with self.assertRaises(ValueError):prepare(self.root/'invalid',maps_npz=self.root/'bad.npz',metadata=self.root/'bad.csv',representation='rf',synthetic_fixture=True)
        audit=read_json(self.root/'invalid/preparation_audit.json')
        self.assertEqual(len(audit['invalid_rows']),1)
        self.assertFalse((self.root/'invalid/dataset.json').exists())

    @unittest.skipUnless(importlib.util.find_spec('torch'),'Install cnn extra for end-to-end CNN smoke')
    def test_cnn_training_checkpoint_and_argmax_replay(self):
        self.prepare('both')
        write_json(self.root/'cnn.json',{'kind':'cnn','seed':42,'model_size':'small','epochs':1,'batch_size':16})
        train(self.root/'data',self.root/'cnn.json',self.root/'cnn')
        result=evaluate(self.root/'cnn',self.root/'data',self.root/'cnn_eval')
        p=np.load(self.root/'cnn_eval/probabilities.npy',allow_pickle=False)
        self.assertEqual(result['metrics']['n'],18);self.assertEqual(p.shape,(18,9))
        self.assertTrue(np.allclose(p.sum(1),1))


if __name__=='__main__':unittest.main()
