"""Synthetic mathematical fixtures; these tests are not model-quality results."""
import importlib.util
import unittest
import numpy as np
from waferdefect.labels import CLASSES, canonical_label, encode_labels, decode_labels
from waferdefect.features import extract_features, FEATURE_NAMES
from waferdefect.split import connected_components, grouped_split, assert_disjoint, align_external_split
from waferdefect.evaluation import classification_metrics, decide, scratch_pr
from waferdefect.experiment import build_rf
from waferdefect.geometry import extract_linearity_features


class LabelTests(unittest.TestCase):
    def test_explicit_order_and_missing_none_distinction(self):
        self.assertEqual(canonical_label(np.array([['None']])), 'none')
        self.assertIsNone(canonical_label(np.empty((0,0))))
        self.assertIsNone(canonical_label(None))
        np.testing.assert_array_equal(encode_labels(CLASSES),np.arange(9))
        self.assertEqual(decode_labels(np.arange(9)),list(CLASSES))
        for bad in ('Unknown',['Scratch','none'],None):
            with self.assertRaises(ValueError): encode_labels([bad])


class SplitTests(unittest.TestCase):
    def records(self):
        rows=[{'source_row':i,'lot':f'lot{i}','map_sha256':f'hash{i}','label':'none'} for i in range(12)]
        rows[1]['lot']=rows[0]['lot']
        rows[2]['map_sha256']=rows[1]['map_sha256']
        return rows

    def test_transitive_lot_or_hash_union_and_no_cross_role(self):
        rows=self.records();components=connected_components(rows)
        self.assertEqual(components[0],components[2])
        manifest=grouped_split(rows,seed=7)
        self.assertEqual(manifest,grouped_split(rows,seed=7))
        self.assertEqual(assert_disjoint(rows,[r['role'] for r in manifest])['map_hash_cross_role'],0)
        self.assertEqual(align_external_split(rows,manifest),manifest)

    def test_leaky_external_manifest_is_rejected_not_repaired(self):
        rows=self.records();manifest=grouped_split(rows)
        manifest[0]['role']='train';manifest[2]['role']='audit'
        with self.assertRaises(ValueError): align_external_split(rows,manifest)


class FeatureTests(unittest.TestCase):
    def test_original_features_finite_on_degenerate_geometry(self):
        for a in (np.array([[1]]),np.array([[2]]),np.array([[1,2]]),np.ones((3,17))):
            f=extract_features(a)
            self.assertEqual(f.shape,(71,));self.assertTrue(np.isfinite(f).all())
        with self.assertRaises(ValueError):extract_features(np.zeros((2,2)))

    def test_train_only_imputation_does_not_learn_validation_values(self):
        model=build_rf({'seed':42,'rf':{'n_estimators':3,'n_jobs':1}})
        x=np.array([[1.,np.nan],[3.,2.],[5.,4.],[7.,6.]])
        model.fit(x,np.array([0,1,0,1]))
        before=model.named_steps['imputer'].statistics_.copy()
        model.predict_proba(np.array([[1e8,np.nan],[np.nan,-1e8]]))
        np.testing.assert_array_equal(before,[4.,4.])
        np.testing.assert_array_equal(before,model.named_steps['imputer'].statistics_)

    def test_linearity_count_degeneracies_and_input_unchanged(self):
        for failures in (0,1,2):
            a=np.ones((11,13),dtype=np.uint8);a[5,2:2+failures]=2
            before=a.copy();f=extract_linearity_features(a)
            self.assertTrue(np.isfinite(f).all())
            np.testing.assert_array_equal(f[1:],0)
            if failures<2:self.assertEqual(f[0],0)
            else:self.assertAlmostEqual(float(f[0]),1.)
            np.testing.assert_array_equal(a,before)

    def test_line_vs_scatter_and_pca_reexpression(self):
        line=np.ones((33,33),dtype=np.uint8);line[16,2:31]=2
        scatter=np.ones_like(line);points=np.random.default_rng(19).choice(line.size,29,replace=False);scatter.flat[points]=2
        a,b=extract_linearity_features(line),extract_linearity_features(scatter)
        self.assertGreater(a[1],b[1]);self.assertAlmostEqual(float(a[0]),1.)
        base=dict(zip(FEATURE_NAMES,extract_features(scatter)))
        vx,vy,cov=[float(base[k]) for k in ('defect_variance_x','defect_variance_y','defect_covariance_xy')]
        self.assertAlmostEqual(float(b[0]),np.sqrt((vx-vy)**2+4*cov**2)/(vx+vy),places=6)


@unittest.skipUnless(importlib.util.find_spec('torch'),'Install the cnn extra to test tensor preprocessing')
class CnnTests(unittest.TestCase):
    def test_separate_categorical_masks_area_coverage(self):
        from waferdefect.cnn import preprocess_map
        raw=np.ones((128,128),dtype=np.uint8);raw[0,0]=2
        x=preprocess_map(raw)
        self.assertEqual(x.shape,(2,64,64));self.assertEqual(x.dtype,np.float32)
        self.assertTrue(np.all(x[0]==1));self.assertEqual(float(x[1,0,0]),.25)
        self.assertEqual(np.count_nonzero(x[1]),1)
        rectangular=preprocess_map(np.array([[0,1,2],[1,2,1]],dtype=np.uint8))
        self.assertTrue(np.all(rectangular[1]<=rectangular[0]))
        self.assertTrue(np.isfinite(rectangular).all())
        with self.assertRaises(ValueError):preprocess_map(np.array([[1,3]]))

    def test_class_weights_use_only_training_rows(self):
        from waferdefect.cnn import class_weights
        y=np.tile(np.arange(9),3);indices=np.arange(18)
        expected=class_weights(y,indices,'sqrt_inverse')
        changed=y.copy();changed[18:]=7
        actual=class_weights(changed,indices,'sqrt_inverse')
        np.testing.assert_array_equal(expected[0],actual[0]);np.testing.assert_array_equal(expected[1],actual[1])


class DecisionTests(unittest.TestCase):
    def test_argmax_and_scratch_priority_are_different_rules(self):
        p=np.full((3,9),.01);p[0,7]=.5;p[0,8]=.43;p[1,0]=.55;p[1,7]=.38;p[2,7]=.6;p[2,8]=.33
        # Construct exact normalized distributions, preserving ordering.
        p/=p.sum(1,keepdims=True)
        np.testing.assert_array_equal(decide(p),p.argmax(1))
        priority=decide(p,.55)
        np.testing.assert_array_equal(priority,[8,0,7])
        with self.assertRaises(ValueError):decide(np.ones((2,9)))

    def test_fixed_nine_class_confusion_and_normal_denominator(self):
        y=np.array([8,8,8,7,7,0]);pred=np.array([8,7,0,7,8,0])
        m=classification_metrics(y,pred)
        self.assertEqual(m['normal_any_defect_count'],2);self.assertEqual(m['normal_to_scratch_count'],1)
        self.assertAlmostEqual(m['normal_to_scratch_per_1000'],1000/3)
        self.assertAlmostEqual(m['macro_f1'],(.4+.5+2/3)/9)
        self.assertEqual(m['scratch_recall'],.5);self.assertFalse(m['per_class']['Donut']['recall_defined'])

    def test_precision_floor_handles_tied_scores_and_no_eligible_rule(self):
        p=np.zeros((4,9));p[:,7]=[.9,.8,.8,.1];p[:,8]=1-p[:,7]
        m=scratch_pr(np.array([7,7,8,8]),p,.8)
        self.assertAlmostEqual(m['selected']['threshold'],.9)
        tied=np.zeros((2,9));tied[:,7]=.5;tied[:,8]=.5
        self.assertIsNone(scratch_pr(np.array([7,8]),tied,.8)['selected'])


if __name__=='__main__':unittest.main()

