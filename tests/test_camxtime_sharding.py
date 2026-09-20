import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from camxtime_sharding import partition_plan

class ShardingTests(unittest.TestCase):
    def plan(self):
        return [{'scene':f'Scene{s:03d}','trajectory':f'traj{t}', 'sample_id':f'{s}:{t}:{w}'}
                for s in range(12) for t in range(4) for w in range(3)]
    def test_partition_keeps_shared_assets_on_one_worker(self):
        plan=self.plan();parts=partition_plan(plan,4);seen=set();owners={}
        self.assertEqual([len(p) for p in parts],[36]*4)
        for worker,part in enumerate(parts):
            for row in part:
                self.assertNotIn(row['sample_id'],seen);seen.add(row['sample_id'])
                key=(row['scene'],row['trajectory'])
                self.assertEqual(owners.setdefault(key,worker),worker)
        self.assertEqual(len(seen),len(plan))
    def test_assignment_is_independent_of_plan_enumeration(self):
        def mapping(plan):
            return {r['sample_id']:i for i,part in enumerate(partition_plan(plan,4)) for r in part}
        self.assertEqual(mapping(self.plan()),mapping(list(reversed(self.plan()))))

if __name__=='__main__':unittest.main()
