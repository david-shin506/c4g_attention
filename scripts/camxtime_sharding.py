"""Deterministic, balanced rig partitions; shared GT never crosses workers."""
import hashlib

def partition_plan(plan,num_shards):
    if num_shards<1: raise ValueError('num_shards must be positive')
    rigs=sorted({(r['scene'],r['trajectory']) for r in plan},
                key=lambda key:(hashlib.sha256(f'camxtime-shard:111123:{key[0]}:{key[1]}'.encode()).digest(),key))
    owners={rig:i%num_shards for i,rig in enumerate(rigs)}
    partitions=[[] for _ in range(num_shards)]
    for row in plan: partitions[owners[row['scene'],row['trajectory']]].append(row)
    return partitions
