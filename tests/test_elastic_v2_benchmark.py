from patchlib.elastic_v2_benchmark import run_optimizer_benchmark

def test_reproducible_synthetic_ab():
    a=run_optimizer_benchmark(peers=8,epochs=5,seed=22,local_steps=2)
    b=run_optimizer_benchmark(peers=8,epochs=5,seed=22,local_steps=2)
    assert a['status']=='SYNTHETIC_OPTIMIZER_BENCHMARK'
    assert a['trace']==b['trace']
    assert a['trace'][2]['online']==6
    assert a['loss_last']['v2']<a['loss_first']['v2']
    assert a['loss_last']['legacy']<a['loss_first']['legacy']
