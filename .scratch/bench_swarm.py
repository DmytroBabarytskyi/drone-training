import sys; sys.path.insert(0,'src')
import numpy as np
from omegaconf import OmegaConf
from dronesim.core.config import load_config
from dronesim.ml.swarm.swarm_env import SwarmInterceptEnv, SwarmSceneConfig, SwarmRewardConfig
from dronesim.ml.swarm.analytic_policy import AnalyticPlacementPolicy

cfg=load_config('ml/swarm_intercept')
sd=OmegaConf.to_container(cfg.scene,resolve=True)
sd['target_speed_range_mps']=tuple(sd['target_speed_range_mps'])
sd['target_start_distance_range_m']=tuple(sd['target_start_distance_range_m'])
scene=SwarmSceneConfig(**sd)
rew=SwarmRewardConfig(**OmegaConf.to_container(cfg.reward,resolve=True))
env=SwarmInterceptEnv(scene_cfg=scene,reward_cfg=rew)

def run(name, act_fn, n=100, reset_fn=None):
    hits=[];colls=[];rews=[]
    for ep in range(n):
        obs,_=env.reset(seed=3000+ep)
        if reset_fn: reset_fn()
        done=False;s=0.0
        while not done:
            obs,r,done,_,info=env.step(act_fn(obs));s+=r
        hits.append(info['hits']);colls.append(info['collision_pairs']);rews.append(s)
    print(f'{name:38s} hit_rate={np.mean([h>0 for h in hits]):.2f}  mean_hits={np.mean(hits):.2f}  '
          f'collision_rate={np.mean([c>0 for c in colls]):.2f}  reward={np.mean(rews):7.1f}')

print(f'scene: {scene.n_drones} drones, sector +/-{scene.coverage_half_extent_m} m, '
      f'crossing +/-{scene.crossing_area_half_extent_m} m, kill r={scene.hit_radius_m} m, '
      f'v={scene.max_speed_mps} a={scene.max_accel_mps2}')
print()
run('null policy (static lattice)', lambda o: np.zeros(env.action_space.shape))

def mk(rep):
    return AnalyticPlacementPolicy(scene.n_drones, scene.hit_radius_m, scene.min_separation_m,
                                   scene.max_speed_mps, scene.max_accel_mps2, repulsion_gain=rep)
for rep,label in [(0.0,'analytic hex packing (no repulsion)'),(1.2,'analytic hex packing (+repulsion)')]:
    pol=mk(rep)
    run(label, lambda o,p=pol: p.act(env._positions), reset_fn=lambda p=pol: p.reset(env._predicted_point))
