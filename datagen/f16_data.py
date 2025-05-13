# packages
from functools import partial
import math
import numpy as np
from numpy import deg2rad
import torch
from tqdm import tqdm

# jax imports
import jax
import jax.numpy as jnp
from jax_f16.f16_types import S
from jax_f16.f16_utils import f16state
from jax_f16.highlevel.controlled_f16 import controlled_f16


#*******************************************************************************
# class definitions
#*******************************************************************************
class GCAS:
    STANDBY = 0
    ROLL = 1
    PULL = 2
    WAITING = 3


class GcasAutopilot:
    '''ground collision avoidance autopilot'''

    def __init__(self, init_mode=0, gain_str='old', stdout=False):

        assert init_mode in range(4)

        # config
        self.cfg_eps_phi = deg2rad(5)       # max abs roll angle before pull
        self.cfg_eps_p = deg2rad(10)        # max abs roll rate before pull
        self.cfg_path_goal = deg2rad(0)     # min path angle before completion
        self.cfg_k_prop = 4                 # proportional control gain
        self.cfg_k_der = 2                  # derivative control gain
        self.cfg_flight_deck = 1000         # altitude at which GCAS activates
        self.cfg_min_pull_time = 2          # min duration of pull up

        self.cfg_nz_des = 5

        self.pull_start_time = 0
        self.stdout = stdout

        self.waiting_cmd = jnp.zeros(4)
        self.waiting_time = 2
        self.mode = init_mode

    def log(self, s):
        'print to terminal if stdout is true'

        if self.stdout:
            print(s)

    def are_wings_level(self, x_f16):
        'are the wings level?'

        phi = x_f16[S.PHI]

        radsFromWingsLevel = jnp.round(phi / (2 * jnp.pi))

        return jnp.abs(phi - (2 * jnp.pi)  * radsFromWingsLevel) < self.cfg_eps_phi

    def is_roll_rate_low(self, x_f16):
        'is the roll rate low enough to switch to pull?'

        p = x_f16[S.P]

        return abs(p) < self.cfg_eps_p

    def is_above_flight_deck(self, x_f16):
        'is the aircraft above the flight deck?'

        alt = x_f16[S.ALT]

        return alt >= self.cfg_flight_deck

    def is_nose_high_enough(self, x_f16):
        'is the nose high enough?'

        theta = x_f16[S.THETA]
        alpha = x_f16[S.ALPHA]
        # Determine which angle is "level" (0, 360, 720, etc)
        radsFromNoseLevel = jnp.round((theta-alpha)/(2 * jnp.pi))
        # Evaluate boolean
        return ((theta-alpha) - 2 * jnp.pi * radsFromNoseLevel) > self.cfg_path_goal

    def get_u_ref(self, x_f16):
        '''get the reference input signals'''
        def roll_or_pull():
            roll_condition = jnp.logical_and(self.is_roll_rate_low(x_f16), self.are_wings_level(x_f16))
            return jax.lax.cond(roll_condition, lambda _: self.pull_nose_level(), lambda _: self.roll_wings_level(x_f16), None)

        def standby_or_roll():
            standby_condition = jnp.logical_and(jnp.logical_not(self.is_nose_high_enough(x_f16)), jnp.logical_not(self.is_above_flight_deck(x_f16)))
            return jax.lax.cond(standby_condition, lambda _: roll_or_pull(), lambda _: jnp.zeros(4), None)

        pull_condition = jnp.logical_and(self.is_nose_high_enough(x_f16), True)
        return jax.lax.cond(pull_condition, lambda _: jnp.zeros(4), lambda _: standby_or_roll(), None)
    

    def get_u_ref_orig(self, _t, x_f16):
        '''get the reference input signals'''

        if self.mode == 'waiting':
            # time-triggered start after two seconds
            if _t + 1e-6 >= self.waiting_time:
                self.mode = 'roll'
        elif self.mode == 'standby':
            if not self.is_nose_high_enough(x_f16) and not self.is_above_flight_deck(x_f16):
                self.mode = 'roll'
        elif self.mode == 'roll':
            if self.is_roll_rate_low(x_f16) and self.are_wings_level(x_f16):
                self.mode = 'pull'
                self.pull_start_time = _t
        else:
            assert self.mode == 'pull', f"unknown mode: {self.mode}"

            if self.is_nose_high_enough(x_f16) and _t >= self.pull_start_time + self.cfg_min_pull_time:
                self.mode = 'standby'

        if self.mode == 'standby':
            rv = np.zeros(4)
        elif self.mode == 'waiting':
            rv = self.waiting_cmd
        elif self.mode == 'roll':
            rv = self.roll_wings_level(x_f16)
        else:
            assert self.mode == 'pull', f"unknown mode: {self.mode}"
            rv = self.pull_nose_level()

        return rv

    def pull_nose_level(self):
        'get commands in mode PULL'
        rv = jnp.array([self.cfg_nz_des, 0.0, 0.0, 0.0]) 

        return rv

    def roll_wings_level(self, x_f16):
        'get commands in mode ROLL'

        phi = x_f16[S.PHI]
        p = x_f16[S.P]
        # Determine which angle is "level" (0, 360, 720, etc)
        radsFromWingsLevel = jnp.round(phi / (2 * jnp.pi))
        # PD Control until phi == pi * radsFromWingsLevel
        ps = -(phi - (2 * jnp.pi) * radsFromWingsLevel) * self.cfg_k_prop - p * self.cfg_k_der
        # Build commands to roll wings level
        rv = jnp.array([0.0, ps, 0.0, 0.0])

        return rv


def inner_step(autopilot, x, dt, inner_steps=10):
    for _ in range(inner_steps):
        u = autopilot.get_u_ref(x)
        xdot = controlled_f16(x, u).xd
        x = x + xdot * dt
    return x


@partial(jax.jit, static_argnames=['T', 'dt', 'inner_steps'])
def sim_gcas(
        f16_state,
        T=150,
        dt=1/500,
        inner_steps=10
        ):
    
    ap = GcasAutopilot()
    
    x = f16_state
    alts = jnp.zeros(T)

    def body_fun(carry, i):
        alts, x = carry
        alts = alts.at[i].set(x[S.ALT])
        x = inner_step(ap, x, dt, inner_steps=inner_steps)
        return (alts, x), x

    (alts, x), xs = jax.lax.scan(body_fun, (alts, x), jnp.arange(T))

    return xs


#*******************************************************************************
# generate data
#*******************************************************************************
# define initial state means
vt0     = 540
alpha0  =deg2rad(2.1215)
beta0   =0
phi0    = -math.pi/8
theta0  = (-math.pi/2)*0.3
psi0    = 0.0
p0      = 0.0
q0      = 0.0
r0      = 0.0
alt0    = 900
power0  = 9

N       = 1000000   # number of simulations to generate
T       = 150       # number of time steps

targets = np.zeros((N, 12))
for i in tqdm(range(N)):
    vt = vt0 + 10. * np.random.randn()
    alpha = alpha0 + 0.05 * np.random.randn()
    phi = phi0 + 0.1 * np.random.randn()
    theta = theta0 + 0.1 * np.random.randn()
    alt = alt0 + 10. * np.random.randn()

    f16_state = f16state(
        vt, [alpha, beta0], [phi, theta, psi0], [p0, q0, r0], 
        [0, 0, alt], power0, [0, 0, 0])

    xs = sim_gcas(f16_state, T=T, inner_steps=10)
    # skip angle of attack, engine power lag, stability roll rate, side accel 
    # and yaw rate
    targets[i, 0] = xs[-1, 0]
    targets[i, 1:11] = xs[-1, 2:12]
    targets[i, 11] = xs[-1, 13]

targets = torch.tensor(targets)
normalized_targets = (targets - targets.mean(dim=0))/targets.std(dim=0)

np.savetxt("f16-flow.csv", normalized_targets, delimiter=",")
