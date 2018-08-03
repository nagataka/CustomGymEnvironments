"""
Extended [Taxi-v2](https://gym.openai.com/envs/Taxi-v2/) environment.
Map is 16 x 16 grid and it can be modified.
"""

import sys
from six import StringIO
from gym import utils
from gym.envs.toy_text import discrete
import numpy as np

"""
Col x Row x has passenger or not

  => 16 x 16 * 2 = 512 states in total
  
"""
MAP = [
    "+-------------------------------+",
    "| : | : : : : : : : : : : : : : |",
    "| : : : : : : : : : : : : : : : |",
    "| : : : : : : : : : : : : : : : |",
    "| | : | : : : : : | : : : : : : |",
    "| | : | : : : : : | : : : : : : |",
    "| : : | : : : : : | : : : : : : |",
    "| : : : : : : : : | : : : : : : |",
    "| : : : : : : : : | : : : : : : |",
    "| : : | : : : : : | : : : : : : |",
    "| : : | : : : : : | : : : : : : |",
    "| : : | : : : : : : : : : : : : |",
    "| : : : : : : : : : : : : : : : |",
    "| : | : : : : : : : : : : : : : |",
    "| : | : : : : : : : : : : : : : |",
    "| : : : : : : : : : : : : : : : |",
    "| : : : : : : : : : : : : : : : |",
    "+-------------------------------+",
]

class CustomTaxiEnv(discrete.DiscreteEnv):
    """
    The Taxi Problem
    from "Hierarchical Reinforcement Learning with the MAXQ Value Function Decomposition"
    by Tom Dietterich

    rendering:
    - blue: passenger
    - magenta: destination
    - yellow: empty taxi
    - green: full taxi
    - other letters: locations

    """
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self):
        self.desc = np.asarray(MAP,dtype='c')
        self.locs = locs = [(4,3), (0,4), (4,0), (0,0)]
        self.destinations = [(15,15), (15,0), (0,15), (8,8)]
        
        self.passidx = 0   # Need to be parametarized in the future
        self.destidx = 0
        pi, pj = self.locs[self.passidx]
        di, dj = self.destinations[self.destidx]
        self.desc[1+pi][2*pj+1] = 'P'
        self.desc[1+di][2*dj+1] = 'D'
        """
        Has the following members
        - nS: number of states
        - nA: number of actions
        - P: transitions (*)
        - isd: initial state distribution (**)

        (*) dictionary dict of dicts of lists, where
        P[s][a] == [(probability, nextstate, reward, done), ...]
        (**) list or array of length nS

        https://github.com/openai/gym/blob/4c460ba6c8959dd8e0a03b13a1ca817da6d4074f/gym/envs/toy_text/discrete.py#L16
        """

        nS = 512
        self.nR = 16
        self.nC = 16
        maxR = self.nR-1
        maxC = self.nC-1
        isd = np.zeros(nS)
        nA = 6
        P = {s : {a : [] for a in range(nA)} for s in range(nS)}
        for row in range(self.nR):
            for col in range(self.nC):
                for passidx in range(2):  # confusing. Shoud be changed to, for example, has_passenger
                    for a in range(nA):
                        state = self.encode(row, col, passidx, self.destidx)
                        newrow, newcol, newpassidx = row, col, passidx
                        reward = -1
                        done = False
                        taxiloc = (row, col)

                        if a==0:
                            newrow = min(row+1, maxR)
                            #print("Take action 0. Move row from {} to {}".format(row, newrow))
                        elif a==1:
                            newrow = max(row-1, 0)
                            #print("Take action 1. Move row from {} to {}".format(row, newrow))
                        if a==2 and self.desc[1+row,2*col+2]==b":":
                            newcol = min(col+1, maxC)
                            #print("Take action 2. Move col from {} to {}".format(col, newcol))
                        elif a==3 and self.desc[1+row,2*col]==b":":
                            newcol = max(col-1, 0)
                            #print("Take action 3. Move col from {} to {}".format(col, newcol))
                        elif a==4: # pickup
                            if (self.passidx == 0 and taxiloc == locs[self.passidx]):
                                newpassidx = 1
                            else:
                                reward = -10
                        elif a==5: # dropoff
                            if (taxiloc == self.destinations[self.destidx]) and passidx==1:
                                done = True
                                reward = 20
                            elif (taxiloc in locs) and self.passidx==1:  # What is this for??
                                newpassidx = locs.index(taxiloc)
                            else:
                                reward = -10
                        newstate = self.encode(newrow, newcol, newpassidx, self.destidx)
                        print("Take action {} at state {} transit to new state {}".format(a, state, newstate))
                        P[state][a].append((1.0, newstate, reward, done))
        #isd /= isd.sum()
        discrete.DiscreteEnv.__init__(self, nS, nA, P, isd)

    def encode(self, taxirow, taxicol, passloc, destidx):

        num = self.nC * taxirow + taxicol + passloc*256

        """
        map is expressed like this:

        0 ,1 ,2 ,3 , ... ,15
        16,17,18,19, ... ,31
        ..
        ..

        """
        #print("num is {}".format(num))

        return num

    def decode(self, i):
        print("decode {}".format(i))
        out = []
        out.append(self.destidx)
        has_passenger = False
        if i >= 256:
            has_passenger = True
            out.append(1)   # passidx
            row = int( (i-256)/self.nR )
            col = int( (i-256)%self.nR )
        else:
            out.append(0)
            row = int( i/self.nR )
            col = int( i%self.nR )
        out.append(col)
        out.append(row)
        
        return reversed(out)

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        out = self.desc.copy().tolist()
        out = [[c.decode('utf-8') for c in line] for line in out]
        taxirow, taxicol, passidx, destidx = self.decode(self.s)
        print("taxirow, taxicol is {} {}".format(taxirow, taxicol))
        def ul(x): return "_" if x == " " else x
        if passidx == 0:
            out[1+taxirow][2*taxicol+1] = utils.colorize(out[1+taxirow][2*taxicol+1], 'yellow', highlight=True)
            pi, pj = self.locs[passidx]
            out[1+pi][2*pj+1] = utils.colorize(out[1+pi][2*pj+1], 'blue', bold=True)
        else: # passenger in taxi
            out[1+taxirow][2*taxicol+1] = utils.colorize(ul(out[1+taxirow][2*taxicol+1]), 'green', highlight=True)

        di, dj = self.destinations[self.destidx]
        out[1+di][2*dj+1] = utils.colorize(out[1+di][2*dj+1], 'magenta')
        outfile.write("\n".join(["".join(row) for row in out])+"\n")
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(["South", "North", "East", "West", "Pickup", "Dropoff"][self.lastaction]))
        else: outfile.write("\n")

        # No need to return anything for human
        if mode != 'human':
            return outfile

    def get_idx(self):
        """return a list contains index
        """
        taxirow, taxicol, passidx, destidx = self.decode(self.s)
        if passidx < 4:
            pi, pj = self.locs[passidx]
        else:
            pi, pj = taxirow, taxicol
        di, dj = self.locs[destidx]
        return [taxirow, taxicol, di, dj, pi, pj, passidx]
