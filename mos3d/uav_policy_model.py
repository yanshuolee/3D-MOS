# Copyright 2022 Kaiyu Zheng
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# policy model based on topological graph
import pomdp_py
from mos3d.oopomdp import M3Action, MotionAction, LookAction, DetectAction
from mos3d.util import euclidean_dist
from mos3d.robot_demo.conversion import convert, Frame
import random

class TopoMotionAction(MotionAction):
    def __init__(self, src, dst, src_nid=None, dst_nid=None):
        """Moves the robot from src to dst;
        Both should be points in the POMDP frame."""
        self.src = src
        self.dst = dst
        motion = (((dst[0] - src[0]),
                   (dst[1] - src[1]),
                   (dst[2] - src[2])),
                  (0,0,0))
        self._dist = euclidean_dist(src, dst)
        super().__init__(motion,
                         "move({}->{})".format(src_nid, dst_nid),
                         10*self._dist)  # distance cost; Motion is actually quite expensive
# TODO
class TorsoAction(MotionAction):
    """A torso action moves the robot up or down by one POMDP search space cell."""
    UP = "torso-up"
    DOWN = "torso-down"
    def __init__(self, direction):
        """Moves the torso up or down 1 step by POMDP space resolution."""
        if direction == TorsoAction.UP:
            motion = ((0,0,1), (0,0,0))
        else:
            motion = ((0,0,-1), (0,0,0))
        super().__init__(motion, direction)

    @classmethod
    def valid_actions(cls, robot_pose, search_space_resolution, torso_range):
        """Given robot_pose (in POMDP space), resolution, and
        torso_range (height, in meters, in world frame), returns
        a set of valid torso actions."""
        current_height = float(robot_pose[2]) * search_space_resolution
        actions = set({})
        if current_height + search_space_resolution < torso_range[1]:
            # can go up
            actions.add(TorsoAction(TorsoAction.UP))
        if current_height - search_space_resolution > torso_range[0]:
            # can go down
            actions.add(TorsoAction(TorsoAction.DOWN))
        return actions

# The camera on movo by default looks at +x direction (forward).
# Apply the following camera installation pose, which, if all 0,
# makes the camera looks at -z direction (in robot frame)
CAMERA_INSTALLATION_POSE = (0, 0, 0, 0, -90, 0)  # points to +x
MOVO_LOOK_DIRECTIONS = {
    "look-front": ((0,0,0), (0,0,0)),    # point to +x; (ros convention)
    "look-back":  ((0,0,0), (0,0,180)),   # point to -x
    "look-left":  ((0,0,0), (0,0,90)),    # point to +y
    "look-right": ((0,0,0), (0,0,-90)),  # point to -y
}

MOVO_LOOK_ACTIONS = {LookAction(motion=MOVO_LOOK_DIRECTIONS[look_direction],
                                look_direction=look_direction)
                     for look_direction in MOVO_LOOK_DIRECTIONS}

def look_action_for(look_direction):
    return LookAction(motion=MOVO_LOOK_DIRECTIONS[look_direction],
                      look_direction=look_direction)

NODE = {
(10, 10, 0):[(20, 22, 6),(20, 22, 8),(20, 44, 6),
             (20, 44, 8),(20, 66, 6),(20, 66, 8),
             (42, 22, 6),(42, 22, 8),(42, 44, 6),
             (42, 44, 8),(42, 66, 6),(42, 66, 8)],
(20, 22, 6):[(20, 44, 6),(42, 22, 6),(20, 22, 8)],
(20, 22, 8):[(20, 22, 6),(20, 44, 8),(42, 22, 8)],
(20, 44, 6):[(20, 66, 6),(20, 22, 6),(42, 44, 6),(20, 44, 8)],
(20, 44, 8):[(20, 66, 8),(20, 22, 8),(42, 44, 8),(20, 44, 6)],
(20, 66, 6):[(20, 44, 6),(42, 66, 6),(20, 66, 8)],
(20, 66, 8):[(20, 44, 8),(42, 66, 8),(20, 66, 6)],
(42, 22, 6):[(20, 22, 6),(42, 44, 6),(42, 22, 8)],
(42, 22, 8):[(20, 22, 8),(42, 44, 8),(42, 22, 6)],
(42, 44, 6):[(20, 44, 6),(42, 66, 6),(42, 22, 6),(42, 44, 8)],
(42, 44, 8):[(20, 44, 8),(42, 66, 8),(42, 22, 8),(42, 44, 6)],
(42, 66, 6):[(20, 66, 6),(42, 44, 6),(42, 66, 8)],
(42, 66, 8):[(20, 66, 8),(42, 44, 8),(42, 66, 6)],
}

class TopoPolicyModel(pomdp_py.RolloutPolicy):

    def __init__(self,
                 region_origin=(0,0,0),
                 search_space_resolution=0.3,
                 torso_range=(0.1, 1.5),
                 detect_after_look=True):
        # self._topo_map = topo_map
        self._detect_after_look = detect_after_look
        self._search_space_resolution = search_space_resolution
        self._region_origin = region_origin
        self._torso_range = torso_range
        self._motion_map = {}  # map from node to motion actions (in POMDP space coordinate)
        
        for nid in NODE:
            motion_actions = set({})
            src_pos = nid
            for dst_pos in NODE[nid]:
                motion_actions.add(TopoMotionAction(src_pos, dst_pos,
                                                    src_nid=src_pos, dst_nid=dst_pos))
            self._motion_map[nid] = motion_actions

        self._look_actions = MOVO_LOOK_ACTIONS

    def sample(self, state, history=None):
        return random.sample(self._get_all_actions(state=state, history=history), 1)[0]

    def probability(self, action, state, **kwargs):
        raise NotImplemented

    def argmax(self, state, **kwargs):
        """Returns the most likely action"""
        raise NotImplemented

    def get_all_actions(self, state, history=None):
        """note: detect can only happen after look."""
        can_detect = False
        if history is not None and len(history) > 1:
            # last action
            last_action = history[-1][0]
            if isinstance(last_action, LookAction):
                can_detect = True
        # world_point = convert(state.robot_pose, Frame.POMDP_SPACE, Frame.WORLD,
        #                       region_origin=self._region_origin,
        #                       search_space_resolution=self._search_space_resolution)
        # nid = self._topo_map.closest_node(world_point[0], world_point[1])
        motion_actions = self._motion_map[state.robot_pose[:3]]
        # torso_actions = TorsoAction.valid_actions(state.robot_pose,
        #                                           self._search_space_resolution,
        #                                           self._torso_range)
        
        if state.robot_pose[:3] == (10,10,0):
            return motion_actions
        
        if can_detect:
            return motion_actions | self._look_actions | set({DetectAction()})
        else:
            return motion_actions | self._look_actions

    def rollout(self, state, history=None):
        return random.sample(self.get_all_actions(state, history=history), 1)[0]
