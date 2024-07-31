# Copyright 2023 The RoboPianist Authors.
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

from robopianist.wrappers.evaluation import MidiEvaluationWrapper
from robopianist.wrappers.pixels import PixelWrapper
from robopianist.wrappers.sound import PianoSoundVideoWrapper
from robopianist.wrappers.deep_mimic import DeepMimicWrapper
from robopianist.wrappers.fingering_emb import FingeringEmbWrapper
from robopianist.wrappers.dm2gym import Dm2GymWrapper
from robopianist.wrappers.residual import ResidualWrapper

__all__ = [
    "MidiEvaluationWrapper",
    "PianoSoundVideoWrapper",
    "PixelWrapper",
    "DeepMimicWrapper",
    "FingeringEmbWrapper",
    "Dm2GymWrapper",
    "ResidualWrapper",
]
