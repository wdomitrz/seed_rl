#!/bin/bash
# Copyright 2019 The SEED Authors
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


set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source $DIR/setup.sh

export CONFIG=football
export ENVIRONMENT=football
export AGENT=vtrace
export WORKERS=25
export ACTORS_PER_WORKER=8

cat > /tmp/config.yaml <<EOF
trainingInput:
  scaleTier: CUSTOM
  masterType: standard_p100
  masterConfig:
    imageUri: ${IMAGE_URI}:${CONFIG}
  workerCount: ${WORKERS}
  workerType: standard
  workerConfig:
    imageUri: ${IMAGE_URI}:${CONFIG}
  parameterServerCount: 0
  hyperparameters:
    goal: MAXIMIZE
    hyperparameterMetricTag: episode_return
    maxTrials: 10
    maxParallelTrials: 1
    enableTrialEarlyStopping: True
    params:
    - parameterName: game
      type: CATEGORICAL
      categoricalValues:
      - 11_vs_11_hard_stochastic
    - parameterName: reward_experiment
      type: CATEGORICAL
      categoricalValues:
      - scoring
    - parameterName: inference_batch_size
      type: INTEGER
      minValue: 64
      maxValue: 64
      scaleType: UNIT_LOG_SCALE
    - parameterName: batch_size
      type: INTEGER
      minValue: 32
      maxValue: 32
      scaleType: UNIT_LOG_SCALE
    - parameterName: unroll_length
      type: INTEGER
      minValue: 32
      maxValue: 32
      scaleType: UNIT_LOG_SCALE
    - parameterName: total_environment_frames
      type: INTEGER
      minValue: 500000000
      maxValue: 500000000
      scaleType: UNIT_LOG_SCALE
    - parameterName: discounting
      type: DOUBLE
      minValue: 0.997
      maxValue: 0.997
      scaleType: UNIT_LOG_SCALE
    - parameterName: entropy_cost
      type: DOUBLE
      minValue: 0.000031948327487872845
      maxValue: 0.000031948327487872845
      scaleType: UNIT_LOG_SCALE
    - parameterName: lambda_
      type: DOUBLE
      minValue: 0.9
      maxValue: 0.9
      scaleType: UNIT_LOG_SCALE
    - parameterName: learning_rate
      type: DOUBLE
      minValue: 0.00005533923433816456
      maxValue: 0.00005533923433816456
      scaleType: UNIT_LOG_SCALE
EOF

start_training
