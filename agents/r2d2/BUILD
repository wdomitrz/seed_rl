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

licenses(["notice"])

exports_files(["LICENSE"])

py_library(
    name = "learner",
    srcs = ["learner.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow/cc/seed_rl:__subpackages__"],
    deps = [
        "//third_party/py/absl/flags",
        "//third_party/py/absl/logging",
        "//tensorflow/cc/seed_rl/common",
        "//tensorflow/cc/seed_rl/common:common_flags",
        "//tensorflow/cc/seed_rl/grpc",
        "//third_party/py/tensorflow:tensorflow_google",
    ],
)

py_test(
    name = "learner_test",
    srcs = ["learner_test.py"],
    python_version = "PY3",
    deps = [
        ":learner",
        "//third_party/py/numpy",
        "//tensorflow/cc/seed_rl/atari:lib",
        "//tensorflow/cc/seed_rl/common",
        "//third_party/py/tensorflow",
    ],
)
