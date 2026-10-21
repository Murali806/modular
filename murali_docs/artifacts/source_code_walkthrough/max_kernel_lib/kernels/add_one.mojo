# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

import extensibility

from extensibility import InputTensor, OutputTensor, foreach
from max.gpu.host import DeviceContext
from std.utils.coord import Coord


@extensibility.register("demo_add_one")
struct DemoAddOne:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        output: OutputTensor,
        x: InputTensor[dtype=output.dtype, rank=output.rank, ...],
        ctx: DeviceContext,
    ) raises:
        @inline(.always)
        def add_one[width: Int](idx: Coord) {var x} -> SIMD[x.dtype, width]:
            return x.load[width](idx) + 1

        foreach[target=target](add_one, output, ctx)
