# Copyright 2026 You
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

"""ament_copyright check (skipped when ament tooling is unavailable)."""

import pytest

ament_copyright = pytest.importorskip('ament_copyright.main')


@pytest.mark.copyright
@pytest.mark.linter
def test_copyright():
    """Verify all source files carry a copyright/license header."""
    rc = ament_copyright.main(argv=['.', 'test'])
    assert rc == 0, 'Found errors'
