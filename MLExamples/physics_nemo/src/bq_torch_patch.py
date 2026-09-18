"""Force PhysicsNeMo radius_search onto the pure-torch backend.

On AMD/ROCm, `warp-lang` imports but has no HIP backend, so it registers as
`available=True` yet fails at runtime (IndexError: cuda_devices[0]) when
GeoTransolver's BQWarp runs its ball query. Warp is rank 0, so default dispatch
picks it over the shipped pure-torch backend (rank 1).

Importing this module marks the warp backend unavailable, so dispatch selects
the torch backend (validated to match warp's padding convention and to run on
the MI300X). Local features stay ON — full-accuracy GeoTransolver/GeoFLARE.

Import once, before any model is built:  `import bq_torch_patch`
"""
from dataclasses import replace

from physicsnemo.nn.functional.neighbors.radius_search.radius_search import (
    RadiusSearch,
)


def apply() -> str:
    impls = RadiusSearch._get_impls()
    if "warp" in impls and impls["warp"].available:
        impls["warp"] = replace(impls["warp"], available=False)
    selected = sorted(
        (i for i in impls.values() if i.available), key=lambda i: i.rank
    )[0].name
    return selected


_selected = apply()
print(f"[bq_torch_patch] radius_search backend forced -> '{_selected}'")
