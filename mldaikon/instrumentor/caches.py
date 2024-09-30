from collections import defaultdict

from mldaikon.instrumentor.types import PTID

cache_meta_vars: dict[PTID, dict[str, dict]] = defaultdict(lambda: defaultdict(dict))
meta_vars: dict[str, object] = {}
