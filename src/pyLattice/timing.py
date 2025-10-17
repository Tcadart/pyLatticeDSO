# =============================================================================
# CLASS: Timing
# =============================================================================

from __future__ import annotations
from collections import defaultdict
from functools import wraps
import threading
import inspect
from typing import Callable, Iterable, Optional

# Top-level factory so pickle can find it (no lambdas inside __init__)
def _dd_float():
    return defaultdict(float)

class Timing:
    """
    General-purpose timing & call-graph collector.
    """

    def __init__(self):
        self.timings = defaultdict(list)                 # {qualified_name: [durations]}
        self.call_stack = []                             # [qualified_name, ...]
        self.call_graph = defaultdict(_dd_float)         # {parent: {child: total_time}}
        self.call_counts = defaultdict(int)              # {qualified_name: count}
        self.local = threading.local()                   # not pickled; rebuilt in __setstate__
        self.func_category = {}                          # qualified_name -> category string
        self._first_start = None
        self._last_end = None

    # --------------------------- pickle support --------------------------- #
    def __getstate__(self):
        state = self.__dict__.copy()
        state['local'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.local = threading.local()
        if not isinstance(self.timings, defaultdict):
            self.timings = defaultdict(list, self.timings)
        if not isinstance(self.call_counts, defaultdict):
            self.call_counts = defaultdict(int, self.call_counts)
        if not isinstance(self.call_graph, defaultdict) or self.call_graph.default_factory is None:
            fixed = defaultdict(_dd_float)
            for k, v in dict(self.call_graph).items() if hasattr(self.call_graph, 'items') else []:
                fixed[k] = defaultdict(float, dict(v))
            self.call_graph = fixed

    # ------------------------------ helpers ------------------------------ #
    def _qualified_name(self, func: Callable, args: tuple) -> str:
        """
        Derive a readable, stable name:
        - If bound method: "ClassName.method"
        - Else: "module:function"
        """
        try:
            # Detect bound method: first arg is 'self' or 'cls'
            if args:
                first = args[0]
                # instance method
                if hasattr(first, "__class__") and hasattr(func, "__name__"):
                    return f"{first.__class__.__name__}.{func.__name__}"
                # classmethod (cls)
                if inspect.isclass(first) and hasattr(func, "__name__"):
                    return f"{first.__name__}.{func.__name__}"
            # Fallback: module:function
            mod = getattr(func, "__module__", None) or "<unknown>"
            name = getattr(func, "__qualname__", None) or getattr(func, "__name__", "<unnamed>")
            return f"{mod}:{name}"
        except Exception:
            return getattr(func, "__name__", "<unnamed>")

    def category(self, label: str):
        """
        Decorator to tag functions with a category (e.g., 'sim', 'mesh', 'io').
        Use together with @timeit to enable grouped summaries.
        """

        def _decorator(func):
            setattr(func, "_timing_category", label)
            return func

        return _decorator

    # ------------------------------ API ---------------------------------- #
    def timeit(self, func: Callable):
        """
        Decorator to time execution and populate timings + call graph.
        """

        @wraps(func)
        def wrapper(*args, **kwargs):
            import time
            qname = self._qualified_name(func, args)
            cat = getattr(func, "_timing_category", None)
            if cat is not None:
                self.func_category[qname] = cat

            parent = self.call_stack[-1] if self.call_stack else None
            self.call_stack.append(qname)
            start = time.perf_counter()
            if self._first_start is None:
                self._first_start = start
            try:
                return func(*args, **kwargs)
            finally:
                end = time.perf_counter()
                self._last_end = end
                elapsed = end - start
                self.timings[qname].append(elapsed)
                self.call_counts[qname] += 1
                if parent:
                    self.call_graph[parent][qname] += elapsed
                self.call_stack.pop()

        return wrapper

    def reset(self):
        self.timings = defaultdict(list)
        self.call_graph = defaultdict(_dd_float)
        self.call_counts = defaultdict(int)
        self.call_stack = []
        self._first_start = None
        self._last_end = None

    # ------------------------------ report -------------------------------- #
    def summary(
            self,
            classes: Optional[Iterable[str]] = None,
            name_pattern: Optional[str] = None,
            max_depth: Optional[int] = None,
            min_total: float = 0.0,
            top_n: Optional[int] = None,
            print_children: bool = True,
            name_width: int = 40,
            group_by_category: bool = False,  # NEW
    ):
        """
        Print a filtered, aligned summary with truncated names.

        Parameters
        ----------
        name_width : int
            Fixed width for the function column. Names longer than this are truncated with an ellipsis.
        (Other params unchanged; see docstring above.)
        """
        from colorama import Fore, Style
        import re

        # -------- helpers -------- #
        def _truncate(s: str, width: int) -> str:
            if len(s) <= width:
                return s
            return s[: max(0, width - 1)] + "…"

        def _avg(times):
            return sum(times) / len(times) if times else 0.0

        # Build filters
        cls_set = set(classes) if classes else None
        pattern = re.compile(name_pattern) if name_pattern else None

        def include_name(name: str) -> bool:
            if cls_set and not any(name.startswith(c + ".") for c in cls_set):
                return False
            if pattern and not pattern.search(name):
                return False
            return True

        # Totals
        totals = {name: sum(times) for name, times in self.timings.items() if include_name(name)}

        # Depth approximation
        depth_cache: dict[str, int] = {}

        def depth_of(name: str) -> int:
            if name in depth_cache:
                return depth_cache[name]
            parents = [p for p, kids in self.call_graph.items() if name in kids]
            if not parents:
                depth_cache[name] = 0
            else:
                depth_cache[name] = min(depth_of(p) for p in parents) + 1
            return depth_cache[name]

        if max_depth is not None:
            totals = {n: t for n, t in totals.items() if depth_of(n) <= max_depth}

        totals = {n: t for n, t in totals.items() if t >= float(min_total)}
        ordered = sorted(totals.items(), key=lambda x: x[1], reverse=True)
        if top_n is not None:
            ordered = ordered[:top_n]

        # ----- printing -----
        def _avg(times):
            return sum(times) / len(times) if times else 0.0

        def _truncate(s: str, width: int) -> str:
            return s if len(s) <= width else s[: max(0, width - 1)] + "…"

        header = (
            f"{'Function':<{name_width}} "
            f"{'Calls':>10} {'Total (s)':>12} {'Avg (s)':>12} {'Max (s)':>12}"
        )
        print(Fore.GREEN + header)
        print("-" * (name_width + 1 + 10 + 1 + 12 + 1 + 12 + 1 + 12) + Style.RESET_ALL)

        def _print_row(name: str, total: float):
            times = self.timings[name]
            count = len(times)
            print(
                f"{_truncate(name, name_width):<{name_width}} "
                f"{count:>10} {total:>12.6f} {_avg(times):>12.6f} {max(times):>12.6f}"
            )

        def _print_children(name: str):
            if not (print_children and name in self.call_graph):
                return
            # children sorted by time
            for subname, subtime in sorted(self.call_graph[name].items(), key=lambda x: x[1], reverse=True):
                if not include_name(subname):
                    continue
                if max_depth is not None and depth_of(subname) > max_depth:
                    continue
                prefix = "└─ "
                child_width = max(0, name_width - len(prefix))
                child_name = prefix + _truncate(subname, child_width)
                print(f"{child_name:<{name_width}} {self.call_counts[subname]:>10} {subtime:>12.6f}")

        if not group_by_category:
            for name, total in ordered:
                _print_row(name, total)
                _print_children(name)
        else:
            # group by category (uncategorized last)
            buckets: dict[str, list[tuple[str, float]]] = defaultdict(list)
            for name, total in ordered:
                cat = self.func_category.get(name, "uncategorized")
                buckets[cat].append((name, total))

            # order categories by their cumulative time (desc), then 'uncategorized'
            def cat_total(cat_items):
                return sum(t for _, t in cat_items)

            ordered_cats = sorted(
                (c for c in buckets if c != "uncategorized"),
                key=lambda c: cat_total(buckets[c]),
                reverse=True,
            )
            if "uncategorized" in buckets:
                ordered_cats.append("uncategorized")

            for cat in ordered_cats:
                print(Fore.CYAN + f"\n[{cat}]" + Style.RESET_ALL)
                for name, total in buckets[cat]:
                    _print_row(name, total)
                    _print_children(name)

        # Global wall time
        if self._first_start is not None and self._last_end is not None:
            total_elapsed = self._last_end - self._first_start
            print(Fore.LIGHTYELLOW_EX, f"\n🔧 Total runtime: {total_elapsed:.4f} s" + Style.RESET_ALL)
        else:
            print(Fore.LIGHTYELLOW_EX, "\n🔧 Total runtime: n/a" + Style.RESET_ALL)

# Expose a shared singleton to be reused across modules
timing = Timing()