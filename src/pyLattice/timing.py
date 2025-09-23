from collections import defaultdict
import threading
from functools import wraps
from colorama import Fore, Style

# Top-level factory so pickle can find it (no lambdas inside __init__)
def _dd_float():
    return defaultdict(float)


class Timing:
    """
    A class to time function execution and track call hierarchy.
    Now pickle-friendly (no local lambdas / thread-locals in state).
    """
    def __init__(self):
        self.timings = defaultdict(list)
        self.call_stack = []  # to track call hierarchy
        self.call_graph = defaultdict(_dd_float)  # nested defaultdict(float) via top-level factory
        self.call_counts = defaultdict(int)
        self.local = threading.local()  # not pickled; rebuilt in __setstate__
        self._first_start = None
        self._last_end = None

    def __getstate__(self):
        """Drop unpicklable attributes and keep lightweight state."""
        state = self.__dict__.copy()
        # thread-local objects are not picklable
        state['local'] = None
        return state

    def __setstate__(self, state):
        """Restore state and rebuild factories / thread-local."""
        self.__dict__.update(state)
        self.local = threading.local()
        # Ensure defaultdict factories are preserved after unpickling
        if not isinstance(self.timings, defaultdict):
            self.timings = defaultdict(list, self.timings)
        if not isinstance(self.call_counts, defaultdict):
            self.call_counts = defaultdict(int, self.call_counts)
        if not isinstance(self.call_graph, defaultdict) or self.call_graph.default_factory is None:
            fixed = defaultdict(_dd_float)
            for k, v in dict(self.call_graph).items() if hasattr(self.call_graph, 'items') else []:
                fixed[k] = defaultdict(float, dict(v))
            self.call_graph = fixed

    def timeit(self, func):
        """
        Decorator to time the execution of a function and track its calls.
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            import time
            parent = self.call_stack[-1] if self.call_stack else None
            self.call_stack.append(func.__name__)
            start = time.perf_counter()
            if self._first_start is None:
                self._first_start = start
            result = func(*args, **kwargs)
            end = time.perf_counter()
            self._last_end = end
            elapsed = end - start
            self.timings[func.__name__].append(elapsed)
            self.call_counts[func.__name__] += 1
            if parent:
                self.call_graph[parent][func.__name__] += elapsed
            self.call_stack.pop()
            return result
        return wrapper

    def reset(self):
        """Optional: call before a new run if you want a fresh report."""
        self.timings = defaultdict(list)
        self.call_graph = defaultdict(_dd_float)
        self.call_counts = defaultdict(int)
        self.call_stack = []
        self._first_start = None
        self._last_end = None

    def summary(self):
        """
        Print a summary of the timings collected.
        """
        avg = lambda times: sum(times) / len(times) if times else 0
        print(Fore.GREEN + f"{'Function':<30} {'Calls':<10} {'Total (s)':<12} {'Avg (s)':<12} {'Max (s)':<12}")
        print("-" * 80 + Style.RESET_ALL)
        sorted_funcs = sorted(self.timings.items(), key=lambda x: sum(x[1]), reverse=True)
        for name, times in sorted_funcs:
            total = sum(times)
            count = len(times)
            print(f"{name:<30} {count:<10} {total:<12.6f} {avg(times):<12.6f} {max(times):<12.6f}")
            if name in self.call_graph:
                for subname, subtime in self.call_graph[name].items():
                    print(f"  └─ {subname:<26} {self.call_counts[subname]:<10} {subtime:<12.6f}")

        if self._first_start is not None and self._last_end is not None:
            total_elapsed = self._last_end - self._first_start
            print(Fore.LIGHTYELLOW_EX, f"\n🔧 Total lattice generation runtime: {total_elapsed:.4f} s" + Style.RESET_ALL)
        else:
            print(Fore.LIGHTYELLOW_EX, "\n🔧 Total lattice generation runtime: n/a" + Style.RESET_ALL)
