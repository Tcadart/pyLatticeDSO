from colorama import Fore, Style

class Timing:
    """
    A class to time function execution and track call hierarchy.
    """
    def __init__(self):
        from collections import defaultdict
        import threading, time
        self.timings = defaultdict(list)
        self.call_stack = []  # to track call hierarchy
        self.call_graph = defaultdict(lambda: defaultdict(float))
        self.call_counts = defaultdict(int)
        self.local = threading.local()
        # start/end are now set on first/last decorated call, not at import
        self._first_start = None
        self._last_end = None

    def timeit(self, func):
        """
        Decorator to time the execution of a function and track its calls.
        """
        import time
        def wrapper(*args, **kwargs):
            parent = self.call_stack[-1] if self.call_stack else None
            self.call_stack.append(func.__name__)
            start = time.perf_counter()
            # record the first start lazily
            if self._first_start is None:
                self._first_start = start

            result = func(*args, **kwargs)

            end = time.perf_counter()
            # update the last end time on every decorated call
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
        from collections import defaultdict
        self.timings = defaultdict(list)
        self.call_graph = defaultdict(lambda: defaultdict(float))
        self.call_counts = defaultdict(int)
        self.call_stack = []
        self._first_start = None
        self._last_end = None

    def summary(self):
        """
        Print a summary of the timings collected.
        """
        import time
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

        # compute total only over the timed section (first→last decorated call)
        if self._first_start is not None and self._last_end is not None:
            total_elapsed = self._last_end - self._first_start
            print(Fore.LIGHTYELLOW_EX, f"\n🔧 Total lattice generation runtime: {total_elapsed:.4f} s" + Style.RESET_ALL)
        else:
            print(Fore.LIGHTYELLOW_EX, "\n🔧 Total lattice generation runtime: n/a" + Style.RESET_ALL)
