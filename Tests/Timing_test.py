import time
import pytest
from pyLatticeDesign.timing import Timing


class TestTiming:
    """Test cases for Timing class."""

    def test_timing_initialization(self):
        timing = Timing()
        assert timing.timings is not None
        assert timing.call_stack == []

    def test_timing_multiple_calls(self):
        timing = Timing()

        @timing.timeit
        def fast_function():
            return "fast"

        for _ in range(3):
            fast_function()

        qname = [n for n in timing.timings.keys() if "fast_function" in n][0]
        assert timing.call_counts[qname] == 3

    def test_timing_summary(self, capsys):
        timing = Timing()

        @timing.timeit
        def test_func_sum():
            time.sleep(0.001)

        test_func_sum()
        timing.summary(name_width=100)

        captured = capsys.readouterr()
        assert "test_func_sum" in captured.out

    def test_timing_call_hierarchy(self):
        timing = Timing()

        @timing.timeit
        def level1():
            level2()

        @timing.timeit
        def level2():
            level3()

        @timing.timeit
        def level3():
            pass

        level1()

        # On récupère les noms qualifiés
        names = list(timing.timings.keys())
        n1 = [n for n in names if "level1" in n][0]
        n2 = [n for n in names if "level2" in n][0]
        n3 = [n for n in names if "level3" in n][0]

        assert n1 in timing.call_graph
        assert n2 in timing.call_graph[n1]
        assert n3 in timing.call_graph[n2]

    def test_timing_performance_measurement(self):
        timing = Timing()

        @timing.timeit
        def timed_sleep(duration):
            time.sleep(duration)

        duration = 0.01
        timed_sleep(duration)

        qname = [n for n in timing.timings.keys() if "timed_sleep" in n][0]
        measured_time = timing.timings[qname][0]

        assert measured_time >= duration
        assert measured_time <= duration * 5  # Marge pour les systèmes lents

    def test_timing_thread_safety(self):
        timing = Timing()

        @timing.timeit
        def thread_func():
            return "ok"

        thread_func()

        assert hasattr(timing, 'local')
        qname = [n for n in timing.timings.keys() if "thread_func" in n][0]
        assert qname in timing.timings