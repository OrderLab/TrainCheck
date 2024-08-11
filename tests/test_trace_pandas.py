import timeit
import unittest

from mldaikon.trace.trace import Trace, read_trace_file

# Import the module to test
from mldaikon.trace.trace_pandas import Trace_Pandas, read_trace_file_Pandas


# Differential test: check the output of the two implementations (modin.pandas and polars) are the same
class TestTracePandas(unittest.TestCase):

    def setUp(self):
        self.trace_file = "tests/test_trace_pandas/test_API_trace.log"
        self.trace_data_polars = read_trace_file(self.trace_file)
        self.trace_data_pandas = read_trace_file_Pandas(self.trace_file)
        self.start_time = self.trace_data_pandas.get_start_time()
        self.end_time = self.trace_data_pandas.get_end_time()
        assert isinstance(self.trace_data_polars, Trace)
        assert isinstance(self.trace_data_pandas, Trace_Pandas)

    def test_read_trace_file(self):
        # Check the two implementations return the same output
        self.assertEqual(self.trace_data_polars, self.trace_data_pandas)
        # Check the efficiency of the two implementations
        time_polars = timeit.timeit(lambda: read_trace_file(self.trace_file), number=10)
        time_pandas = timeit.timeit(
            lambda: read_trace_file_Pandas(self.trace_file), number=10
        )
        self.assertLess(time_pandas, time_polars)
        # print the ratio of the two time
        print("The ratio of the two time is: ", time_polars / time_pandas)

    def test_rm_incomplete_trailing_func_calls(self):
        self.trace_data_polars.rm_incomplete_trailing_func_calls()
        self.trace_data_pandas.rm_incomplete_trailing_func_calls()
        self.assertEqual(self.trace_data_polars, self.trace_data_pandas)

        # check efficiency of the two implementations
        time_polars = timeit.timeit(
            lambda: self.trace_data_polars.rm_incomplete_trailing_func_calls(),
            number=10,
        )
        time_pandas = timeit.timeit(
            lambda: self.trace_data_pandas.rm_incomplete_trailing_func_calls(),
            number=10,
        )
        self.assertLess(time_pandas, time_polars)
        # print the ratio of the two time
        print("The ratio of the two time is: ", time_polars / time_pandas)

    def test_get_start_time(self):
        self.assertEqual(
            self.trace_data_polars.get_start_time(),
            self.trace_data_pandas.get_start_time(),
        )

        # check efficiency of the two implementations
        time_polars = timeit.timeit(
            lambda: self.trace_data_polars.get_start_time(), number=10
        )
        time_pandas = timeit.timeit(
            lambda: self.trace_data_pandas.get_start_time(), number=10
        )
        self.assertLess(time_pandas, time_polars)
        # print the ratio of the two time
        print("The ratio of the two time is: ", time_polars / time_pandas)

    def test_get_end_time(self):
        self.assertEqual(
            self.trace_data_polars.get_end_time(), self.trace_data_pandas.get_end_time()
        )

        # check efficiency of the two implementations
        time_polars = timeit.timeit(
            lambda: self.trace_data_polars.get_end_time(), number=10
        )
        time_pandas = timeit.timeit(
            lambda: self.trace_data_pandas.get_end_time(), number=10
        )
        self.assertLess(time_pandas, time_polars)
        # print the ratio of the two time
        print("The ratio of the two time is: ", time_polars / time_pandas)

    def test_get_process_ids(self):
        self.assertEqual(
            self.trace_data_polars.get_process_ids(),
            self.trace_data_pandas.get_process_ids(),
        )

        # check efficiency of the two implementations
        time_polars = timeit.timeit(
            lambda: self.trace_data_polars.get_process_ids(), number=10
        )
        time_pandas = timeit.timeit(
            lambda: self.trace_data_pandas.get_process_ids(), number=10
        )
        self.assertLess(time_pandas, time_polars)
        # print the ratio of the two time
        print("The ratio of the two time is: ", time_polars / time_pandas)

    def test_get_thread_ids(self):
        self.assertEqual(
            self.trace_data_polars.get_thread_ids(),
            self.trace_data_pandas.get_thread_ids(),
        )

        # check efficiency of the two implementations
        time_polars = timeit.timeit(
            lambda: self.trace_data_polars.get_thread_ids(), number=10
        )
        time_pandas = timeit.timeit(
            lambda: self.trace_data_pandas.get_thread_ids(), number=10
        )
        self.assertLess(time_pandas, time_polars)
        # print the ratio of the two time
        print("The ratio of the two time is: ", time_polars / time_pandas)

    def test_get_start_time_with_specific_id(self):
        # get the last process_id and thread_id
        process_id = self.get_process_ids()[-1]
        thread_id = self.get_thread_ids()[-1]
        self.assertEqual(
            self.trace_data_polars.get_start_time(process_id, thread_id),
            self.trace_data_pandas.get_start_time(process_id, thread_id),
        )

        # check efficiency of the two implementations
        time_polars = timeit.timeit(
            lambda: self.trace_data_polars.get_start_time(process_id, thread_id),
            number=10,
        )
        time_pandas = timeit.timeit(
            lambda: self.trace_data_pandas.get_start_time(process_id, thread_id),
            number=10,
        )
        self.assertLess(time_pandas, time_polars)
        # print the ratio of the two time
        print("The ratio of the two time is: ", time_polars / time_pandas)

    def test_get_end_time_with_specific_id(self):
        # get the first process_id and thread_id
        process_id = self.get_process_ids()[0]
        thread_id = self.get_thread_ids()[0]
        self.assertEqual(
            self.trace_data_polars.get_end_time(process_id, thread_id),
            self.trace_data_pandas.get_end_time(process_id, thread_id),
        )

        # check efficiency of the two implementations
        time_polars = timeit.timeit(
            lambda: self.trace_data_polars.get_end_time(process_id, thread_id),
            number=10,
        )
        time_pandas = timeit.timeit(
            lambda: self.trace_data_pandas.get_end_time(process_id, thread_id),
            number=10,
        )
        self.assertLess(time_pandas, time_polars)
        # print the ratio of the two time
        print("The ratio of the two time is: ", time_polars / time_pandas)

    def test_get_func_names(self):
        self.assertEqual(
            self.trace_data_polars.get_func_names(),
            self.trace_data_pandas.get_func_names(),
        )

        # check efficiency of the two implementations
        time_polars = timeit.timeit(
            lambda: self.trace_data_polars.get_func_names(), number=10
        )
        time_pandas = timeit.timeit(
            lambda: self.trace_data_pandas.get_func_names(), number=10
        )
        self.assertLess(time_pandas, time_polars)
        # print the ratio of the two time
        print("The ratio of the two time is: ", time_polars / time_pandas)

    def test_get_func_call_ids(self):
        self.assertEqual(
            self.trace_data_polars.get_func_call_ids(),
            self.trace_data_pandas.get_func_call_ids(),
        )

        # check efficiency of the two implementations
        time_polars = timeit.timeit(
            lambda: self.trace_data_polars.get_func_call_ids(), number=10
        )
        time_pandas = timeit.timeit(
            lambda: self.trace_data_pandas.get_func_call_ids(), number=10
        )
        self.assertLess(time_pandas, time_polars)
        # print the ratio of the two time
        print("The ratio of the two time is: ", time_polars / time_pandas)

    def test_get_var_ids(self):
        self.assertEqual(
            self.trace_data_polars.get_var_ids(), self.trace_data_pandas.get_var_ids()
        )

        # check efficiency of the two implementations
        time_polars = timeit.timeit(
            lambda: self.trace_data_polars.get_var_ids(), number=10
        )
        time_pandas = timeit.timeit(
            lambda: self.trace_data_pandas.get_var_ids(), number=10
        )
        self.assertLess(time_pandas, time_polars)
        # print the ratio of the two time
        print("The ratio of the two time is: ", time_polars / time_pandas)

    def test_get_var_insts(self):
        self.assertEqual(
            self.trace_data_polars.get_var_insts(),
            self.trace_data_pandas.get_var_insts(),
        )

        # check efficiency of the two implementations
        time_polars = timeit.timeit(
            lambda: self.trace_data_polars.get_var_insts(), number=10
        )
        time_pandas = timeit.timeit(
            lambda: self.trace_data_pandas.get_var_insts(), number=10
        )
        self.assertLess(time_pandas, time_polars)
        # print the ratio of the two time
        print("The ratio of the two time is: ", time_polars / time_pandas)

    def test_get_func_is_bpund_method(self):
        func_name_list = self.time_polars.get_func_names()
        for func_name in func_name_list:
            self.assertEqual(
                self.trace_data_polars.get_func_is_bound_method(func_name),
                self.trace_data_pandas.get_func_is_bound_method(func_name),
            )

        # check efficiency of the two implementations
        time_polars = timeit.timeit(
            lambda: self.trace_data_polars.get_func_is_bound_method(func_name_list[0]),
            number=10,
        )
        time_pandas = timeit.timeit(
            lambda: self.trace_data_pandas.get_func_is_bound_method(func_name_list[0]),
            number=10,
        )
        self.assertLess(time_pandas, time_polars)
        # print the ratio of the two time
        print("The ratio of the two time is: ", time_polars / time_pandas)

    def test_get_causally_related_vars(self):
        func_call_id_list = self.trace_data_polars.get_func_call_ids()
        for func_call_id in func_call_id_list:
            self.assertEqual(
                self.trace_data_polars.get_causally_related_vars(func_call_id),
                self.trace_data_pandas.get_causally_related_vars(func_call_id),
            )

        # check efficiency of the two implementations
        time_polars = timeit.timeit(
            lambda: self.trace_data_polars.get_causally_related_vars(
                func_call_id_list[0]
            ),
            number=10,
        )
        time_pandas = timeit.timeit(
            lambda: self.trace_data_pandas.get_causally_related_vars(
                func_call_id_list[0]
            ),
            number=10,
        )
        self.assertLess(time_pandas, time_polars)
        # print the ratio of the two time
        print("The ratio of the two time is: ", time_polars / time_pandas)

    def test_get_var_raw_event_before_time(self):
        var_id_list = self.trace_data_polars.get_var_ids()
        for var_id in var_id_list:
            self.assertEqual(
                self.trace_data_polars.get_var_raw_event_before_time(var_id, 0),
                self.trace_data_pandas.get_var_raw_event_before_time(var_id, 0),
            )

        # check efficiency of the two implementations
        time_polars = timeit.timeit(
            lambda: self.trace_data_polars.get_var_raw_event_before_time(
                var_id_list[0], 0
            ),
            number=10,
        )
        time_pandas = timeit.timeit(
            lambda: self.trace_data_pandas.get_var_raw_event_before_time(
                var_id_list[0], 0
            ),
            number=10,
        )
        self.assertLess(time_pandas, time_polars)
        # print the ratio of the two time
        print("The ratio of the two time is: ", time_polars / time_pandas)

    def test_get_var_changes(self):
        self.assertEqual(
            self.trace_data_polars.get_var_changes(),
            self.trace_data_pandas.get_var_changes(),
        )

        # check efficiency of the two implementations
        time_polars = timeit.timeit(
            lambda: self.trace_data_polars.get_var_changes(), number=10
        )
        time_pandas = timeit.timeit(
            lambda: self.trace_data_pandas.get_var_changes(), number=10
        )
        self.assertLess(time_pandas, time_polars)
        # print the ratio of the two time
        print("The ratio of the two time is: ", time_polars / time_pandas)

    def test_query_var_changes_within_time(self):
        time_range = (self.start_time, self.end_time)
        self.assertEqual(
            self.trace_data_polars.query_var_changes_within_time(time_range),
            self.trace_data_pandas.query_var_changes_within_time(time_range),
        )

        # also need to make sure the result is the same as get_var_changes
        self.assertEqual(
            self.trace_data_polars.query_var_changes_within_time(time_range),
            self.trace_data_polars.get_var_changes(),
        )
        self.assertEqual(
            self.trace_data_pandas.query_var_changes_within_time(time_range),
            self.trace_data_pandas.get_var_changes(),
        )

        # check efficiency of the two implementations
        time_polars = timeit.timeit(
            lambda: self.trace_data_polars.query_var_changes_within_time(time_range),
            number=10,
        )
        time_pandas = timeit.timeit(
            lambda: self.trace_data_pandas.query_var_changes_within_time(time_range),
            number=10,
        )
        self.assertLess(time_pandas, time_polars)

        # print the ratio of the two time
        print("The ratio of the two time is: ", time_polars / time_pandas)

    def test_query_func_calls_within_func_call(self):
        func_call_id_list = self.trace_data_polars.get_func_call_ids()
        for func_call_id in func_call_id_list:
            self.assertEqual(
                self.trace_data_polars.query_func_calls_within_func_call(func_call_id),
                self.trace_data_pandas.query_func_calls_within_func_call(func_call_id),
            )

        # check efficiency of the two implementations
        time_polars = timeit.timeit(
            lambda: self.trace_data_polars.query_func_calls_within_func_call(
                func_call_id_list[0]
            ),
            number=10,
        )
        time_pandas = timeit.timeit(
            lambda: self.trace_data_pandas.query_func_calls_within_func_call(
                func_call_id_list[0]
            ),
            number=10,
        )
        self.assertLess(time_pandas, time_polars)

        # print the ratio of the two time
        print("The ratio of the two time is: ", time_polars / time_pandas)

    def test_get_pre_func_call_record(self):
        func_call_id_list = self.trace_data_polars.get_func_call_ids()
        for func_call_id in func_call_id_list:
            self.assertEqual(
                self.trace_data_polars.get_pre_func_call_record(func_call_id),
                self.trace_data_pandas.get_pre_func_call_record(func_call_id),
            )

        # check efficiency of the two implementations
        time_polars = timeit.timeit(
            lambda: self.trace_data_polars.get_pre_func_call_record(
                func_call_id_list[0]
            ),
            number=10,
        )
        time_pandas = timeit.timeit(
            lambda: self.trace_data_pandas.get_pre_func_call_record(
                func_call_id_list[0]
            ),
            number=10,
        )
        self.assertLess(time_pandas, time_polars)

        # print the ratio of the two time
        print("The ratio of the two time is: ", time_polars / time_pandas)

    def test_get_post_func_call_record(self):
        func_call_id_list = self.trace_data_polars.get_func_call_ids()
        for func_call_id in func_call_id_list:
            self.assertEqual(
                self.trace_data_polars.get_post_func_call_record(func_call_id),
                self.trace_data_pandas.get_post_func_call_record(func_call_id),
            )

        # check efficiency of the two implementations
        time_polars = timeit.timeit(
            lambda: self.trace_data_polars.get_post_func_call_record(
                func_call_id_list[0]
            ),
            number=10,
        )
        time_pandas = timeit.timeit(
            lambda: self.trace_data_pandas.get_post_func_call_record(
                func_call_id_list[0]
            ),
            number=10,
        )
        self.assertLess(time_pandas, time_polars)

        # print the ratio of the two time
        print("The ratio of the two time is: ", time_polars / time_pandas)

    def test_query_var_changes_within_func_call(self):
        func_call_id_list = self.trace_data_polars.get_func_call_ids()
        for func_call_id in func_call_id_list:
            self.assertEqual(
                self.trace_data_polars.query_var_changes_within_func_call(func_call_id),
                self.trace_data_pandas.query_var_changes_within_func_call(func_call_id),
            )

        # check efficiency of the two implementations
        time_polars = timeit.timeit(
            lambda: self.trace_data_polars.query_var_changes_within_func_call(
                func_call_id_list[0]
            ),
            number=10,
        )
        time_pandas = timeit.timeit(
            lambda: self.trace_data_pandas.query_var_changes_within_func_call(
                func_call_id_list[0]
            ),
            number=10,
        )
        self.assertLess(time_pandas, time_polars)

        # print the ratio of the two time
        print("The ratio of the two time is: ", time_polars / time_pandas)

    def test_query_high_level_events_within_func_call(self):
        func_call_id_list = self.trace_data_polars.get_func_call_ids()
        for func_call_id in func_call_id_list:
            self.assertEqual(
                self.trace_data_polars.query_high_level_events_within_func_call(
                    func_call_id
                ),
                self.trace_data_pandas.query_high_level_events_within_func_call(
                    func_call_id
                ),
            )

        # check efficiency of the two implementations
        time_polars = timeit.timeit(
            lambda: self.trace_data_polars.query_high_level_events_within_func_call(
                func_call_id_list[0]
            ),
            number=10,
        )
        time_pandas = timeit.timeit(
            lambda: self.trace_data_pandas.query_high_level_events_within_func_call(
                func_call_id_list[0]
            ),
            number=10,
        )
        self.assertLess(time_pandas, time_polars)

        # print the ratio of the two time
        print("The ratio of the two time is: ", time_polars / time_pandas)

    def test_query_func_call_events_within_time(self):
        time_range = (self.start_time, self.end_time)
        process_id_list = self.get_process_ids()
        thread_id_list = self.get_thread_ids()
        for process_id in process_id_list:
            for thread_id in thread_id_list:
                self.assertEqual(
                    self.trace_data_polars.query_func_call_events_within_time(
                        time_range, process_id, thread_id
                    ),
                    self.trace_data_pandas.query_func_call_events_within_time(
                        time_range, process_id, thread_id
                    ),
                )

        # check efficiency of the two implementations
        time_polars = timeit.timeit(
            lambda: self.trace_data_polars.query_func_call_events_within_time(
                time_range, process_id_list[0], thread_id_list[0]
            ),
            number=10,
        )
        time_pandas = timeit.timeit(
            lambda: self.trace_data_pandas.query_func_call_events_within_time(
                time_range, process_id_list[0], thread_id_list[0]
            ),
            number=10,
        )
        self.assertLess(time_pandas, time_polars)

        # print the ratio of the two time
        print("The ratio of the two time is: ", time_polars / time_pandas)


if __name__ == "__main__":
    unittest.main()
