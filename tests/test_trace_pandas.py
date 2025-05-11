import cProfile
import timeit
import unittest

# Import the module to test
from traincheck.trace.trace_pandas import TracePandas, read_trace_file_Pandas
from traincheck.trace.trace_polars import TracePolars, read_trace_file_polars

# import modin.pandas as pd
# import polars


def profile(func):
    def wrapper(*args, **kwargs):
        profile = cProfile.Profile()
        profile.enable()
        result = func(*args, **kwargs)
        profile.disable()
        # dump to the file func_name.prof
        profile.dump_stats(f"{func.__name__}.prof")
        return result

    return wrapper


# Differential test: check the output of the two implementations (modin.pandas and polars) are the same
class TestTracePandas(unittest.TestCase):

    def setUp(self):
        self.trace_file = "tests/test_trace_pandas/test_API_trace.log"
        self.trace_data_polars = read_trace_file_polars(self.trace_file)
        self.trace_data_pandas = read_trace_file_Pandas(self.trace_file)
        self.start_time = self.trace_data_pandas.get_start_time()
        self.end_time = self.trace_data_pandas.get_end_time()
        assert isinstance(self.trace_data_polars, TracePolars)
        assert isinstance(self.trace_data_pandas, TracePandas)

    # def test_read_trace_file(self):
    #     # Check the two implementations return the same output
    #     for attr in dir(self.trace_data_polars):
    #         if not attr.startswith("_"):
    #             if isinstance(
    #                 getattr(self.trace_data_polars, attr),
    #                 (pd.DataFrame, polars.dataframe.frame.DataFrame),
    #             ):
    #                 # check if the number of lines and the column keys are the same
    #                 # print the keys of the two dataframes
    #                 print("The keys of the two dataframes are: ")
    #                 print(getattr(self.trace_data_polars, attr).columns)
    #                 print(getattr(self.trace_data_pandas, attr).columns)
    #                 self.assertEqual(
    #                     getattr(self.trace_data_polars, attr).shape,
    #                     getattr(self.trace_data_pandas, attr).shape,
    #                     f"The keys of the two dataframes are: {getattr(self.trace_data_polars, attr).columns} {getattr(self.trace_data_pandas, attr).columns}",
    #                 )
    #             else:
    #                 self.assertEqual(
    #                     getattr(self.trace_data_polars, attr),
    #                     getattr(self.trace_data_pandas, attr),
    #                 )
    #     # Check the efficiency of the two implementations
    #     time_polars = timeit.timeit(lambda: read_trace_file_polars(self.trace_file), number=10)
    #     time_pandas = timeit.timeit(
    #         lambda: read_trace_file_Pandas(self.trace_file), number=10
    #     )
    #     self.assertLess(time_pandas, time_polars)
    #     # print the ratio of the two time
    #     print("The ratio of the two time is: ", time_polars / time_pandas)

    # def test_rm_incomplete_trailing_func_calls(self):
    #     self.trace_data_polars._rm_incomplete_trailing_func_calls()
    #     self.trace_data_pandas._rm_incomplete_trailing_func_calls()
    #     # self.assertEqual(self.trace_data_polars, self.trace_data_pandas)

    #     # check efficiency of the two implementations
    #     time_polars = timeit.timeit(
    #         lambda: self.trace_data_polars._rm_incomplete_trailing_func_calls(),
    #         number=10,
    #     )
    #     time_pandas = timeit.timeit(
    #         lambda: self.trace_data_pandas._rm_incomplete_trailing_func_calls(),
    #         number=10,
    #     )
    #     self.assertLess(time_pandas, time_polars)
    #     # print the ratio of the two time
    #     print("The ratio of the two time is: ", time_polars / time_pandas)

    # def test_get_start_time(self):
    #     self.assertEqual(
    #         self.trace_data_polars.get_start_time(),
    #         self.trace_data_pandas.get_start_time(),
    #     )

    #     # check efficiency of the two implementations
    #     time_polars = timeit.timeit(
    #         lambda: self.trace_data_polars.get_start_time(), number=10
    #     )
    #     time_pandas = timeit.timeit(
    #         lambda: self.trace_data_pandas.get_start_time(), number=10
    #     )
    #     self.assertLess(time_pandas, time_polars)
    #     # print the ratio of the two time
    #     print("The ratio of the two time is: ", time_polars / time_pandas)

    # def test_get_end_time(self):
    #     self.assertEqual(
    #         self.trace_data_polars.get_end_time(), self.trace_data_pandas.get_end_time()
    #     )

    #     # check efficiency of the two implementations
    #     time_polars = timeit.timeit(
    #         lambda: self.trace_data_polars.get_end_time(), number=10
    #     )
    #     time_pandas = timeit.timeit(
    #         lambda: self.trace_data_pandas.get_end_time(), number=10
    #     )
    #     self.assertLess(time_pandas, time_polars)
    #     # print the ratio of the two time
    #     print("The ratio of the two time is: ", time_polars / time_pandas)

    # def test_get_process_ids(self):
    #     self.assertEqual(
    #         set(self.trace_data_polars.get_process_ids()),
    #         set(self.trace_data_pandas.get_process_ids()),
    #     )

    #     # check efficiency of the two implementations
    #     time_polars = timeit.timeit(
    #         lambda: self.trace_data_polars.get_process_ids(), number=10
    #     )
    #     time_pandas = timeit.timeit(
    #         lambda: self.trace_data_pandas.get_process_ids(), number=10
    #     )
    #     self.assertLess(time_pandas, time_polars)
    #     # print the ratio of the two time
    #     print("The ratio of the two time is: ", time_polars / time_pandas)

    # def test_get_thread_ids(self):
    #     self.assertEqual(
    #         set(self.trace_data_polars.get_thread_ids()),
    #         set(self.trace_data_pandas.get_thread_ids()),
    #     )

    #     # check efficiency of the two implementations
    #     time_polars = timeit.timeit(
    #         lambda: self.trace_data_polars.get_thread_ids(), number=10
    #     )
    #     time_pandas = timeit.timeit(
    #         lambda: self.trace_data_pandas.get_thread_ids(), number=10
    #     )
    #     self.assertLess(time_pandas, time_polars)
    #     # print the ratio of the two time
    #     print("The ratio of the two time is: ", time_polars / time_pandas)

    # def test_get_start_time_with_specific_id(self):
    #     # get the last process_id and thread_id
    #     process_id = self.get_process_ids()[-1]
    #     thread_id = self.get_thread_ids()[-1]
    #     self.assertEqual(
    #         self.trace_data_polars.get_start_time(process_id, thread_id),
    #         self.trace_data_pandas.get_start_time(process_id, thread_id),
    #     )

    #     # check efficiency of the two implementations
    #     time_polars = timeit.timeit(
    #         lambda: self.trace_data_polars.get_start_time(process_id, thread_id),
    #         number=10,
    #     )
    #     time_pandas = timeit.timeit(
    #         lambda: self.trace_data_pandas.get_start_time(process_id, thread_id),
    #         number=10,
    #     )
    #     self.assertLess(time_pandas, time_polars)
    #     # print the ratio of the two time
    #     print("The ratio of the two time is: ", time_polars / time_pandas)

    # def test_get_end_time_with_specific_id(self):
    #     # get the first process_id and thread_id
    #     process_id = self.get_process_ids()[0]
    #     thread_id = self.get_thread_ids()[0]
    #     self.assertEqual(
    #         self.trace_data_polars.get_end_time(process_id, thread_id),
    #         self.trace_data_pandas.get_end_time(process_id, thread_id),
    #     )

    #     # check efficiency of the two implementations
    #     time_polars = timeit.timeit(
    #         lambda: self.trace_data_polars.get_end_time(process_id, thread_id),
    #         number=10,
    #     )
    #     time_pandas = timeit.timeit(
    #         lambda: self.trace_data_pandas.get_end_time(process_id, thread_id),
    #         number=10,
    #     )
    #     self.assertLess(time_pandas, time_polars)
    #     # print the ratio of the two time
    #     print("The ratio of the two time is: ", time_polars / time_pandas)

    # def test_get_func_names(self):
    #     self.assertEqual(
    #         set(self.trace_data_polars.get_func_names()),
    #         set(self.trace_data_pandas.get_func_names()),
    #     )

    #     # check efficiency of the two implementations
    #     time_polars = timeit.timeit(
    #         lambda: self.trace_data_polars.get_func_names(), number=10
    #     )
    #     time_pandas = timeit.timeit(
    #         lambda: self.trace_data_pandas.get_func_names(), number=10
    #     )
    #     self.assertLess(time_pandas, time_polars)
    #     # print the ratio of the two time
    #     print("The ratio of the two time is: ", time_polars / time_pandas)

    def test_get_func_call_ids(self):
        # self.assertEqual(
        #     set(self.trace_data_polars.get_func_call_ids()),
        #     set(self.trace_data_pandas.get_func_call_ids()),
        #     f"diff(polars-pandas): {set(self.trace_data_polars.get_func_call_ids()) - set(self.trace_data_pandas.get_func_call_ids())} diff(pandas-polars): {set(self.trace_data_pandas.get_func_call_ids()) - set(self.trace_data_polars.get_func_call_ids())}",
        # )
        @profile
        def get_func_call_ids():
            self.trace_data_polars.get_func_call_ids()

        @profile
        def get_func_call_ids_pandas():
            self.trace_data_pandas.get_func_call_ids()

        get_func_call_ids()
        get_func_call_ids_pandas()

    # def test_get_var_ids(self):
    #     self.assertEqual(
    #         self.trace_data_polars.get_var_ids(), self.trace_data_pandas.get_var_ids()
    #     )

    #     # check efficiency of the two implementations
    #     time_polars = timeit.timeit(
    #         lambda: self.trace_data_polars.get_var_ids(), number=10
    #     )
    #     time_pandas = timeit.timeit(
    #         lambda: self.trace_data_pandas.get_var_ids(), number=10
    #     )
    #     self.assertLess(time_pandas, time_polars)
    #     # print the ratio of the two time
    #     print("The ratio of the two time is: ", time_polars / time_pandas)

    # def test_get_var_insts(self):
    #     self.assertEqual(
    #         self.trace_data_polars.get_var_insts(),
    #         self.trace_data_pandas.get_var_insts(),
    #     )

    #     # check efficiency of the two implementations
    #     time_polars = timeit.timeit(
    #         lambda: self.trace_data_polars.get_var_insts(), number=10
    #     )
    #     time_pandas = timeit.timeit(
    #         lambda: self.trace_data_pandas.get_var_insts(), number=10
    #     )
    #     self.assertLess(time_pandas, time_polars)
    #     # print the ratio of the two time
    #     print("The ratio of the two time is: ", time_polars / time_pandas)

    def test_get_func_is_bound_method(self):
        func_name_list = self.trace_data_polars.get_func_names()
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

    # @unittest.skip(
    #     "Would trigger AssertionError: Causal relation extraction is only supported for bound methods"
    # )
    # def test_get_causally_related_vars(self):
    #     func_call_id_list = self.trace_data_polars.get_func_call_ids()
    #     for func_call_id in func_call_id_list:
    #         try:
    #             result_polars = self.trace_data_polars.get_causally_related_vars(
    #                 func_call_id
    #             )
    #             result_pandas = self.trace_data_pandas.get_causally_related_vars(
    #                 func_call_id
    #             )
    #         except AssertionError as e:
    #             print(e)
    #             print("func_call_id: ", func_call_id)
    #             continue
    #         self.assertEqual(
    #             result_polars,
    #             result_pandas,
    #         )

    #     # check efficiency of the two implementations
    #     time_polars = timeit.timeit(
    #         lambda: self.trace_data_polars.get_causally_related_vars(
    #             func_call_id_list[0]
    #         ),
    #         number=10,
    #     )
    #     time_pandas = timeit.timeit(
    #         lambda: self.trace_data_pandas.get_causally_related_vars(
    #             func_call_id_list[0]
    #         ),
    #         number=10,
    #     )
    #     self.assertLess(time_pandas, time_polars)
    #     # print the ratio of the two time
    #     print("The ratio of the two time is: ", time_polars / time_pandas)

    # def test_get_var_raw_event_before_time(self):
    #     var_id_list = self.trace_data_polars.get_var_ids()
    #     for var_id in var_id_list:
    #         self.assertEqual(
    #             self.trace_data_polars.get_var_raw_event_before_time(var_id, 0),
    #             self.trace_data_pandas.get_var_raw_event_before_time(var_id, 0),
    #         )
    #     if len(var_id_list):
    #         # check efficiency of the two implementations
    #         time_polars = timeit.timeit(
    #             lambda: self.trace_data_polars.get_var_raw_event_before_time(
    #                 var_id_list[0], 0
    #             ),
    #             number=10,
    #         )
    #         time_pandas = timeit.timeit(
    #             lambda: self.trace_data_pandas.get_var_raw_event_before_time(
    #                 var_id_list[0], 0
    #             ),
    #             number=10,
    #         )
    #         self.assertLess(time_pandas, time_polars)
    #         # print the ratio of the two time
    #         print("The ratio of the two time is: ", time_polars / time_pandas)

    # def test_get_var_changes(self):
    #     self.assertEqual(
    #         self.trace_data_polars.get_var_changes(),
    #         self.trace_data_pandas.get_var_changes(),
    #     )

    #     # check efficiency of the two implementations
    #     time_polars = timeit.timeit(
    #         lambda: self.trace_data_polars.get_var_changes(), number=10
    #     )
    #     time_pandas = timeit.timeit(
    #         lambda: self.trace_data_pandas.get_var_changes(), number=10
    #     )
    #     self.assertLess(time_pandas, time_polars)
    #     # print the ratio of the two time
    #     print("The ratio of the two time is: ", time_polars / time_pandas)

    # def test_query_var_changes_within_time(self):
    #     time_range = (self.start_time, self.end_time)
    #     self.assertEqual(
    #         self.trace_data_polars.query_var_changes_within_time(time_range),
    #         self.trace_data_pandas.query_var_changes_within_time(time_range),
    #     )

    #     # also need to make sure the result is the same as get_var_changes
    #     self.assertEqual(
    #         self.trace_data_polars.query_var_changes_within_time(time_range),
    #         self.trace_data_polars.get_var_changes(),
    #     )
    #     self.assertEqual(
    #         self.trace_data_pandas.query_var_changes_within_time(time_range),
    #         self.trace_data_pandas.get_var_changes(),
    #     )

    #     # check efficiency of the two implementations
    #     time_polars = timeit.timeit(
    #         lambda: self.trace_data_polars.query_var_changes_within_time(time_range),
    #         number=10,
    #     )
    #     time_pandas = timeit.timeit(
    #         lambda: self.trace_data_pandas.query_var_changes_within_time(time_range),
    #         number=10,
    #     )
    #     self.assertLess(time_pandas, time_polars)

    #     # print the ratio of the two time
    #     print("The ratio of the two time is: ", time_polars / time_pandas)

    # # def test_query_func_calls_within_func_call(self):
    # #     func_call_id_list = self.trace_data_polars.get_func_call_ids()
    # #     for func_call_id in func_call_id_list:
    # #         try:
    # #             result_polars = set(
    # #                 self.trace_data_polars.query_var_changes_within_func_call(
    # #                     func_call_id
    # #                 )
    # #             )
    # #             result_pandas = set(
    # #                 self.trace_data_pandas.query_var_changes_within_func_call(
    # #                     func_call_id
    # #                 )
    # #             )
    # #         except AssertionError as e:
    # #             print(e)
    # #             print("func_call_id: ", func_call_id)
    # #             continue
    # #         self.assertEqual(
    # #             result_polars,
    # #             result_pandas,
    # #         )

    # #     # check efficiency of the two implementations
    # #     time_polars = timeit.timeit(
    # #         lambda: self.trace_data_polars.query_func_calls_within_func_call(
    # #             func_call_id_list[0]
    # #         ),
    # #         number=10,
    # #     )
    # #     time_pandas = timeit.timeit(
    # #         lambda: self.trace_data_pandas.query_func_calls_within_func_call(
    # #             func_call_id_list[0]
    # #         ),
    # #         number=10,
    # #     )
    # #     self.assertLess(time_pandas, time_polars)

    # #     # print the ratio of the two time
    # #     print("The ratio of the two time is: ", time_polars / time_pandas)

    # def test_get_pre_func_call_record(self):
    #     func_call_id_list = self.trace_data_polars.get_func_call_ids()
    #     for func_call_id in func_call_id_list:
    #         try:
    #             result_polars = self.trace_data_polars.get_pre_func_call_record(
    #                 func_call_id
    #             )
    #             result_pandas = self.trace_data_pandas.get_pre_func_call_record(
    #                 func_call_id
    #             )
    #         except AssertionError as e:
    #             print(e)
    #             print("func_call_id: ", func_call_id)
    #             continue
    #         # convert all nan values to None

    #         self.assertEqual(
    #             result_polars,
    #             result_pandas,
    #         )

    #     # check efficiency of the two implementations
    #     time_polars = timeit.timeit(
    #         lambda: self.trace_data_polars.get_pre_func_call_record(
    #             func_call_id_list[0]
    #         ),
    #         number=10,
    #     )
    #     time_pandas = timeit.timeit(
    #         lambda: self.trace_data_pandas.get_pre_func_call_record(
    #             func_call_id_list[0]
    #         ),
    #         number=10,
    #     )
    #     self.assertLess(time_pandas, time_polars)

    #     # print the ratio of the two time
    #     print("The ratio of the two time is: ", time_polars / time_pandas)

    # def test_get_post_func_call_record(self):
    #     func_call_id_list = self.trace_data_polars.get_func_call_ids()
    #     for func_call_id in func_call_id_list:
    #         try:
    #             result_polars = self.trace_data_polars.get_post_func_call_record(
    #                 func_call_id
    #             )
    #             result_pandas = self.trace_data_pandas.get_post_func_call_record(
    #                 func_call_id
    #             )
    #         except AssertionError as e:
    #             print(e)
    #             print("func_call_id: ", func_call_id)
    #             continue
    #         self.assertEqual(
    #             result_polars,
    #             result_pandas,
    #         )

    #     # check efficiency of the two implementations
    #     time_polars = timeit.timeit(
    #         lambda: self.trace_data_polars.get_post_func_call_record(
    #             func_call_id_list[0]
    #         ),
    #         number=10,
    #     )
    #     time_pandas = timeit.timeit(
    #         lambda: self.trace_data_pandas.get_post_func_call_record(
    #             func_call_id_list[0]
    #         ),
    #         number=10,
    #     )
    #     self.assertLess(time_pandas, time_polars)

    #     # print the ratio of the two time
    #     print("The ratio of the two time is: ", time_polars / time_pandas)

    # def test_query_var_changes_within_func_call(self):
    #     func_call_id_list = self.trace_data_polars.get_func_call_ids()
    #     for func_call_id in func_call_id_list:
    #         try:
    #             result_polars = (
    #                 self.trace_data_polars.query_var_changes_within_func_call(
    #                     func_call_id
    #                 )
    #             )
    #             result_pandas = (
    #                 self.trace_data_pandas.query_var_changes_within_func_call(
    #                     func_call_id
    #                 )
    #             )
    #         except AssertionError as e:
    #             print(e)
    #             print("func_call_id: ", func_call_id)
    #             continue

    #         self.assertEqual(
    #             result_polars,
    #             result_pandas,
    #         )

    #     # check efficiency of the two implementations
    #     time_polars = timeit.timeit(
    #         lambda: self.trace_data_polars.query_var_changes_within_func_call(
    #             func_call_id_list[0]
    #         ),
    #         number=10,
    #     )
    #     time_pandas = timeit.timeit(
    #         lambda: self.trace_data_pandas.query_var_changes_within_func_call(
    #             func_call_id_list[0]
    #         ),
    #         number=10,
    #     )
    #     self.assertLess(time_pandas, time_polars)

    #     # print the ratio of the two time
    #     print("The ratio of the two time is: ", time_polars / time_pandas)

    # def test_query_high_level_events_within_func_call(self):
    #     func_call_id_list = self.trace_data_polars.get_func_call_ids()
    #     for func_call_id in func_call_id_list:
    #         self.assertEqual(
    #             self.trace_data_polars.query_high_level_events_within_func_call(
    #                 func_call_id
    #             ),
    #             self.trace_data_pandas.query_high_level_events_within_func_call(
    #                 func_call_id
    #             ),
    #         )

    #     # check efficiency of the two implementations
    #     time_polars = timeit.timeit(
    #         lambda: self.trace_data_polars.query_high_level_events_within_func_call(
    #             func_call_id_list[0]
    #         ),
    #         number=10,
    #     )
    #     time_pandas = timeit.timeit(
    #         lambda: self.trace_data_pandas.query_high_level_events_within_func_call(
    #             func_call_id_list[0]
    #         ),
    #         number=10,
    #     )
    #     self.assertLess(time_pandas, time_polars)

    #     # print the ratio of the two time
    #     print("The ratio of the two time is: ", time_polars / time_pandas)

    # def test_query_func_call_events_within_time(self):
    #     time_range = (self.start_time, self.end_time)
    #     process_id_list = self.trace_data_pandas.get_process_ids()
    #     thread_id_list = self.trace_data_polars.get_thread_ids()
    #     for process_id in process_id_list:
    #         for thread_id in thread_id_list:
    #             self.assertEqual(
    #                 self.trace_data_polars.query_func_call_events_within_time(
    #                     time_range, process_id, thread_id
    #                 ),
    #                 self.trace_data_pandas.query_func_call_events_within_time(
    #                     time_range, process_id, thread_id
    #                 ),
    #             )

    #     # check efficiency of the two implementations
    #     time_polars = timeit.timeit(
    #         lambda: self.trace_data_polars.query_func_call_events_within_time(
    #             time_range, process_id_list[0], thread_id_list[0]
    #         ),
    #         number=10,
    #     )
    #     time_pandas = timeit.timeit(
    #         lambda: self.trace_data_pandas.query_func_call_events_within_time(
    #             time_range, process_id_list[0], thread_id_list[0]
    #         ),
    #         number=10,
    #     )
    #     self.assertLess(time_pandas, time_polars)

    #     # print the ratio of the two time
    #     print("The ratio of the two time is: ", time_polars / time_pandas)


if __name__ == "__main__":
    unittest.main()
