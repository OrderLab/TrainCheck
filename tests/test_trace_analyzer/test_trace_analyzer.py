from traincheck.toolkit.analyze_trace import diff_lists_of_dicts

# Example usage:
list1 = [
    {"name": "Alice", "age": 30, "city": "NYC"},
    {"name": "Bob", "age": 25, "city": "LA"},
    {"name": "Charlie", "age": 35, "city": "Chicago"},
]

list2 = [
    {"name": "Alice", "age": 30, "city": "NYC"},
    {"name": "Dave", "age": 40, "city": "Seattle"},  # New insertion
    {"name": "Bob", "age": 26, "city": "LA"},  # Slightly modified (age changed)
    {"name": "Charlie", "age": 36, "city": "Boston"},  # Moved line, slightly modified
]

diff_lists_of_dicts(list1, list2)
