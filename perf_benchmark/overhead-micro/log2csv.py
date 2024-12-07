import pandas as pd

program_output_path = "execution.log"
wrapper_time_prefix = "WRAPPER TIME: "

results: list[tuple[str, float, float]] = []

with open(program_output_path, "r") as f:
    for line in f:
        line = line.strip()
        if wrapper_time_prefix in line:
            result_line = line.split(wrapper_time_prefix)[1]
            api_name, orig_time_str, wrapper_time_str = result_line.split(",")
            orig_time = float(orig_time_str)
            wrapper_time = float(wrapper_time_str)
            results.append((api_name, orig_time, wrapper_time))

# create a df, assign columns
df = pd.DataFrame(results, columns=["API", "Original Time", "Wrapper Time"])
print(df)
df["Wrapper Time Overhead Ratio"] = df["Wrapper Time"] / df["Original Time"]
# df to csv
df.to_csv("wrapper_overhead_micro.csv", index=False)
