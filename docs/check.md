# TrainCheck Checker Usage Guide

`traincheck-check` is the **final stage** of the TrainCheck workflow. It verifies a set of invariants against trace files or streams from target programs, reporting any detected violations—helping you catch silent issues in your ML training pipelines.

## 🔧 Current Status

`traincheck-check` is designed to support two modes:

- **Offline Checking**:  
   Perform invariant checking on completed trace files after the training job finishes. ✅ *[Fully Supported]*

- **Online Checking**:  
   Perform real-time checking while the target training job is running. 🚧 *[In Development]*

At present, only **offline checking** is available. Support for online mode is actively being developed.

## How to Use: Offline Checking

Run the following command:

```bash
traincheck-check -f <trace_folder> -i <path_to_invariant_file>
```

- `-f <trace_folder>`: Path to the folder containing traces collected by `traincheck-collect`.
- `-i <path_to_invariant_file>`: Path to the JSON file containing inferred invariants.

For details on result format and interpretation, refer to [5. Detection & Diagnosis)](./5-min-tutorial.md#5-detection--diagnosis) in the **5-Minute Tutorial**.

## How to Use: Online Checking

**🚧 Coming Soon**
Support for real-time, online checking is under construction. This mode will allow TrainCheck to monitor running training jobs and surface invariant violations as they happen.

Stay tuned for updates in future releases.
