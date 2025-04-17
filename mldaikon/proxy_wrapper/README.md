# Proxy-Wrapper

## Instrumentation

- import the `Proxy` class from `src.proxy_wrapper.proxy`

`import src.proxy_wrapper.proxy as Proxy`

- wrap the machine learning model with `recurse = True`

`model = Proxy(model, recurse = True)`

- Examples:

As shown in line 140 in `./proxyclass_tracer_result/instrumented_mnist.py`

`model=Proxy.Proxy(model, recurse = True, logdir='log-model-proxy-example.log', log_level=logging.INFO)`

line 99 in `./proxyclass_tracer_result/instrumented_84911.py`

`model_transfer=Proxy.Proxy(model_transfer, recurse = True, "model_transfer-example.log", log_level = logging.INFO)`

Note: Initially, we want to achieve automatic instrumentation via `__new__ wrapper`, wrapping all `torch.nn` modules and add a proxy to the targeted modules inherently. However, it is discovered that this instrumentation would interfere with inherent torch functionality, such as `torch.autograd` behavior.

## Functionality

- Recursive wrapping: wrap every submodule, call function result, attributes of a given Proxy wrapper

- Object identification: a common problem is from Python allocation mechanism. For example, when a new Tensor is created x=Tensor([1,2,3,4]), Python first intialize the Tensor project and points x to the new Tensor. This assignment mechanism hinders identification of the old Tensor object x is pointing to. If we need to know how a tensor is updated, we must establish the relationship between the old object and the new object.

To realize this functionality, we use `inspect.currentframe()` to gather the traceback of the execution line and file name. This helps to trace an object creation & modification at the same **location** in the machine learning framework. Thanks to the largely iterative parameter updating mechanism in the ML execution, this method could effectively trace layer tensor update, layer attribute modification, global configuration setup, etc. with few false negatives and minimum instrumentation effort (only requires a proxy wrapper for the `model` module).

```python
frame = inspect.currentframe()

frame_array = []
while frame:
    frame_array.append((frame.f_code.co_filename, frame.f_lineno))
    frame = frame.f_back
```

However, this method still leads to `multiple wrappings` of the same object given that the same objects could be updated at multiple positions in the code. To merge the 'duplicated' proxy classes, a static analysis could be carried on in advance to identify the execution paths that denote to the same object. (TODO)

- Verbosity: verbosity is controlled and could be easily adjusted by changing the logging.level. By default, the verbosity from sub-module inherits from the super-module.

- Logging file dir: to support easy partition of the tracer, the logging file dir for every module (or submodule) could be specified by the `log_dir` parameter. By default, the logging file dir from sub-module inherits from the super-module.

## Enforced Rules (to make proxy wrapper well-defined)

1. every proxied object should **NOT** be proxied
2. every encounterd `torch.nn.Module` object should be proxied
3. *Almost* every return value of the function call should be proxied (despite the interfaces specified by *us* to unwrap the object, e.g. `def __float__(obj)`)
4. every function parameters, despite `torch.nn.Module` object, should be proxied
5. the lower level `torch.nn.Module` object (which could invoke python built-in type & functions) should be put in the blacklist and should **NOT** be wrapped ever to avoid unexpected behavior.
6. ~~~functions that alters (create/add/delete + deepcopy) the input parameters should be wrapped to a special proxy method to ensure those special parameters are still traced~~~ (not required for now)
(e.g. wrapped `torch.utils.distributed.broadcast` in `proxy.py` as follows):

```python
# Save the original broadcast function
original_broadcast = torch.distributed.broadcast

def broadcast(tensor, src, group=None, async_op=False):
    # Perform the original broadcast operation
    original_broadcast(tensor, src, group, async_op)

    # Wrap the first argument in a Proxy object
    tensor = Proxy(tensor, logdir='proxy_log.log', log_level=logging.INFO)

# Override the broadcast function
torch.distributed.broadcast = broadcast
```

## Engineering challenge

Special care with the wrapping/unwrapping:
1. Do Modification instead of Initialization:
don't creat new object unless its an unmutable type. There are some inherent types in `pytorch` that inherits from dict or list type, `isintance` would pass under this case.

2. ~~~Do not modify list~~~ (Update: this could be well-handled if we take enough care not to pass `Proxy` objects into Python built-in calls):

```python
# Ziming: comment out the dict unwrapping here, it would interfere 
# with the _try_get_data functionality in dataloader 
# elif isinstance(obj, dict):
#     for key in obj:
#         obj[key] = unwrap_proxies(obj[key], level+1)
#     return obj
```

The `_try_get_data` function is responsible for fetching data from the DataLoader's underlying queue. If the data in the queue is wrapped in `Proxy` objects, unwrapping them before `_try_get_data` has a chance to process them might lead to unexpected behavior or errors.


## Output


### Proxy wrapper v2

For now, turn on debug_mode in config.py to see full proxy_trace debug output.

Tracing log for DS-1801:
run `PYTORCH_JIT=0 python -m traincheck.collect_trace -p Megatron-DeepSpeed/pretrain_gpt.py -s ./pretrain_gpt2_codeparrot_short.sh -t megatron deepspeed torch`

```
found the root object
logger_proxy: Proxying all submodules of 'GPTModelPipe'
logger_proxy: Proxying submodule 'tied_modules'
logger_proxy: Proxying submodule 'embed'
logger_proxy: Proxying submodule 'embed'
logger_proxy: Proxying submodule '3'
logger_proxy: Proxying submodule '3'
logger_proxy: Proxying submodule 'input_layernorm'
logger_proxy: Proxying submodule 'self_attention'
logger_proxy: Proxying submodule 'input_layernorm'
logger_proxy: Proxying submodule 'self_attention'
logger_proxy: Proxying submodule 'post_attention_layernorm'
logger_proxy: Proxying submodule 'mlp'
logger_proxy: Proxying submodule 'post_attention_layernorm'
logger_proxy: Proxying submodule 'mlp'
logger_proxy: Proxying submodule '4'
logger_proxy: Proxying submodule '4'
logger_proxy: Proxying submodule 'input_layernorm'
logger_proxy: Proxying submodule 'self_attention'
logger_proxy: Proxying submodule 'input_layernorm'
logger_proxy: Proxying submodule 'self_attention'
logger_proxy: Proxying submodule 'post_attention_layernorm'
logger_proxy: Proxying submodule 'mlp'
logger_proxy: Proxying submodule 'post_attention_layernorm'
logger_proxy: Proxying submodule 'mlp'
logger_proxy: Proxying submodule '5'
logger_proxy: Proxying submodule '5'
logger_proxy: Proxying submodule 'input_layernorm'
logger_proxy: Proxying submodule 'self_attention'
logger_proxy: Proxying submodule 'input_layernorm'
logger_proxy: Proxying submodule 'self_attention'
logger_proxy: Proxying submodule 'post_attention_layernorm'
logger_proxy: Proxying submodule 'mlp'
logger_proxy: Proxying submodule 'post_attention_layernorm'
logger_proxy: Proxying submodule 'mlp'
logger_proxy: Proxying submodule '6'
logger_proxy: Proxying submodule 'embed'
logger_proxy: Proxying submodule '6'
logger_proxy: Proxying submodule 'embed'
logger_proxy: Proxying submodule '3'
logger_proxy: Proxying submodule 'embed'
logger_proxy: Proxying submodule '3'
logger_proxy: Proxying submodule 'input_layernorm'
logger_proxy: Proxying submodule 'self_attention'
logger_proxy: Proxying submodule 'input_layernorm'
logger_proxy: Proxying submodule '3'
logger_proxy: Proxying submodule 'self_attention'
logger_proxy: Proxying submodule 'post_attention_layernorm'
logger_proxy: Proxying submodule 'input_layernorm'
logger_proxy: Proxying submodule 'mlp'
logger_proxy: Proxying submodule 'post_attention_layernorm'
logger_proxy: Proxying submodule 'self_attention'
logger_proxy: Proxying submodule 'input_layernorm'
logger_proxy: Proxying submodule 'mlp'
logger_proxy: Proxying submodule 'self_attention'
logger_proxy: Proxying submodule '8'
logger_proxy: Proxying submodule 'input_layernorm'
logger_proxy: Proxying submodule '8'
logger_proxy: Proxying submodule 'post_attention_layernorm'
logger_proxy: Proxying submodule 'self_attention'
logger_proxy: Proxying submodule 'mlp'
logger_proxy: Proxying submodule 'post_attention_layernorm'
logger_proxy: Proxying submodule 'mlp'
logger_proxy: Proxying submodule '4'
logger_proxy: Proxying submodule 'post_attention_layernorm'
logger_proxy: Proxying submodule 'mlp'
logger_proxy: Proxying submodule '4'
```

### Proxy wrapper v1

Tracing log for `python3 ./proxyclass_tracer_result/instrumented_mnist.py`

```
INFO:root:Accessing attribute 'grad'
INFO:root:Object 'Tensor' is already proxied
INFO:root:Setting attribute '_obj' to an updated tensor with shape'torch.Size([10])'
INFO:root:Minimum value: -0.09099041670560837
INFO:root:Maximum value: 0.07728726416826248
INFO:root:Accessing attribute 'is_sparse'
INFO:root:Accessing attribute 'grad'
INFO:root:Object 'Tensor' is already proxied
INFO:root:Setting attribute '_obj' to an updated tensor with shape'torch.Size([10])'
INFO:root:Minimum value: -0.09099041670560837
INFO:root:Maximum value: 0.07728726416826248
INFO:root:Go to __torch_function__ for function 'is_complex'
INFO:root:Accessing attribute 'mul_'
INFO:root:Object 'builtin_function_or_method' is already proxied
INFO:root:Go to __call__ for object 'builtin_function_or_method'
INFO:root:Object 'Tensor' is already proxied
INFO:root:Setting attribute '_obj' to an updated tensor with shape'torch.Size([32, 1, 3, 3])'
INFO:root:Minimum value: 3.43624378729146e-07
INFO:root:Maximum value: 0.0027863564901053905
```

Tracing log for `python3 ./proxyclass_tracer_result/instrumented_84911.py`

```
INFO:root:Go to __call__ for object 'builtin_function_or_method'
INFO:root:Go to __init__ for object 'Tensor'
INFO:root:Object 'Tensor' is already proxied
INFO:root:Setting attribute '_obj' to an updated tensor with shape'torch.Size([1280])'
INFO:root:Minimum value: 4.3368290789658204e-05
INFO:root:Maximum value: 0.0002152030065190047
INFO:root:Accessing attribute 'addcdiv_'
INFO:root:Go to __init__ for object 'builtin_function_or_method'
INFO:root:Object 'builtin_function_or_method' is already proxied
INFO:root:Go to __call__ for object 'builtin_function_or_method'
INFO:root:Go to __init__ for object 'Parameter'
INFO:root:Object 'Parameter' is already proxied
INFO:root:Accessing attribute 'grad'
INFO:root:Go to __init__ for object 'Tensor'
INFO:root:Object 'Tensor' is already proxied
INFO:root:Setting attribute '_obj' to an updated tensor with shape'torch.Size([1152])'
INFO:root:Minimum value: -0.00257011572830379
INFO:root:Maximum value: 0.0033490750938653946
INFO:root:Setting attribute 'grad' to 'None'
INFO:root:Accessing attribute 'grad'
INFO:root:Go to __init__ for object 'Tensor'
INFO:root:Object 'Tensor' is already proxied
INFO:root:Setting attribute '_obj' to an updated tensor with shape'torch.Size([48, 1152, 1, 1])'
INFO:root:Minimum value: -0.0011981773423030972
INFO:root:Maximum value: 0.001024699304252863
INFO:root:Setting attribute 'grad' to 'None'
INFO:root:Accessing attribute 'grad'
INFO:root:Go to __init__ for object 'Tensor'
INFO:root:Object 'Tensor' is already proxied
INFO:root:Setting attribute '_obj' to an updated tensor with shape'torch.Size([48])'
```