# This import is necessary to make the observer utility inside torch_proxy.py executed before the instrumented code. This would ensure the observer function is successfully registred before the instrumented code is executed.

import traincheck.proxy_wrapper.proxy_config  # noqa
import traincheck.proxy_wrapper.torch_proxy  # noqa
