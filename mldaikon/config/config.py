TMP_FILE_PREFIX = "_ml_daikon_"
MODULES_TO_INSTRUMENT = ["torch"]
INCLUDED_WRAP_LIST = ["Net", "DataParallel"]  # FIXME: Net & DataParallel seem ad-hoc
proxy_log_dir = "proxy_log.log"  # FIXME: ad-hoc
disable_proxy_class = False # Ziming: Currently disable proxy_class in default
debug_mode = False