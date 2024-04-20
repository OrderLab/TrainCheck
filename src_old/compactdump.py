# compactdump.py

# compress json event and json values dumps via replacing wiht a uuid. 

import json


def dump_json(input: dict):
    to_dump = {}

    for key, val in input.items():
        compressed_key = -1
        compressed_val = -1

        if key not in CompressedDataSingleton.get_key_to_compressed_key_map():
            compressed_key = CompressedDataSingleton.add_compressed_key(key)
        else:
            compressed_key = CompressedDataSingleton.get_key_to_compressed_key_map()[key]

        if compressed_val not in CompressedDataSingleton.get_compressed_key_to_vals_map_vec()[compressed_key]:
            compressed_val = CompressedDataSingleton.add_compressed_val(val)
        else:
            compressed_val = CompressedDataSingleton.get_compressed_key_to_vals_map_vec()[compressed_key][val]

        to_dump[compressed_key] = compressed_val

    return json.dumps(to_dump)

def retrieve_json(input : dict):
    retrieved = {}
    for compressed_key, compressed_val in input.items():
        key = CompressedDataSingleton.get_compressed_key_to_key_vec()[int(compressed_key)]
        val = CompressedDataSingleton.get_compressed_val_to_val_map()[compressed_val]
        retrieved[key] = val

    return retrieved

# HACK: use a singleton to store uuid maps globally across wrappers, issues with concurency?
class CompressedDataSingleton:
    _instance = None
    val_counter = -1

    key_to_compressed_key_map = {}

    compressed_key_to_key_vec = []

    compressed_key_to_vals_map_vec = [] # compressed_key -> {val -> compressed_val}

    compressed_val_to_val_vec = []

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CompressedDataSingleton, cls).__new__(cls)
        return cls._instance

    # pytorch function names are known at compile time, so we might be able to just store a huge map instead. 
    @staticmethod
    def get_and_increment_val_counter():
        CompressedDataSingleton().val_counter += 1
        return CompressedDataSingleton().val_counter
    
    @staticmethod
    def get_key_to_compressed_key_map():
        return CompressedDataSingleton().key_to_compressed_key_map
    
    @staticmethod
    def get_compressed_key_to_key_vec():
        return CompressedDataSingleton().compressed_key_to_key_vec
    
    @staticmethod
    def get_compressed_key_to_vals_map_vec():
        return CompressedDataSingleton().compressed_key_to_vals_map_vec
    
    @staticmethod
    def get_compressed_val_to_val_map():
        return CompressedDataSingleton().compressed_val_to_val_vec
    
    @staticmethod
    def add_compressed_key(key):
        compressed_key = len(CompressedDataSingleton().key_to_compressed_key_map)
        CompressedDataSingleton().key_to_compressed_key_map[key] = compressed_key
        CompressedDataSingleton().compressed_key_to_key_vec.append(key)
        CompressedDataSingleton().compressed_key_to_vals_map_vec.append({})
        return compressed_key
    
    @staticmethod
    def add_compressed_val(val):
        compressed_val = CompressedDataSingleton.get_and_increment_val_counter()
        CompressedDataSingleton().compressed_key_to_vals_map_vec[-1][val] = compressed_val
        CompressedDataSingleton().compressed_val_to_val_vec.append(val)
        return compressed_val
    

