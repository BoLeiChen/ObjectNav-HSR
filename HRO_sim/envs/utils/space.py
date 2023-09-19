import gym
import pickle
import msgpack
from collections.abc import Sequence
import torch

class SerializableSpace(gym.Space):
    METHOD_PICKLE = 'pickle'
    METHOD_MSGPACK = 'msgpack'

    def __init__(self, method = METHOD_PICKLE, max_len = 1024) -> None:
        self.method = method
        self.max_len = max_len

    def dumps(self, obj):
        if self.method == self.METHOD_PICKLE:
            return pickle.dumps(obj) 
        elif self.method == self.METHOD_MSGPACK:
            return msgpack.dumps(obj)
    
    def loads(self, byte_pack):
        if self.method == self.METHOD_PICKLE:
            return pickle.loads(byte_pack)
        elif self.method == self.METHOD_MSGPACK:
            return msgpack.loads(byte_pack)

class ObstaclesSpace(SerializableSpace):
    def __init__(self, method=SerializableSpace.METHOD_PICKLE, num_sem_class = 23) -> None:
        super().__init__(method, 65536)
        self.num_sem_class = num_sem_class

    def sample(self):
        return [[] for _ in range(self.num_sem_class)]
    
    def contains(self, x):
        if isinstance(x, Sequence):
            return True
        return False
    
    def dumps(self, raw_obj):
        obj = []
        for sem_class in raw_obj: 
            sem_list = []
            for approx in sem_class: 
                sem_list.append([p.tolist() for p in approx])
            obj.append(sem_list)
        return super().dumps(obj)
    
    def loads(self, byte_pack):
        raw_obj = super().loads(byte_pack)
        obj = []
        for sem_class in raw_obj: 
            sem_list = []
            for approx in sem_class: 
                sem_list.append([torch.Tensor(p).float() for p in approx])
            obj.append(sem_list)
        return obj