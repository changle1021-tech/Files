import argparse,atexit,json,os,runpy,sys,time
from pathlib import Path
import torch
from vllm.worker.worker import Worker
from vllm.executor.ray_utils import RayWorkerWrapper

class ExternalEventPair:
    def __init__(self):
        import ctypes
        self.ctypes = ctypes
        self.api = ctypes.CDLL("libcuda.so.1")
        self.api.cuEventCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
        self.api.cuEventRecordWithFlags.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint]
        self.api.cuEventQuery.argtypes = [ctypes.c_void_p]
        self.api.cuEventElapsedTime.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_void_p]
        self.start, self.end = ctypes.c_void_p(), ctypes.c_void_p()
        self.check(self.api.cuEventCreate(ctypes.byref(self.start), 0))
        self.check(self.api.cuEventCreate(ctypes.byref(self.end), 0))

    def check(self, code):
        if code:
            raise RuntimeError(f"CUDA driver event API returned {code}")

    def record(self, event):
        stream = self.ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        self.check(self.api.cuEventRecordWithFlags(event, stream, 1))

    def elapsed(self):
        code = self.api.cuEventQuery(self.end)
        if code == 600:
            return None
        self.check(code)
        value = self.ctypes.c_float()
        self.check(self.api.cuEventElapsedTime(self.ctypes.byref(value), self.start, self.end))
        return float(value.value)

class HTTPWorker(Worker):
    def load_model(self):
        super().load_model()
        self.embedding_events, self.model_events, self.calls = {}, {}, []
        import vllm.model_executor.layers.vocab_parallel_embedding as module
        original_reduce = module.tensor_model_parallel_all_reduce
        def reduce(tensor):
            if not torch.cuda.is_current_stream_capturing():
                return original_reduce(tensor)
            pair = ExternalEventPair()
            self.embedding_events[int(tensor.shape[0])] = pair
            pair.record(pair.start)
            result = original_reduce(tensor)
            pair.record(pair.end)
            return result
        module.tensor_model_parallel_all_reduce = reduce
        model = self.model_runner.model
        original_forward = model.forward
        def forward(*args, **kwargs):
            if not torch.cuda.is_current_stream_capturing():
                return original_forward(*args, **kwargs)
            ids = args[0] if args else kwargs['input_ids']
            pair = ExternalEventPair()
            self.model_events[int(ids.shape[0])] = pair
            pair.record(pair.start)
            result = original_forward(*args, **kwargs)
            pair.record(pair.end)
            return result
        model.forward = forward
        original_execute = self.model_runner.execute_model
        def execute(model_input, *args, **kwargs):
            result = original_execute(model_input, *args, **kwargs)
            meta = model_input.attn_metadata
            if self.rank == 0 and meta.num_prefills == 0 and meta.decode_metadata.use_cuda_graph:
                count = int(meta.decode_metadata.block_tables.shape[0])
                if count in self.model_events:
                    self.calls.append({'step':len(self.calls)+1,'batch_size':count,
                        'embedding_allreduce_ms':self.embedding_events[count].elapsed(),
                        'graph_model_ms':self.model_events[count].elapsed()})
            return result
        self.model_runner.execute_model = execute
        if self.rank == 0:
            atexit.register(self.save)

    def save(self):
        Path(os.environ['VIDUR_ENTRY_OUTPUT']).write_text(json.dumps({'rank':0,'calls':self.calls}))

class HTTPRayWrapper(RayWorkerWrapper):
    def __init__(self,*args,**kwargs):
        kwargs['worker_module_name']='vllm_051_http_entry_probe'
        kwargs['worker_class_name']='HTTPWorker'
        super().__init__(*args,**kwargs)

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--diagnostic-tp',type=int,required=True)
    args,rest=parser.parse_known_args()
    import ray
    import vllm.executor.ray_gpu_executor as module
    module.RayWorkerWrapper=HTTPRayWrapper
    ray.init(num_cpus=4,num_gpus=args.diagnostic_tp,object_store_memory=512*1024*1024,
        include_dashboard=False,_temp_dir=f'/tmp/vidur_http_matched_entry_tp{args.diagnostic_tp}')
    sys.argv=['vllm.entrypoints.openai.api_server']+rest
    runpy.run_module('vllm.entrypoints.openai.api_server',run_name='__main__')
