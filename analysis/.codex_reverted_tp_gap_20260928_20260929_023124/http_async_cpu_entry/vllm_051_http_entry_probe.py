import argparse,atexit,gc,json,os,random,runpy,sys,time
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

    def eager_record(self, event):
        stream = self.ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        self.check(self.api.cuEventRecordWithFlags(event, stream, 0))

    def elapsed(self):
        return self.elapsed_between(self.start, self.end)

    def elapsed_between(self, start_event, end_event):
        code = self.api.cuEventQuery(end_event)
        if code == 600:
            return None
        self.check(code)
        value = self.ctypes.c_float()
        self.check(self.api.cuEventElapsedTime(
            self.ctypes.byref(value), start_event, end_event))
        return float(value.value)

class HTTPWorker(Worker):
    def load_model(self):
        super().load_model()
        self.embedding_events, self.model_events, self.calls = {}, {}, []
        self.gc_events, self._gc_active = [], {}
        self._active_decode = False
        self._step_regions = {}
        self._pending_prepare_ms = []
        self._pending_broadcast_ms = []
        self._pending_broadcast_event_calls = 0
        self._pending_broadcast_calls = 0
        self._broadcast_event_patterns = {}
        self._decode_metadata_cache_checks = []
        self._broadcast_metadata_cache_checks = []
        self._decode_cache_state_for_step = None
        for group in range(5 + 8):
            pattern = [True] * 24 + [False] * 25
            random.Random(36 + group).shuffle(pattern)
            self._broadcast_event_patterns[group] = pattern
        self.prepare_model_input_calls = []
        self.broadcast_tensor_dict_calls = []
        self.execute_input_timings = []
        self.logits_event = self.sampling_event = None
        if self.rank == 0:
            gc.callbacks.append(self._record_gc_event)
            self.logits_event = ExternalEventPair()
            self.sampling_event = ExternalEventPair()
            self.broadcast_event = ExternalEventPair()
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

        if self.rank == 0:
            self._wrap_stage_region(model, "compute_logits", "logits", self.logits_event)
            self._wrap_stage_region(model, "sample", "sampling", self.sampling_event)

            runner = self.model_runner
            original_prepare = runner.prepare_model_input
            def prepare_model_input(*args, **kwargs):
                start_ns = time.perf_counter_ns()
                try:
                    model_input = original_prepare(*args, **kwargs)
                    meta = getattr(model_input, 'attn_metadata', None)
                    cache_none_before = (
                        getattr(meta, '_cached_decode_metadata', None) is None
                        if meta is not None else True
                    )
                    self._active_decode = bool(
                        meta is not None
                        and meta.num_prefills == 0
                        and meta.num_decode_tokens > 0
                        and meta.use_cuda_graph
                    )
                    cache_none_after = (
                        getattr(meta, '_cached_decode_metadata', None) is None
                        if meta is not None else True
                    )
                    cache_check = {
                        'check_index': len(self._decode_metadata_cache_checks) + 1,
                        'cache_none_before': bool(cache_none_before),
                        'cache_none_after': bool(cache_none_after),
                    }
                    self._decode_metadata_cache_checks.append(cache_check)
                    self._decode_cache_state_for_step = cache_check
                    if not cache_none_before:
                        raise RuntimeError(
                            'Expected _cached_decode_metadata to be None before safe decode flag checks'
                        )
                    if cache_none_after != cache_none_before:
                        raise RuntimeError(
                            'Safe decode flag checks changed _cached_decode_metadata state'
                        )
                    return model_input
                finally:
                    elapsed_ms = (time.perf_counter_ns() - start_ns) / 1e6
                    self._pending_prepare_ms.append(elapsed_ms)
                    self.prepare_model_input_calls.append(elapsed_ms)
            runner.prepare_model_input = prepare_model_input

            from vllm.distributed.parallel_state import get_tp_group
            tp_group = get_tp_group()
            original_broadcast = tp_group.broadcast_tensor_dict
            def broadcast_tensor_dict(*args, **kwargs):
                tensor_dict = args[0] if args else kwargs.get('tensor_dict')
                if not isinstance(tensor_dict, dict):
                    raise RuntimeError(
                        'Expected rank0 broadcast_tensor_dict argument to be a dict'
                    )
                cache_none = tensor_dict.get('_cached_decode_metadata') is None
                cache_check = {
                    'check_index': len(self._broadcast_metadata_cache_checks) + 1,
                    'cache_none': bool(cache_none),
                }
                self._broadcast_metadata_cache_checks.append(cache_check)
                if not cache_none:
                    raise RuntimeError(
                        'broadcast_tensor_dict received non-None _cached_decode_metadata'
                    )
                group, offset = divmod(len(self.calls), 49)
                events_enabled = bool(
                    self._active_decode
                    and group >= 5 and group in self._broadcast_event_patterns
                    and self._broadcast_event_patterns[group][offset]
                )
                self._pending_broadcast_calls += 1
                if events_enabled:
                    self.broadcast_event.eager_record(self.broadcast_event.start)
                    self._pending_broadcast_event_calls += 1
                start_ns = time.perf_counter_ns()
                try:
                    return original_broadcast(*args, **kwargs)
                finally:
                    elapsed_ms = (time.perf_counter_ns() - start_ns) / 1e6
                    if events_enabled:
                        self.broadcast_event.eager_record(self.broadcast_event.end)
                    self._pending_broadcast_ms.append(elapsed_ms)
                    self.broadcast_tensor_dict_calls.append(elapsed_ms)
            tp_group.broadcast_tensor_dict = broadcast_tensor_dict

        original_execute = self.model_runner.execute_model
        def execute(model_input, *args, **kwargs):
            meta = model_input.attn_metadata
            is_decode = bool(
                self.rank == 0
                and meta is not None
                and meta.num_prefills == 0
                and meta.decode_metadata.use_cuda_graph
            )
            self._step_regions = {}
            self._active_decode = is_decode
            host_start_ns = time.perf_counter_ns()
            try:
                result = original_execute(model_input, *args, **kwargs)
                host_end_ns = time.perf_counter_ns()
                pending_prepare = self._pending_prepare_ms
                pending_broadcast = self._pending_broadcast_ms
                pending_broadcast_event_calls = self._pending_broadcast_event_calls
                pending_broadcast_calls = self._pending_broadcast_calls
                self._pending_prepare_ms = []
                self._pending_broadcast_ms = []
                self._pending_broadcast_event_calls = 0
                self._pending_broadcast_calls = 0
                self.execute_input_timings.append({
                    'execute_index': len(self.execute_input_timings) + 1,
                    'pure_decode_cuda_graph': is_decode,
                    'prepare_model_input_ms': pending_prepare,
                    'broadcast_tensor_dict_ms': pending_broadcast,
                    'broadcast_event_calls': pending_broadcast_event_calls,
                    'broadcast_calls': pending_broadcast_calls,
                })
                if is_decode:
                    count = int(meta.decode_metadata.block_tables.shape[0])
                    if count not in self.model_events:
                        raise RuntimeError('Decode graph model event was not recorded')
                    for region in ('logits', 'sampling'):
                        measured = self._step_regions.get(region)
                        if not measured or measured.get('calls') != 1:
                            raise RuntimeError(
                                f'Expected exactly one {region} region in pure decode; '
                                f"got {None if measured is None else measured.get('calls')}"
                            )
                    group, offset = divmod(len(self.calls), 49)
                    events_enabled = bool(
                        self._active_decode
                        and group >= 5 and group in self._broadcast_event_patterns
                        and self._broadcast_event_patterns[group][offset]
                    )
                    expected_event_calls = 1 if events_enabled else 0
                    if pending_broadcast_calls != 1 or pending_broadcast_event_calls != expected_event_calls:
                        raise RuntimeError(
                            'Unexpected broadcast call/event counts in pure decode: '
                            f'calls={pending_broadcast_calls}, events={pending_broadcast_event_calls}, '
                            f'expected_events={expected_event_calls}'
                        )
                    logits_ms = self.logits_event.elapsed()
                    sampling_ms = self.sampling_event.elapsed()
                    embedding_ms = self.embedding_events[count].elapsed()
                    graph_ms = self.model_events[count].elapsed()
                    broadcast_stream_ms = None
                    pre_graph_queued_ms = None
                    if events_enabled:
                        broadcast_stream_ms = self.broadcast_event.elapsed()
                        pre_graph_queued_ms = self.broadcast_event.elapsed_between(
                            self.broadcast_event.end, self.model_events[count].start)
                    required_events = (logits_ms, sampling_ms, embedding_ms, graph_ms)
                    if any(value is None for value in required_events):
                        raise RuntimeError('A CUDA event was not complete after execute_model')
                    if events_enabled and (broadcast_stream_ms is None or pre_graph_queued_ms is None):
                        raise RuntimeError('An enabled broadcast CUDA event was not complete')
                    if pre_graph_queued_ms is not None and pre_graph_queued_ms < 0:
                        raise RuntimeError(
                            f'Negative pre-graph queued interval: {pre_graph_queued_ms} ms'
                        )
                    self.calls.append({
                        'step': len(self.calls) + 1,
                        'batch_size': count,
                        'embedding_allreduce_ms': embedding_ms,
                        'graph_model_ms': graph_ms,
                        'host_start_ns': host_start_ns,
                        'host_end_ns': host_end_ns,
                        'runner_host_ms': (host_end_ns - host_start_ns) / 1e6,
                        'logits_host_ms': self._step_regions['logits']['host_ms'],
                        'sampling_host_ms': self._step_regions['sampling']['host_ms'],
                        'logits_stream_ms': logits_ms,
                        'sampling_stream_ms': sampling_ms,
                        'prepare_model_input_ms': pending_prepare,
                        'prepare_model_input_total_ms': sum(pending_prepare),
                        'broadcast_tensor_dict_ms': pending_broadcast,
                        'broadcast_tensor_dict_total_ms': sum(pending_broadcast),
                        'broadcast_event_calls': pending_broadcast_event_calls,
                        'broadcast_calls': pending_broadcast_calls,
                        'broadcast_events_enabled': events_enabled,
                        'broadcast_metadata_cache_none': (
                            self._broadcast_metadata_cache_checks[-1]['cache_none']
                            if self._broadcast_metadata_cache_checks else None
                        ),
                        'decode_metadata_cache_none_before': (
                            self._decode_cache_state_for_step['cache_none_before']
                            if self._decode_cache_state_for_step else None
                        ),
                        'decode_metadata_cache_none_after': (
                            self._decode_cache_state_for_step['cache_none_after']
                            if self._decode_cache_state_for_step else None
                        ),
                        'broadcast_stream_ms': broadcast_stream_ms,
                        'pre_graph_queued_ms': pre_graph_queued_ms,
                    })
                return result
            finally:
                # Consume/reset pending timings for every runner execution,
                # including prefill, so values cannot leak into another step.
                if (self._pending_prepare_ms or self._pending_broadcast_ms
                        or self._pending_broadcast_event_calls
                        or self._pending_broadcast_calls):
                    self.execute_input_timings.append({
                        'execute_index': len(self.execute_input_timings) + 1,
                        'pure_decode_cuda_graph': is_decode,
                        'prepare_model_input_ms': self._pending_prepare_ms,
                        'broadcast_tensor_dict_ms': self._pending_broadcast_ms,
                        'broadcast_event_calls': self._pending_broadcast_event_calls,
                        'broadcast_calls': self._pending_broadcast_calls,
                    })
                    self._pending_prepare_ms = []
                    self._pending_broadcast_ms = []
                    self._pending_broadcast_event_calls = 0
                    self._pending_broadcast_calls = 0
                self._active_decode = False
        self.model_runner.execute_model = execute
        if self.rank == 0:
            original_worker_execute = self.execute_model
            def worker_execute(*args, **kwargs):
                self._active_decode = False
                before_calls = len(self.calls)
                worker_start_ns = time.perf_counter_ns()
                try:
                    return original_worker_execute(*args, **kwargs)
                finally:
                    worker_end_ns = time.perf_counter_ns()
                    added = len(self.calls) - before_calls
                    if added == 1:
                        row = self.calls[-1]
                        row['worker_start_ns'] = worker_start_ns
                        row['worker_end_ns'] = worker_end_ns
                        row['worker_host_e2e_ms'] = (worker_end_ns - worker_start_ns) / 1e6
                    elif added > 1:
                        raise RuntimeError(f'Worker execute_model added {added} decode rows')
                    self._active_decode = False
            self.execute_model = worker_execute
            self._install_async_cpu_timers()
            atexit.register(self.save)

    def _install_async_cpu_timers(self):
        # These wrappers use CPU clocks only. They never synchronize a CUDA
        # stream or inspect lazy GPU metadata. PP=1 has one engine step active.
        import vllm.engine.async_llm_engine as ae
        from vllm.engine.llm_engine import LLMEngine
        from vllm.core.scheduler import Scheduler
        from vllm.executor.ray_gpu_executor import RayGPUExecutorAsync
        self._cpu_context = None

        def record(label, start, end):
            context = self._cpu_context
            if context is not None:
                context.setdefault(label, []).append({
                    'start_ns': start, 'end_ns': end, 'ms': (end-start)/1e6})

        def wrap_sync(cls, name, label):
            original = getattr(cls, name)
            def wrapped(instance, *args, **kwargs):
                start = time.perf_counter_ns()
                try:
                    return original(instance, *args, **kwargs)
                finally:
                    record(label, start, time.perf_counter_ns())
            setattr(cls, name, wrapped)

        wrap_sync(Scheduler, 'schedule', 'schedule')
        wrap_sync(LLMEngine, '_process_model_outputs', 'process_outputs')
        original_executor = RayGPUExecutorAsync.execute_model_async
        async def executor(instance, *args, **kwargs):
            start = time.perf_counter_ns()
            try:
                return await original_executor(instance, *args, **kwargs)
            finally:
                record('async_executor', start, time.perf_counter_ns())
        RayGPUExecutorAsync.execute_model_async = executor

        original_step = ae._AsyncLLMEngine.step_async
        async def step(instance, *args, **kwargs):
            if self._cpu_context is not None:
                raise RuntimeError('Overlapping engine steps in PP=1 probe')
            before = len(self.calls)
            start = time.perf_counter_ns()
            context = {}; self._cpu_context = context
            try:
                return await original_step(instance, *args, **kwargs)
            finally:
                end = time.perf_counter_ns()
                self._cpu_context = None
                added = len(self.calls)-before
                if added > 1:
                    raise RuntimeError('More than one decode row in engine step')
                if added == 1:
                    for label in ('schedule', 'process_outputs', 'async_executor'):
                        if len(context.get(label, [])) != 1:
                            raise RuntimeError(f'Expected one {label}: {context}')
                    row = self.calls[-1]
                    row['async_cpu_regions'] = context
                    row['engine_step_start_ns'] = start
                    row['engine_step_end_ns'] = end
                    row['engine_step_host_ms'] = (end-start)/1e6
                    for label, entries in context.items():
                        row[label+'_host_ms'] = sum(x['ms'] for x in entries)
        ae._AsyncLLMEngine.step_async = step

    def _wrap_stage_region(self, model, method_name, label, event_pair):
        original = getattr(model, method_name)
        def wrapped(*args, **kwargs):
            if not self._active_decode:
                return original(*args, **kwargs)
            region = self._step_regions.setdefault(label, {'calls': 0})
            region['calls'] += 1
            event_pair.eager_record(event_pair.start)
            host_start_ns = time.perf_counter_ns()
            try:
                return original(*args, **kwargs)
            finally:
                host_end_ns = time.perf_counter_ns()
                event_pair.eager_record(event_pair.end)
                region['host_ms'] = (host_end_ns - host_start_ns) / 1e6
        setattr(model, method_name, wrapped)

    def _record_gc_event(self, phase, info):
        now_ns = time.perf_counter_ns()
        generation = int(info.get('generation', -1))
        if phase == 'start':
            self._gc_active[generation] = now_ns
        elif phase == 'stop':
            start_ns = self._gc_active.pop(generation, None)
            if start_ns is not None:
                duration_ns = now_ns - start_ns
                self.gc_events.append({
                    'start_perf_ns': start_ns,
                    'end_perf_ns': now_ns,
                    'duration_ns': duration_ns,
                    'duration_ms': duration_ns / 1e6,
                    'generation': generation,
                    'collected': int(info.get('collected', 0)),
                    'uncollectable': int(info.get('uncollectable', 0)),
                })

    def save(self):
        Path(os.environ['VIDUR_ASYNC_CPU_OUTPUT']).write_text(json.dumps(
            {
                'rank': 0,
                'calls': self.calls,
                'gc_events': self.gc_events,
                'stage_fields': {
                    'execute_input_timings': self.execute_input_timings,
                    'prepare_model_input_calls_ms': self.prepare_model_input_calls,
                    'broadcast_tensor_dict_calls_ms': self.broadcast_tensor_dict_calls,
                    'decode_metadata_cache_checks': self._decode_metadata_cache_checks,
                    'broadcast_metadata_cache_checks': self._broadcast_metadata_cache_checks,
                },
            }, indent=2
        ))

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
        include_dashboard=False,_temp_dir=f'/tmp/vidur_http_async_cpu_entry_tp{args.diagnostic_tp}')
    sys.argv=['vllm.entrypoints.openai.api_server']+rest
    runpy.run_module('vllm.entrypoints.openai.api_server',run_name='__main__')
