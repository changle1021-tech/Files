import argparse
import runpy
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--diagnostic-tp', type=int, required=True)
    args, rest = parser.parse_known_args()

    import ray
    ray.init(
        num_cpus=4,
        num_gpus=args.diagnostic_tp,
        object_store_memory=512 * 1024 * 1024,
        include_dashboard=False,
        _temp_dir=f'/tmp/vidur_http_native_validation_entry_tp{args.diagnostic_tp}',
    )
    sys.argv = ['vllm.entrypoints.openai.api_server'] + rest
    runpy.run_module('vllm.entrypoints.openai.api_server', run_name='__main__')


if __name__ == '__main__':
    main()
