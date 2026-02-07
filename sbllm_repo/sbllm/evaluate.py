from sbllm.core.arg_parser import cfg_parsing
from sbllm.execution import testing_and_reporting
from sbllm.core.logger import setup_app_logging


def main():
    cfg = cfg_parsing()
    setup_app_logging(cfg)
    testing_and_reporting(cfg)

if __name__ == "__main__":
    main()
    