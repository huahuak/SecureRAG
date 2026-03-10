from argparse import ArgumentParser, Namespace


class Config:

    def __init__(self):
        self.parser = ArgumentParser()
        self.__add_options()
        self.__add_generator_options()
        self.__add_seucure_options()
        self.__add_service_options()
        args, _ = self.parser.parse_known_args()
        self.args = args

    def __getattr__(self, name: str):
        return getattr(self.args, name)

    def __setattr__(self, name, value):
        self.__dict__[name] = value

    def __add_options(self):
        self.parser.add_argument("--load_size", type=int, default=1000)
        self.parser.add_argument("--eval_print_freq", type=int, default=1)
        self.parser.add_argument("--log_path", type=str, default="./log/stdout.log")
        self.parser.add_argument("--process_name", type=str, default="unkown_process")

    def __add_generator_options(self):
        self.parser.add_argument("--n_context", type=int, default=10)
        self.parser.add_argument("--batch_size", type=int, default=32)
        self.parser.add_argument("--text_maxlength", type=int, default=200)
        self.parser.add_argument("--answer_maxlength", type=int, default=50)
        self.parser.add_argument("--device", type=str, default="cuda")
        self.parser.add_argument(
            "--generator_model_path", type=str, default="models/nq_reader_base"
        )

    def __add_seucure_options(self):
        self.parser.add_argument("--private_passage_ratio", type=float, default=0.5)

    def __add_service_options(self):
        self.parser.add_argument("--gpu_service_port", type=str, default="8081")
        self.parser.add_argument("--tee_service_port", type=str, default="8082")
