class Data_process_pipe:
    def __init__(self, funcs_names):
        possible_funcs = list(filter(lambda x: callable(self.__getattribute__(x)), dir(self)))

        self.funcs = []
        funcs_names_final = []
        for func_name in funcs_names:
            if func_name in possible_funcs:
                func = self.__getattribute__(func_name)
                self.funcs.append(func)
                funcs_names_final.append(func_name)
        self.get_config = lambda: funcs_names_final

    def __call__(self, **kwargs):
        for func in self.funcs:
            kwargs = func(**kwargs)
        return kwargs


class Data_process_pipes:
    def __init__(self, pipes):
        self.funcs = []
        funcs_names = []
        for pipe in pipes:
            assert isinstance(pipe, Data_process_pipe)
            self.funcs.extend(pipe.funcs)
            funcs_names.extend(pipe.get_config())
        self.get_config = lambda: funcs_names

    def __call__(self, **kwargs):
        for func in self.funcs:
            x = func(**kwargs)
        return x
