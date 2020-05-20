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
            kwargs = func(**kwargs)
        return kwargs


class Tape:
    def __init__(self):
        self.central_pad = CentralPad(224)

    def __call__(self, **kwargs):
        img = kwargs['x']
        self.h, self.w, self.c = img.shape
        self.central_pad(**kwargs)


class TapeSharingParams:
    pass


class Configurable:
    def __init__(self, **kwargs):
        self.__dict__ = kwargs

    def get_config(self):
        return self.__dict__

import numpy as np
class CentralPad(Configurable, TapeSharingParams):
    def __init__(self, output_size):
        super(CentralPad, self).__init__(output_size=output_size)

    def __call__(self, **kwargs):
        img = kwargs['x']
        y, x = self.h, self.w
        assert y <= self.output_size and x <= self.output_size

        pad_y = self.output_size-y
        residue_y = pad_y % 2
        pad_y //= 2

        pad_x = self.output_size - x
        residue_x = pad_x % 2
        pad_x //= 2

        img = np.pad(img, [
            [pad_y, pad_y + residue_y],
            [pad_x, pad_x + residue_x],
            [0, 0]
        ], 'constant', constant_values=[0., 0.])
        kwargs['x'] = img
        return kwargs


t = Tape()
t(**{'x': np.zeros((10,20,3))})