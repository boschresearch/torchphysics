import torch
import math

from .functionset import FunctionSet

from ...spaces import Points

class HarmonicFunctionSet1D(FunctionSet):

    def __init__(self, function_space, function_set_size, 
                 period, max_frequence, 
                 random_sample_fn = torch.randn):
        super().__init__(function_space, function_set_size)
        self.max_frequence = max_frequence
        self.period_len = period
        self.random_sample_fn = random_sample_fn

    def create_functions(self, device="cpu"):
        self.fourier_coefficients = self.random_sample_fn(
                (self.function_set_size, self.max_frequence+1, 2), device=device
            )

    def get_function(self, idx):
        if isinstance(idx, int): idx = [idx]
        self.current_idx = idx
        return self._eval_basis_at_locaction
    
    
    def _eval_basis_at_locaction(self, location : Points):
        if location.as_tensor.shape[0] == 1:
            location_copy = torch.repeat_interleave(
                    location[self.function_space.input_space].as_tensor, 
                    len(self.current_idx), dim=0
            )
        else:
            location_copy = location[self.function_space.input_space].as_tensor

        output = torch.zeros((len(self.current_idx), location_copy.shape[1], 1))

        pi_scale = 2 * math.pi / self.period_len
        for i in range(self.max_frequence+1):
            output[:, :, 0] += \
                self.fourier_coefficients[self.current_idx, i:i+1, 0] * torch.sin(pi_scale * i * location_copy[:, :, 0]) + \
                self.fourier_coefficients[self.current_idx, i:i+1, 1] * torch.cos(pi_scale * i * location_copy[:, :, 0])

        return Points(output, self.function_space.output_space)
    

class HarmonicFunctionSet2D(HarmonicFunctionSet1D):

    def __init__(self, function_space, function_set_size, 
                 period, max_frequence, 
                 random_sample_fn = torch.randn):
        assert isinstance(period, (list, tuple))
        assert isinstance(max_frequence, (list, tuple))
        super().__init__(function_space, function_set_size, period, max_frequence, random_sample_fn)


    def create_functions(self, device="cpu"):
        self.fourier_coefficients = self.random_sample_fn(
            (self.function_set_size, self.max_frequence[0]+1, self.max_frequence[1]+1, 4), device=device
            )
    
    
    def _eval_basis_at_locaction(self, location : Points):
        if location.as_tensor.shape[0] == 1:
            location_copy = torch.repeat_interleave(
                    location[self.function_space.input_space].as_tensor, 
                    len(self.current_idx), dim=0
            )
        else:
            location_copy = location[self.function_space.input_space].as_tensor

        shape = [len(self.current_idx)]
        shape.extend(location_copy.shape[1:-1])
        shape.append(1)
        output = torch.zeros(shape)

        pi_scale_x = 2 * math.pi / self.period_len[0]
        pi_scale_y = 2 * math.pi / self.period_len[1]

        for i in range(self.max_frequence[0]+1):
            sin_x = torch.sin(pi_scale_x * i * location_copy[..., 0])
            cos_x = torch.cos(pi_scale_x * i * location_copy[..., 0])

            for j in range(self.max_frequence[1]+1):
                sin_y = torch.sin(pi_scale_y * j * location_copy[..., 1])
                cos_y = torch.cos(pi_scale_y * j * location_copy[..., 1])

                output[..., 0] += \
                    self.fourier_coefficients[self.current_idx, i, j, 0:1] * sin_x * sin_y + \
                    self.fourier_coefficients[self.current_idx, i, j, 1:2] * cos_x * sin_y + \
                    self.fourier_coefficients[self.current_idx, i, j, 2:3] * sin_x * cos_y + \
                    self.fourier_coefficients[self.current_idx, i, j, 3:4] * cos_x * cos_y
                
        return Points(output, self.function_space.output_space)
    


class HarmonicFunctionSet3D(HarmonicFunctionSet2D):

    def create_functions(self, device="cpu"):
        self.fourier_coefficients = self.random_sample_fn(
            (self.function_set_size, 
             self.max_frequence[0]+1, 
             self.max_frequence[1]+1, 
             self.max_frequence[2]+1,
             8), device=device
            )
    
    def _eval_basis_at_locaction(self, location : Points):
        if location.as_tensor.shape[0] == 1:
            location_copy = torch.repeat_interleave(
                    location[self.function_space.input_space].as_tensor, 
                    len(self.current_idx), dim=0
            )
        else:
            location_copy = location[self.function_space.input_space].as_tensor

        shape = [len(self.current_idx)]
        shape.extend(location_copy.shape[1:-1])
        shape.append(1)
        output = torch.zeros(shape)

        pi_scale_x = 2 * math.pi / self.period_len[0]
        pi_scale_y = 2 * math.pi / self.period_len[1]
        pi_scale_z = 2 * math.pi / self.period_len[1]

        for i in range(self.max_frequence[0]+1):
            sin_x = torch.sin(pi_scale_x * i * location_copy[..., 0])
            cos_x = torch.cos(pi_scale_x * i * location_copy[..., 0])

            for j in range(self.max_frequence[1]+1):
                sin_y = torch.sin(pi_scale_y * j * location_copy[..., 1])
                cos_y = torch.cos(pi_scale_y * j * location_copy[..., 1])

                for k in range(self.max_frequence[2]+1):
                    sin_z = torch.sin(pi_scale_z * k * location_copy[..., 2])
                    cos_z = torch.cos(pi_scale_z * k * location_copy[..., 2])


                    output[..., 0] += \
                        self.fourier_coefficients[self.current_idx, i, j, 0:1] * sin_x * sin_y * sin_z + \
                        self.fourier_coefficients[self.current_idx, i, j, 1:2] * sin_x * sin_y * cos_z + \
                        self.fourier_coefficients[self.current_idx, i, j, 2:3] * sin_x * cos_y * cos_z + \
                        self.fourier_coefficients[self.current_idx, i, j, 3:4] * sin_x * cos_y * sin_z + \
                        self.fourier_coefficients[self.current_idx, i, j, 4:5] * cos_x * sin_y * sin_z + \
                        self.fourier_coefficients[self.current_idx, i, j, 5:6] * cos_x * sin_y * cos_z + \
                        self.fourier_coefficients[self.current_idx, i, j, 6:7] * cos_x * cos_y * cos_z + \
                        self.fourier_coefficients[self.current_idx, i, j, 7:8] * cos_x * cos_y * sin_z
                
        return Points(output, self.function_space.output_space)