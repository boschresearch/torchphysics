from torchphysics.models.model import Model
from torchphysics.problem.spaces import Points

from modulus.sym.hydra.utils import compose
from modulus.sym.hydra import instantiate_arch
from modulus.sym.models.arch import Arch
from modulus.sym.key import Key

from typing import Dict
from torch import Tensor
import torch

import logging
import os

class TPModelArch(Arch):
    """ 
    Wrapper class to convert instances of TorchPhysics model base class
    getting all necessary attributes of Modulus Arch models
    
    Parameters
    ----------
    model: TorchPhysics model object
        The model object to be converted into Modulus model object
    
    """

    def __init__( self,model) -> None:
        input_keys, output_keys = self.getVarKeys(model)
        super().__init__(
            input_keys=input_keys,
            output_keys=output_keys,
            detach_keys=[],
            periodicity=None,
        )
        
        self._impl = model     
    
    def forward(self, in_vars: Dict[str, Tensor]) -> Dict[str, Tensor]:        
        conv_in_vars_2 = self.convertSpatialVariables(in_vars)        
        y=self._tensor_forward(conv_in_vars_2)
               
        res= self.split_output(y, self.output_key_dict, dim=-1)
        return res
    
    def _tensor_forward(self, x: Tensor) -> Tensor:       
        # there were an error message coming out of an x-dictionary with tensor.float64 type entries
        # so is converted to normal float here         
        x_t = {k: v.float() for k, v in Points.from_coordinates(x).coordinates.items()}        
        points = Points.from_coordinates(x_t)          
        y = self._impl(points)                
        return y
    
    
    def convertSpatialVariables(self,vars):        
        conv_vars = {}
        for key in self._impl.input_space.keys():            
            if key =='x':                
                if self._impl.input_space['x'] > 1:
                    if self._impl.input_space['x'] ==2:
                        conv_vars['x']=torch.cat((vars['x'],vars['y']),dim=1)
                    else:
                        conv_vars['x']=torch.cat((vars['x'],vars['y'],vars['z']),dim=1)
                else:
                    conv_vars['x']=vars['x']                    
            else:
                if self._impl.input_space[key] >1:
                    cat_var = list(vars[key+str(l+1)] for l in range(self._impl.input_space[key]))
                    conv_vars[key] = torch.cat(cat_var,dim=1)
                else:
                    conv_vars[key]=vars[key]      
        return conv_vars
          
        
    def getVarKeys(self,model):        
        inputkeys = []
        outputkeys = []
        for key in model.input_space.keys():
            if key =='x':
                inputkeys.append(Key('x'))                  
                if model.input_space['x'] > 1:
                    inputkeys.append(Key('y'))                         
                    if model.input_space['x']== 3:            
                        inputkeys.append(Key('z'))                          
            else:
                if model.input_space[key] >1:
                    for l in range(model.input_space[key]):
                        inputkeys.append(Key(key+str(l+1)))         
                else:
                    inputkeys.append(Key(key))    
        for key in model.output_space.keys():
            if model.output_space[key] >1:
                    for l in range(model.output_space[key]):
                        outputkeys.append(Key(key+str(l+1)))         
            else:
                    outputkeys.append(Key(key))    
            
        return inputkeys, outputkeys
        
    def getInputSpace(self):
        return self._impl.input_space    
    

      


class ModulusArchitectureWrapper(Model):
    """
    Wrapper class to use all model architectures implemented in Modulus 
    library as TorchPhysics model. The chosen Modulus architecture is 
    defined by the arch_name parameter and optional architecture 
    specific parameters can be passed as keyword arguments. 
    The model then gets all necessary attributes of TorchPhysics model 
    base class. The input points are converted to the Modulus input 
    format and the Modulus output points back to the TorchPhysics 
    output format.    
    
    Parameters
    ----------        
    input_space : Space
        The space of the points the can be put into this model.
    output_space : Space
        The space of the points returned by this model.
    arch_name : {"afno","distributed_afno","deeponet","fno","fourier",
        "fully_connected","conv_fully_connected", 
        "fused_fully_connected","fused_fourier","fused_hash_encoding",
        "hash_encoding","highway_fourier","modified_fourier",
        "multiplicative_fourier","multiscale_fourier","pix2pix",
        "siren","super_res"}
        Name of the Modulus architecture.
    **kwargs : optional
        Additional keyword arguments, depending on the chosen Modulus 
        architecture - listed with default values:
            "afno": 
                img_shape: Tuple[int] = MISSING
                patch_size: int = 16
                embed_dim: int = 256
                depth: int = 4
                num_blocks: int = 8
            "distributed_afno":
                img_shape: Tuple[int] = MISSING
                patch_size: int = 16
                embed_dim: int = 256
                depth: int = 4
                num_blocks: int = 8
                channel_parallel_inputs: bool = False
                channel_parallel_outputs: bool = False
            "deeponet":
                trunk_dim: Any = None  # Union[None, int]
                branch_dim: Any = None  # Union[None, int]
            "fno":
                dimension: int = MISSING
                decoder_net: Arch
                nr_fno_layers: int = 4
                fno_modes: Any = 16 # Union[int, List[int]]
                padding: int = 8
                padding_type: str = "constant"
                activation_fn: str = "gelu"
                coord_features: bool = True
            "fourier":
                frequencies: Any = "('axis', [0, 1, 2, 3, 4, 5, 6, 7,
                     8, 9])"
                frequencies_params: Any = "('axis', [0, 1, 2, 3, 4, 5,
                    6, 7, 8, 9])"
                activation_fn: str = "silu"
                layer_size: int = 512
                nr_layers: int = 6
                skip_connections: bool = False
                weight_norm: bool = True
                adaptive_activations: bool = False
            "fully_connected":
                layer_size: int = 512
                nr_layers: int = 6
                skip_connections: bool = False
                activation_fn: str = "silu"
                adaptive_activations: bool = False
                weight_norm: bool = True
            "conv_fully_connected":
                layer_size: int = 512
                nr_layers: int = 6
                skip_connections: bool = False
                activation_fn: str = "silu"
                adaptive_activations: bool = False
                weight_norm: bool = True
            "fused_fully_connected":
                layer_size: int = 128
                nr_layers: int = 6
                activation_fn: str = "sigmoid"
            "fused_fourier":
                layer_size: int = 128
                nr_layers: int = 6
                activation_fn: str = "sigmoid"
                n_frequencies: int = 12               
            "fused_hash_encoding":             
                layer_size: int = 128
                nr_layers: int = 6
                activation_fn: str = "sigmoid"
                indexing: str = "Hash"
                n_levels: int = 16
                n_features_per_level: int = 2
                log2_hashmap_size: int = 19
                base_resolution: int = 16
                per_level_scale: float = 2.0
                interpolation: str = "Smoothstep"
            "hash_encoding":
                layer_size: int = 64
                nr_layers: int = 3
                skip_connections: bool = False
                weight_norm: bool = True
                adaptive_activations: bool = False
                bounds: Any = "[(1.0, 1.0), (1.0, 1.0)]"
                nr_levels: int = 16
                nr_features_per_level: int = 2
                log2_hashmap_size: int = 19
                base_resolution: int = 2
                finest_resolution: int = 32
            "highway_fourier":
                frequencies: Any = "('axis', [0, 1, 2, 3, 4, 5, 6, 7, 
                    8, 9])"
                frequencies_params: Any = "('axis', [0, 1, 2, 3, 4,
                    5, 6, 7, 8, 9])"
                activation_fn: str = "silu"
                layer_size: int = 512
                nr_layers: int = 6
                skip_connections: bool = False
                weight_norm: bool = True
                adaptive_activations: bool = False
                transform_fourier_features: bool = True
                project_fourier_features: bool = False
            "modified_fourier":
                frequencies: Any = "('axis', [0, 1, 2, 3, 4, 5, 6, 7, 
                    8, 9])"
                frequencies_params: Any = "('axis', [0, 1, 2, 3, 4, 
                    5, 6, 7, 8, 9])"
                activation_fn: str = "silu"
                layer_size: int = 512
                nr_layers: int = 6
                skip_connections: bool = False
                weight_norm: bool = True
                adaptive_activations: bool = False
            "multiplicative_fourier":
                layer_size: int = 512
                nr_layers: int = 6
                skip_connections: bool = False
                activation_fn: str = "identity"
                filter_type: str = "fourier"
                weight_norm: bool = True
                input_scale: float = 10.0
                gabor_alpha: float = 6.0
                gabor_beta: float = 1.0
                normalization: Any = (None # Change to Union[None, 
                    Dict[str, Tuple[float, float]]] when supported)
            "multiscale_fourier":
                frequencies: Any = field(default_factory=lambda: [32])
                frequencies_params: Any = None
                activation_fn: str = "silu"
                layer_size: int = 512
                nr_layers: int = 6
                skip_connections: bool = False
                weight_norm: bool = True
                adaptive_activations: bool = False
            "pix2pix":
                dimension: int = MISSING
                conv_layer_size: int = 64
                n_downsampling: int = 3
                n_blocks: int = 3
                scaling_factor: int = 1
                batch_norm: bool = True
                padding_type: str = "reflect"
                activation_fn: str = "relu"
            "siren":
                layer_size: int = 512
                nr_layers: int = 6
                first_omega: float = 30.0
                omega: float = 30.0
                normalization: Any = (None # Change to Union[None, 
                    Dict[str, Tuple[float, float]]] when supported)
            "super_res":
                large_kernel_size: int = 7
                small_kernel_size: int = 3
                conv_layer_size: int = 32
                n_resid_blocks: int = 8
                scaling_factor: int = 8
                activation_fn: str = "prelu"   
    
    
    Examples
    --------
        >>> testmodel=ModulusArchitectureWrapper(input_space=X*T, 
            output_space=U,arch_name='fully_connected',layer_size=30, nr_layers=3)
        >>> testmodel=ModulusArchitectureWrapper(input_space=X*T, 
            output_space=U,arch_name='fourier',frequencies = ['axis',[0,1,2]])
                
    """
    def __init__(self,input_space, output_space,arch_name,**kwargs):   
        # get the absolute path of the conf-directory
        caller_path=os.path.abspath(os.getcwd()+'/conf')
        os.makedirs(caller_path, exist_ok=True)  
        # Get the relative path of the current file to the conf-directory
        current_path = os.path.relpath(caller_path,os.path.dirname(os.path.abspath(__file__)))        
        with open(caller_path+'/config_model.yaml', 'w') as f:
            f.write('defaults :\n  - modulus_default\n  - arch:\n    - '+arch_name)    
        cfg3 = compose(config_path=current_path, config_name="config_model")       
        
        #from modulus.sym.models.arch import arch_name
        inputkeys = []
        input_keys = []
        for key in input_space.keys():
            if key =='x':
                inputkeys.append(Key('x'))
                input_keys.append('x')
            
                if input_space['x'] > 1:
                    inputkeys.append(Key('y'))
                    input_keys.append('y')
                if input_space['x']== 3:            
                    inputkeys.append(Key('z'))
                    input_keys.append('z')
            else:
                if input_space[key] >1:
                    for l in range(input_space[key]):
                        inputkeys.append(Key(key+str(l+1)))
                        input_keys.append(key+str(l+1))
                else:
                    inputkeys.append(Key(key))
                    input_keys.append(key)
        
        outputkeys = []
        for key in output_space.keys():
            if output_space[key] >1:
                    for l in range(output_space[key]):
                        outputkeys.append(Key(key+str(l+1)))         
            else:
                    outputkeys.append(Key(key))   
        
        super().__init__(input_space,output_space)
        
        self.output_space = output_space
        self.input_keys = input_keys
        hydra_conf = eval(f"cfg3.arch.{arch_name}")

        for key, value in kwargs.items():           
            hydra_conf[key] = value        
        
        self.modulus_net = instantiate_arch(input_keys=inputkeys,output_keys=outputkeys,cfg = hydra_conf)
        
    def forward(self, points):
        num_points = len(points)
        dim = len(self.input_keys) 
        # values of TP input points should get the same order as the keys of the input_space       
        points = self._fix_points_order(points)  
        # ordered values get the corresponding Modulus input keys (if some of TP input keys are of higher dimension than 1, they are split into several Modulus input keys)      
        points_modulus = dict(zip(self.input_keys,[points.as_tensor[:,index].reshape(num_points,1) for index in range(0,dim)]))           
        output_modulus = self.modulus_net(points_modulus)
        for key in self.output_space.keys():
            if self.output_space[key] >1:                    
                    cat_out = [output_modulus[key+str(l+1)] for l in range(self.output_space[key])]
                    output_modulus[key] = torch.cat(cat_out,dim=1) 
                    for l in range(self.output_space[key]):
                        output_modulus.pop(key+str(l+1))         
            
        return Points.from_coordinates(output_modulus)
        
        
class ModulusNetWrapper(Model):
    """
    Wrapper to convert objects of Modulus base class Arch into 
    TorchPhysics models
    
    Parameters
    ----------        
        arch: Modulus Arch object
            
    """       
    def __init__(self,arch):               
        input_space = torchphysics.problem.spaces.Space(arch.input_key_dict)
        output_space = torchphysics.problem.spaces.Space(arch.output_key_dict)
       
        super().__init__(input_space,output_space)        
        
        
        
        