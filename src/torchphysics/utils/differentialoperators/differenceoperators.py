"""File contains difference operators to approximate derivatives of
discrete functions. Intended for the derivatives of operator approaches like FNO.
"""
import torch


def discrete_grad_on_grid(model_out, grid_size):
      """ Approximates the gradient of a discrete function using finite differences.
    
      Parameters  
      ----------
      model_out : torch.Tensor
          The discrete function to approximate the gradient for.
      grid_size : float
          The step size used for the finite difference approximation and underlying grid.

      Notes
      -----
      This methode assumes that the input function which the gradient should be
      computed of is defined on a regular equidistant grid. 
      The shape of function is assumed to be of the form 
      (batch_size, N_1, N_2, ..., N_d, dim), 
      where dim is the output dimension of the functions and N_i is the
      resolution in the different space directions. 
      The gradient will computed in all d directions.
      A central difference scheme is used for the approximation and
      at the boundary a one-sided difference scheme is used (also of order 2).
      """
      number_of_dims = len(model_out.shape) - 2
      gradient = torch.zeros(*model_out.shape[:-1], number_of_dims * model_out.shape[-1], 
                             device=model_out.device)
      
      write_to_slice = [slice(None)] * len(model_out.shape)
      read_from_slice_left = [slice(None)] * len(model_out.shape)
      read_from_slice_right = [slice(None)] * len(model_out.shape)
      read_from_slice_one_sided = [slice(None)] * len(model_out.shape)
      
      for i in range(number_of_dims):
            # Update the last dimension to wrtie the correct gradient components
            write_to_slice[-1] = slice(i * model_out.shape[-1], (i + 1) * model_out.shape[-1])

            # pick correct dimenison for the current direction
            write_to_slice[i+1] = slice(1, -1)
            read_from_slice_left[i+1] = slice(2, None)
            read_from_slice_right[i+1] = slice(0, -2)

            # central gradient scheme
            gradient[write_to_slice] = \
                  (model_out[read_from_slice_left] - model_out[read_from_slice_right]) / (2 * grid_size)
            
            # At boundary one-sided difference scheme
            # "Left" boundary
            write_to_slice[i+1] = slice(0, 1)
            read_from_slice_left[i+1] = slice(0, 1)
            read_from_slice_right[i+1] = slice(1, 2)
            read_from_slice_one_sided[i+1] = slice(2, 3)
            gradient[write_to_slice] = \
                  (- 1.5 * model_out[read_from_slice_left] \
                   + 2 * model_out[read_from_slice_right] \
                   - 0.5 * model_out[read_from_slice_one_sided]) / grid_size
            # "Right" boundary
            write_to_slice[i+1] = slice(-1, None)
            read_from_slice_left[i+1] = slice(-1, None)
            read_from_slice_right[i+1] = slice(-2, -1)
            read_from_slice_one_sided[i+1] = slice(-3, -2)
            gradient[write_to_slice] = \
                  (1.5 * model_out[read_from_slice_left] \
                   - 2 * model_out[read_from_slice_right] \
                   + 0.5 * model_out[read_from_slice_one_sided]) / grid_size

            # reset
            write_to_slice[i+1] = slice(None)
            read_from_slice_left[i+1] = slice(None)
            read_from_slice_right[i+1] = slice(None)
            read_from_slice_one_sided[i+1] = slice(None)

      return gradient


def discrete_laplacian_on_grid(model_out, grid_size):
      """ Approximates the laplacian of a discrete function using finite differences.
    
      Parameters  
      ----------
      model_out : torch.Tensor
          The discrete function to approximate the laplacian for.
      grid_size : float
          The step size used for the finite difference approximation and underlying grid.

      Notes
      -----
      This methode assumes the same properties as `discrete_grad_on_grid`.
      """
      number_of_dims = len(model_out.shape) - 2
      laplace = torch.zeros(*model_out.shape, device=model_out.device)

      write_to_slice = [slice(None)] * len(model_out.shape)
      read_from_slice_left = [slice(None)] * len(model_out.shape)
      read_from_slice_right = [slice(None)] * len(model_out.shape)
      read_from_slice_center = [slice(None)] * len(model_out.shape)
      read_from_slice_one_sided = [slice(None)] * len(model_out.shape)

      for i in range(number_of_dims):
            # pick correct dimenison for the current direction
            write_to_slice[i+1] = slice(1, -1)
            read_from_slice_center[i+1] = slice(1, -1)
            read_from_slice_left[i+1] = slice(2, None)
            read_from_slice_right[i+1] = slice(0, -2)

            # central gradient scheme
            laplace[write_to_slice] += \
                  (model_out[read_from_slice_left] 
                   - 2 * model_out[read_from_slice_center]
                   + model_out[read_from_slice_right]) / (grid_size**2)

            # At boundary one-sided difference scheme
            # "Left" boundary
            write_to_slice[i+1] = slice(0, 1)
            read_from_slice_left[i+1] = slice(0, 1)
            read_from_slice_center[i+1] = slice(1, 2)
            read_from_slice_right[i+1] = slice(2, 3)
            read_from_slice_one_sided[i+1] = slice(3, 4)

            laplace[write_to_slice] += \
                  (2.0 * model_out[read_from_slice_left] \
                   - 5.0 * model_out[read_from_slice_center] \
                   + 4.0 * model_out[read_from_slice_right] \
                   - 1.0 * model_out[read_from_slice_one_sided]) / (grid_size**2)
            
            # "Right" boundary
            write_to_slice[i+1] = slice(-1, None)
            read_from_slice_left[i+1] = slice(-3, -2)
            read_from_slice_center[i+1] = slice(-2, -1)
            read_from_slice_right[i+1] = slice(-1, None)
            read_from_slice_one_sided[i+1] = slice(-4, -3)

            laplace[write_to_slice] += \
                  (2.0 * model_out[read_from_slice_right] \
                   - 5.0 * model_out[read_from_slice_center] \
                   + 4.0 * model_out[read_from_slice_left] \
                   - 1.0 * model_out[read_from_slice_one_sided]) / (grid_size**2)

            # reset
            write_to_slice[i+1] = slice(None)
            read_from_slice_center[i+1] = slice(None)
            read_from_slice_left[i+1] = slice(None)
            read_from_slice_right[i+1] = slice(None)
            read_from_slice_one_sided[i+1] = slice(None)
      
      return laplace