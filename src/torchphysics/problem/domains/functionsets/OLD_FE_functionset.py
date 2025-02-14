import torch

from ...spaces.points import Points
from .functionset import TestFunctionSet

class FEFunctionSet(TestFunctionSet):
    
    def __init__(self, function_space, order, mesh_vertices, mesh_triangles=None):
        super().__init__(function_space=function_space)
        if len(mesh_vertices.shape) == 1:
            mesh_vertices = mesh_vertices.unsqueeze(-1)
        dim = len(mesh_vertices[0])
        if dim == 1 and order == 1:
            self.finite_elements = LinearFE1D(mesh_vertices)
        if dim == 2 and order == 1:
            self.finite_elements = LinearFE2D(mesh_vertices, mesh_triangles)
        else:
            AssertionError(f"FE Space not implemented for dimension {dim} and order {order}.")


    def __call__(self, x=None):
        if self.finite_elements.quadrature_mode_on:
            helper_out = self.eval_fn_helper.apply(x[self.function_space.input_space.variables.pop()], 
                                                   self.finite_elements.basis_at_quadrature, 
                                                   self.finite_elements.grad_at_quadrature)
            
            return Points(helper_out, self.function_space.output_space)
        else:
            Points(self.finite_elements(x), self.function_space.output_space)


    def get_quad_weights(self, n):
        repeats = n // len(self.finite_elements.quadrature_weights_per_dof)
        return self.finite_elements.quadrature_weights_per_dof.repeat((repeats, 1, 1))


    def get_quadrature_points(self):
        return Points(self.finite_elements.quadrature_points_per_dof, self.function_space.input_space)
    

    def to(self, device):
        self.finite_elements.to(device)


    def switch_quadrature_mode_on(self, set_on : bool):
        self.finite_elements.switch_quadrature_mode_on(set_on)


class LinearFE2D():

    def __init__(self, mesh_vertices : torch.tensor, mesh_triangles : torch.tensor):
        self.mesh_vertices = mesh_vertices
        self.mesh_triangles = mesh_triangles
        self.quadrature_mode_on = True

        self.dim = 2
        self.basis_dim = len(self.mesh_vertices)

        center_points, triangle_volume = self.compute_center_and_volume()
        ## Next find which vertex belongs to which triangle, so we can find
        ## where each basis function is not zero and construct the corresponding 
        ## quadrature:
        vertex_to_triangle_map_h = [[] for _ in range(self.basis_dim)]
        for i in range(len(self.mesh_triangles)):
            for k in self.mesh_triangles[i]:
                vertex_to_triangle_map_h[k].append(i)

        # Pad the mapping with -1 to save as one big tensor:
        max_triangles_per_vertex = max(len(vertex_to_triangle_map_h[i]) for i in range(self.basis_dim))
        self.vertex_to_triangle_map = -1 * torch.ones((self.basis_dim, 
                                                       max_triangles_per_vertex), dtype=torch.long)
        for i in range(self.basis_dim):
            self.vertex_to_triangle_map[i, :len(vertex_to_triangle_map_h[i])] = \
                torch.tensor(vertex_to_triangle_map_h[i])


        # We need basis with compact support -> no functions at the boundary:
        self.find_boundary_dofs() 
        use_index = torch.ones(self.basis_dim, dtype=bool)
        use_index[self.boundary_dofs] = False
        self.quadrature_weights_per_dof = triangle_volume[self.vertex_to_triangle_map[use_index]] 
        self.quadrature_weights_per_dof *= (self.vertex_to_triangle_map[use_index] >= 0)
        self.quadrature_weights_per_dof = self.quadrature_weights_per_dof.unsqueeze(-1)
        self.quadrature_points_per_dof = center_points[self.vertex_to_triangle_map[use_index]]
        self.basis_at_quadrature = torch.tensor([1.0/3.0], requires_grad=True)

        self.compute_grad_per_dof(triangle_volume, use_index, max_triangles_per_vertex)

    def switch_quadrature_mode_on(self, set_on : bool):
        self.quadrature_mode_on = set_on
        if not set_on:
            AssertionError("Arbritrary evaluation not implemented!")

    def to(self, device):
        self.quadrature_points_per_dof = self.quadrature_points_per_dof.to(device)
        self.quadrature_weights_per_dof = self.quadrature_weights_per_dof.to(device)
        self.basis_at_quadrature = self.basis_at_quadrature.to(device)
        self.grad_at_quadrature = self.grad_at_quadrature.to(device)


    def compute_center_and_volume(self):
        ## Find center of each triangle
        center_points = torch.zeros((len(self.mesh_triangles), self.dim))
        for i in range(self.dim+1):
            center_points += self.mesh_vertices[self.mesh_triangles[:, i]]
        center_points /= (self.dim+1)

        ## Compute volume of each triangle
        vec_1 = self.mesh_vertices[self.mesh_triangles[:, 1]] \
                - self.mesh_vertices[self.mesh_triangles[:, 0]]
        vec_2 = self.mesh_vertices[self.mesh_triangles[:, 2]] \
                - self.mesh_vertices[self.mesh_triangles[:, 0]]
        triangle_volume = 0.5 *(vec_1[:, 0]*vec_2[:, 1] - vec_1[:, 1]*vec_2[:, 0])
        self.triangle_rot = torch.sign(triangle_volume)
        triangle_volume *= self.triangle_rot
        return center_points, triangle_volume


    def find_boundary_dofs(self):
        ## Not the most efficient way...
        boundary_edge_list = torch.zeros((self.basis_dim, self.basis_dim), dtype=torch.int8)
        for triangle in self.mesh_triangles:
            v = triangle.sort()[0]
            boundary_edge_list[v[0], v[1]] += 1
            boundary_edge_list[v[0], v[2]] += 1
            boundary_edge_list[v[1], v[2]] += 1
        self.boundary_dofs = torch.unique(torch.cat(torch.where(boundary_edge_list == 1)))


    def compute_grad_per_dof(self, triangle_volume, inner_dofs, max_triangles_per_vertex):
        inner_idx = torch.where(inner_dofs)[0]
        self.grad_at_quadrature = torch.zeros((len(inner_idx), max_triangles_per_vertex, 2))
        for c, i in enumerate(inner_idx):
            for j, k in enumerate(self.vertex_to_triangle_map[i]):
                if k == -1: # -1 mapping means not further triangles are neighbours
                    break 
                remove_i = (self.mesh_triangles[k] != i)
                neighbour_dofs = self.mesh_triangles[k][remove_i]
                edge_vec = self.mesh_vertices[neighbour_dofs[1]] - self.mesh_vertices[neighbour_dofs[0]]
                if not remove_i[1]: 
                    # if we removed the second vertex from the triangle, we need to multiple
                    # by -1 to fix the direction of the edge
                    edge_vec *= -1
                edge_vec = torch.tensor([-edge_vec[1], edge_vec[0]]) # rotate
                edge_vec /= (2*triangle_volume[k])
                self.grad_at_quadrature[c, j, :] = self.triangle_rot[k] * edge_vec

    def __call__(self, x):
        pass
    

class LinearFE1D():

    def __init__(self, mesh_vertices : torch.tensor):
        self.mesh_vertices, _ = torch.sort(mesh_vertices)
        self.quadrature_mode_on = True # if evaluation only happens at all quadrature points

        self.dim = 1
        self.basis_dim = len(self.mesh_vertices)

        center_points, interval_length = self.compute_center_and_volume()


        # We need basis with compact support -> no functions at the boundary:
        self.quadrature_weights_per_dof = torch.column_stack((interval_length[:-1], interval_length[1:]))
        self.quadrature_weights_per_dof = self.quadrature_weights_per_dof.unsqueeze(-1)

        self.quadrature_points_per_dof = torch.column_stack((center_points[:-1], center_points[1:]))
        self.quadrature_points_per_dof = self.quadrature_points_per_dof.unsqueeze(-1)
        self.basis_at_quadrature = torch.tensor([1.0/2.0])

        self.compute_grad_per_dof(interval_length)


    def switch_quadrature_mode_on(self, set_on : bool):
        self.quadrature_mode_on = set_on
        if not set_on:
            AssertionError("Arbritrary evaluation not implemented!")


    def to(self, device):
        self.quadrature_points_per_dof = self.quadrature_points_per_dof.to(device)
        self.quadrature_weights_per_dof = self.quadrature_weights_per_dof.to(device)
        self.basis_at_quadrature = self.basis_at_quadrature.to(device)
        self.grad_at_quadrature = self.grad_at_quadrature.to(device)


    def compute_center_and_volume(self):
        ## Find center of each triangle
        center_points = self.mesh_vertices[:-1] + self.mesh_vertices[1:]
        center_points /= 2.0

        interval_length = self.mesh_vertices[:-1] - self.mesh_vertices[1:]
        return center_points, interval_length
    

    def compute_grad_per_dof(self, interval_length):
        self.grad_at_quadrature = torch.zeros((self.basis_dim - 2, 2, 1))
        self.grad_at_quadrature[:, :1, 0] = 1.0/interval_length[:-1]
        self.grad_at_quadrature[:, 1:, 0] = -1.0/interval_length[:-1]


    def __call__(self, x):
        pass