import torch

def generate_ode_data(N_batch, N_time, dt):
    # Sample random data to build different rhs f:
    input_data = torch.zeros((N_batch, N_time, 1))
    random_data = torch.zeros((N_batch, 3))
    random_data[:, :1] = torch.randint(0, 24, (N_batch, 1))
    random_data[:, 1:2] = torch.randint(0, 12, (N_batch, 1))
    random_data[:, 2:] = torch.randint(0, 6, (N_batch, 1))

    output_data = torch.zeros((N_batch, N_time, 1))

    t = 0.0
    input_data[:, 0, 0] = torch.sin(t * random_data[:, 0]) + 0.5 * torch.cos(t * random_data[:, 1]) \
                            + 2.0 * torch.sin(t * random_data[:, 2])
    # Do Euler scheme to compute reference solution
    for i in range(1, N_time):
        t += dt
        input_data[:, i, 0] = torch.sin(t * random_data[:, 0]) + 0.5 * torch.cos(t * random_data[:, 1]) \
                            + 2.0 * torch.sin(t * random_data[:, 2])
        output_data[:, i, 0] = output_data[:, i-1, 0] + dt * input_data[:, i, 0]

    return input_data, output_data