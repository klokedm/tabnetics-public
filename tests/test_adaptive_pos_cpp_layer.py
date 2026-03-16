import pytest


torch = pytest.importorskip("torch")
adaptive_pos_cpp = pytest.importorskip("adaptive_pos_cpp")

if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)


def test_adaptive_pos_cpp_layer_backward_produces_finite_grads():
    # Keep the tensor sizes tiny; this is a smoke check that the custom CUDA
    # extension runs forward/backward without producing NaNs.
    batch = 1
    size = 2
    num_functions = 1

    power_param_value = 2.0
    power_clamp_min = 0.1
    power_clamp_max = 5.0
    exp_param_value = 1.0
    exp_clamp_min = 0.0
    exp_clamp_max = 2.0
    epsilon = 1e-5
    l1_penalty = 0.05

    input_tensor = torch.randn(batch, size, num_functions, dtype=torch.double, requires_grad=True, device="cuda")
    weights_tensor = torch.randn(num_functions, dtype=torch.double, requires_grad=True, device="cuda")
    update_mask_tensor = torch.ones(num_functions, dtype=torch.int, device="cuda")
    power_param = torch.tensor([power_param_value], dtype=torch.double, requires_grad=True, device="cuda")
    exp_param = torch.tensor([exp_param_value], dtype=torch.double, requires_grad=True, device="cuda")

    output = adaptive_pos_cpp.adaptive22(
        input_tensor,
        weights_tensor,
        update_mask_tensor,
        power_param,
        exp_param,
        num_functions,
        power_clamp_min,
        power_clamp_max,
        exp_clamp_min,
        exp_clamp_max,
        epsilon,
        l1_penalty,
    )

    output.sum().backward()

    assert input_tensor.grad is not None
    assert weights_tensor.grad is not None
    assert power_param.grad is not None
    assert exp_param.grad is not None

    assert torch.isfinite(input_tensor.grad).all()
    assert torch.isfinite(weights_tensor.grad).all()
    assert torch.isfinite(power_param.grad).all()
    assert torch.isfinite(exp_param.grad).all()

