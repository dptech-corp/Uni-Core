import torch
import torch.nn.functional as F
from unicore.modules import softmax_dropout


def gen_attn_mask(mask, neg_inf):
    assert neg_inf < -1e4
    attn_mask = torch.zeros_like(mask)
    attn_mask[mask == 0] = neg_inf
    return attn_mask


def normal_softmax(a, mask, bias):
    return F.softmax(a + mask + bias, dim=-1)


def fused_softmax(a, mask, bias):
    return softmax_dropout(a, 0, True, mask=mask, bias=bias)


def wrap_forward_backward(func, a1, mask, bias1):
    a = a1.clone()
    bias = bias1.clone()
    a.requires_grad = True
    bias.requires_grad = True
    output = func(a, mask, bias)
    o = output.float().sum()
    o.backward()
    return output, a.grad, bias.grad


def check_diff(a, b, name, eps=1e-3):
    assert (a - b).abs().max() < eps, "name {}, diff {}".format(
        name, (a - b).abs().max()
    )


def test_softmax():
    n_batch = 4
    n_heads = 8
    n_query = 128
    test_dims = [64, 128, 256, 512, 1024, 1536, 2048]
    test_dtype = [torch.float32, torch.float16, torch.bfloat16]
    test_device = torch.device("cuda")
    for last_dim in test_dims:
        for dtype in test_dtype:
            x = torch.rand(
                n_batch,
                n_heads,
                n_query,
                last_dim,
                dtype=dtype,
                device=test_device,
            )
            mask = gen_attn_mask(
                (
                    torch.rand(
                        n_batch,
                        1,
                        1,
                        last_dim,
                        dtype=dtype,
                        device=test_device,
                    )
                    > 0.2
                ).type(x.dtype),
                -3e4,
            )
            bias = torch.rand(
                n_batch, n_heads, n_query, last_dim, dtype=dtype, device=test_device
            )
            out_a1, out_b1, out_c1 = wrap_forward_backward(
                normal_softmax, x, mask, bias
            )
            out_a2, out_b2, out_c2 = wrap_forward_backward(fused_softmax, x, mask, bias)
            check_diff(out_a1, out_a2, "output")
            check_diff(out_b1, out_b2, "grad_input")
            check_diff(out_c1, out_c2, "grad_bias")


def test_tri_softmax1():
    n_batch = 2
    n_groups = 32
    n_heads = 8
    n_query = 128
    test_dims = [64, 128, 256, 512, 1024, 1536, 2048]
    test_dtype = [torch.float32, torch.float16, torch.bfloat16]
    test_device = torch.device("cuda")
    for last_dim in test_dims:
        for dtype in test_dtype:
            x = torch.rand(
                n_batch,
                n_groups,
                n_heads,
                n_query,
                last_dim,
                dtype=dtype,
                device=test_device,
            )
            mask = gen_attn_mask(
                (
                    torch.rand(
                        n_batch,
                        n_groups,
                        1,
                        1,
                        last_dim,
                        dtype=dtype,
                        device=test_device,
                    )
                    > 0.2
                ).type(x.dtype),
                -3e4,
            )
            bias = torch.rand(
                1, 1, n_heads, n_query, last_dim, dtype=dtype, device=test_device
            )
            out_a1, out_b1, out_c1 = wrap_forward_backward(
                normal_softmax, x, mask, bias
            )
            out_a2, out_b2, out_c2 = wrap_forward_backward(fused_softmax, x, mask, bias)
            check_diff(out_a1, out_a2, "output")
            check_diff(out_b1, out_b2, "grad_input")
            check_diff(out_c1, out_c2, "grad_bias")


def test_tri_softmax2():
    n_batch = 2
    n_groups = 32
    n_heads = 8
    n_query = 128
    test_dims = [64, 128, 256, 512, 1024, 1536, 2048]
    test_dtype = [torch.float32, torch.float16, torch.bfloat16]
    test_device = torch.device("cuda")
    for last_dim in test_dims:
        for dtype in test_dtype:
            x = torch.rand(
                n_batch,
                n_groups,
                n_heads,
                n_query,
                last_dim,
                dtype=dtype,
                device=test_device,
            )
            mask = gen_attn_mask(
                (
                    torch.rand(
                        n_batch,
                        n_groups,
                        n_heads,
                        1,
                        last_dim,
                        dtype=dtype,
                        device=test_device,
                    )
                    > 0.2
                ).type(x.dtype),
                -3e4,
            )
            bias = torch.rand(
                1, n_groups, n_heads, n_query, last_dim, dtype=dtype, device=test_device
            )
            out_a1, out_b1, out_c1 = wrap_forward_backward(
                normal_softmax, x, mask, bias
            )
            out_a2, out_b2, out_c2 = wrap_forward_backward(fused_softmax, x, mask, bias)
            check_diff(out_a1, out_a2, "output")
            check_diff(out_b1, out_b2, "grad_input")
            check_diff(out_c1, out_c2, "grad_bias")
