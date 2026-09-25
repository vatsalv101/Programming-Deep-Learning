import torch
from torch import nn

import attention
import encoder


def _finite_gradients(module):
	gradients = [
		parameter.grad
		for parameter in module.parameters()
		if parameter.requires_grad
	]
	return (
		len(gradients) > 0
		and all(gradient is not None for gradient in gradients)
		and all(torch.isfinite(gradient).all() for gradient in gradients)
	)


def test_attention_forward_shape_and_weights():

	"""
	Checks output shape, attention weight shape, and probability normalization.
	"""

	torch.manual_seed(0)

	try:
		module = attention.MultiHeadSelfAttention(model_dim=8, num_heads=2)
	except Exception as error:
		print(f"FAIL: MultiHeadSelfAttention could not be initialized ({error}).")
		return False

	if not isinstance(module, nn.Module):
		print("FAIL: MultiHeadSelfAttention must inherit from nn.Module.")
		return False

	x = torch.randn(3, 5, 8)

	try:
		output, attn_weights = module(x)
	except Exception as error:
		print(f"FAIL: MultiHeadSelfAttention.forward raised an error ({error}).")
		return False

	if output.shape != (3, 5, 8):
		print(f"FAIL: Attention output has shape {output.shape}, expected (3, 5, 8).")
		return False

	if attn_weights.shape != (3, 2, 5, 5):
		print(
			"FAIL: Attention weights have the wrong shape: "
			f"got {attn_weights.shape}, expected (3, 2, 5, 5)."
		)
		return False

	if not torch.is_floating_point(output) or not torch.is_floating_point(attn_weights):
		print("FAIL: Attention should return floating point tensors.")
		return False

	row_sums = attn_weights.sum(dim=-1)
	if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4, rtol=1e-4):
		print("FAIL: Attention weights should be normalized with a softmax over the key dimension.")
		return False

	if not torch.isfinite(output).all() or not torch.isfinite(attn_weights).all():
		print("FAIL: Attention returned non-finite values.")
		return False

	module.zero_grad()
	output.mean().backward()

	if not _finite_gradients(module):
		print("FAIL: Attention parameters did not receive finite gradients.")
		return False

	print("PASS: MultiHeadSelfAttention forward pass is correct.")
	return True


def test_encoder_block_architecture():

	"""
	Checks that an EncoderBlock contains attention, an MLP, and normalization layers.
	"""

	torch.manual_seed(0)

	try:
		block = encoder.EncoderBlock(model_dim=8, num_heads=2, mlp_hidden_dim=16)
	except Exception as error:
		print(f"FAIL: EncoderBlock could not be initialized ({error}).")
		return False

	if not isinstance(block, nn.Module):
		print("FAIL: EncoderBlock must inherit from nn.Module.")
		return False

	modules = list(block.modules())
	has_attention = any(isinstance(module, attention.MultiHeadSelfAttention) for module in modules)
	has_layernorm = any(isinstance(module, nn.LayerNorm) for module in modules)
	has_linear = sum(isinstance(module, nn.Linear) for module in modules) >= 2
	has_gelu = any(isinstance(module, nn.GELU) for module in modules)

	if not has_attention:
		print("FAIL: EncoderBlock should contain a MultiHeadSelfAttention module.")
		return False

	if not has_layernorm:
		print("FAIL: EncoderBlock should contain LayerNorm modules.")
		return False

	if not has_linear:
		print("FAIL: EncoderBlock should contain at least two Linear layers for the MLP.")
		return False

	if not has_gelu:
		print("FAIL: EncoderBlock should use GELU as the activation function in the MLP.")
		return False

	x = torch.randn(4, 6, 8)

	try:
		y = block(x)
	except Exception as error:
		print(f"FAIL: EncoderBlock.forward raised an error ({error}).")
		return False

	if y.shape != (4, 6, 8):
		print(f"FAIL: EncoderBlock output has shape {y.shape}, expected (4, 6, 8).")
		return False

	if not torch.isfinite(y).all():
		print("FAIL: EncoderBlock output contains non-finite values.")
		return False

	block.zero_grad()
	y.mean().backward()

	if not _finite_gradients(block):
		print("FAIL: EncoderBlock parameters did not receive finite gradients.")
		return False

	print("PASS: EncoderBlock architecture and forward pass are correct.")
	return True


def test_create_transformer_encoder_structure():

	"""
	Checks that create_transformer_encoder returns a stack of EncoderBlocks.
	"""

	try:
		model = encoder.create_transformer_encoder(
			num_layers=3,
			model_dim=8,
			num_heads=2,
			mlp_hidden_dim=16,
		)
	except Exception as error:
		print(f"FAIL: create_transformer_encoder raised an error ({error}).")
		return False

	if not isinstance(model, nn.Sequential):
		print("FAIL: create_transformer_encoder should return nn.Sequential.")
		return False

	if len(model) != 3:
		print(f"FAIL: create_transformer_encoder returned {len(model)} layers, expected 3.")
		return False

	if not all(isinstance(block, encoder.EncoderBlock) for block in model):
		print("FAIL: create_transformer_encoder should contain only EncoderBlock modules.")
		return False

	x = torch.randn(2, 7, 8)

	try:
		y = model(x)
	except Exception as error:
		print(f"FAIL: transformer encoder stack raised an error ({error}).")
		return False

	if y.shape != (2, 7, 8):
		print(f"FAIL: transformer encoder output has shape {y.shape}, expected (2, 7, 8).")
		return False

	if not torch.isfinite(y).all():
		print("FAIL: transformer encoder output contains non-finite values.")
		return False

	model.zero_grad()
	y.mean().backward()

	if not _finite_gradients(model):
		print("FAIL: Transformer encoder parameters did not receive finite gradients.")
		return False

	print("PASS: Transformer encoder stack is correct.")
	return True


def test_attention():

	"""
	Runs the attention-related tests.
	"""

	ok = True
	ok = test_attention_forward_shape_and_weights() and ok
	return ok


def test_encoder():

	"""
	Runs the encoder block tests.
	"""

	ok = True
	ok = test_encoder_block_architecture() and ok
	return ok


def test_transformer_encoder():

	"""
	Runs the transformer encoder stack tests.
	"""

	ok = True
	ok = test_create_transformer_encoder_structure() and ok
	return ok
