import math

import torch
from torch import nn

import encoder
import vit


def _find_modules(model, predicate):
	return [module for module in model.modules() if predicate(module)]


def _replace_named_module(model, module_name, new_module):
	parent_name, _, child_name = module_name.rpartition(".")
	parent = model.get_submodule(parent_name) if parent_name else model
	setattr(parent, child_name, new_module)


def _replace_named_parameter(model, parameter_name, new_parameter):
	parent_name, _, child_name = parameter_name.rpartition(".")
	parent = model.get_submodule(parent_name) if parent_name else model
	setattr(parent, child_name, nn.Parameter(new_parameter))


def _first_module_name(model, predicate):
	for name, module in model.named_modules():
		if predicate(module):
			return name, module
	return None, None


def _single_module_or_fail(model, predicate, message):
	modules = _find_modules(model, predicate)
	if len(modules) != 1:
		print(message)
		return None
	return modules[0]


def test_vit_init_architecture():
	
	"""
	Checks that ViT.init contains the required architectural building blocks.
	"""

	torch.manual_seed(0)

	try:
		model = vit.ViT(
			num_layers=2,
			model_dim=8,
			num_heads=2,
			patch_size=4,
			num_classes=10,
		)
	except Exception as error:
		print(f"FAIL: ViT could not be initialized ({error}).")
		return False

	if not isinstance(model, nn.Module):
		print("FAIL: ViT must inherit from nn.Module.")
		return False

	patch_proj = _single_module_or_fail(
		model,
		lambda module: isinstance(module, nn.Linear)
		and module.in_features == 3 * 4 * 4
		and module.out_features == 8,
		"FAIL: ViT should contain a patch projection Linear layer from flattened patches to model_dim.",
	)
	if patch_proj is None:
		return False

	encoder_stacks = [
		module
		for module in model.modules()
		if isinstance(module, nn.Sequential)
		and len(module) == 2
		and all(isinstance(child, encoder.EncoderBlock) for child in module)
	]
	if len(encoder_stacks) != 1:
		print("FAIL: ViT should contain exactly one Sequential stack of EncoderBlocks.")
		return False

	head_candidates = []
	for module in model.modules():
		if not isinstance(module, nn.Sequential):
			continue
		children = list(module.children())
		linear_layers = [child for child in children if isinstance(child, nn.Linear)]
		if len(linear_layers) < 2:
			continue
		if linear_layers[0].in_features == 8 and linear_layers[-1].out_features == 10:
			head_candidates.append(module)

	if len(head_candidates) != 1:
		print("FAIL: ViT should contain an MLP head that maps model_dim to the class logits.")
		return False

	patch_size = int(math.isqrt(patch_proj.in_features // 3))
	if patch_size * patch_size * 3 != patch_proj.in_features:
		print("FAIL: ViT patch projection should receive flattened RGB patches.")
		return False

	num_patches = (32 // patch_size) ** 2

	cls_parameters = [
		parameter
		for parameter in model.parameters()
		if parameter.shape == (1, 1, 8) and parameter.requires_grad
	]
	if len(cls_parameters) != 1:
		print("FAIL: ViT should contain exactly one learnable CLS token parameter.")
		return False

	pos_parameters = [
		parameter
		for parameter in model.parameters()
		if parameter.shape == (1, num_patches + 1, 8) and parameter.requires_grad
	]
	if len(pos_parameters) != 1:
		print("FAIL: ViT should contain exactly one learnable position embedding parameter.")
		return False

	print("PASS: ViT initialization is correct.")
	return True


def test_vit_forward_computation():

	"""
	Checks that the forward pass uses patch extraction, CLS token prepending, positional embeddings,
	and the final classifier head in the correct order.
	"""

	torch.manual_seed(0)

	model = vit.ViT(
		num_layers=1,
		model_dim=4,
		num_heads=2,
		patch_size=4,
		num_classes=2,
	)

	patch_proj_name, _ = _first_module_name(
		model,
		lambda module: isinstance(module, nn.Linear)
		and module.in_features == 3 * 4 * 4
		and module.out_features == 4,
	)
	if patch_proj_name is None:
		print("FAIL: Could not find the patch projection layer in ViT.")
		return False

	encoder_name, _ = _first_module_name(
		model,
		lambda module: isinstance(module, nn.Sequential)
		and len(module) == 1
		and all(isinstance(child, encoder.EncoderBlock) for child in module),
	)
	if encoder_name is None:
		print("FAIL: Could not find the encoder stack in ViT.")
		return False

	head_name, _ = _first_module_name(
		model,
		lambda module: isinstance(module, nn.Sequential)
		and len(list(module.children())) >= 2
		and any(isinstance(child, nn.Linear) and child.out_features == 2 for child in module.children()),
	)
	if head_name is None:
		print("FAIL: Could not find the classifier head in ViT.")
		return False

	patch_proj = nn.Linear(3 * 4 * 4, 4, bias=False)
	with torch.no_grad():
		patch_proj.weight.zero_()
		patch_proj.weight[:4, :4] = torch.eye(4)
	_replace_named_module(model, patch_proj_name, patch_proj)

	captured = {}

	class CaptureEncoder(nn.Module):
		def forward(self, x):
			captured["encoder_input"] = x.detach().clone()
			return x + 1.0

	_replace_named_module(model, encoder_name, CaptureEncoder())

	head = nn.Linear(4, 2, bias=False)
	with torch.no_grad():
		head.weight.zero_()
		head.weight[0, 0] = 1.0
		head.weight[1, 1] = 1.0
	_replace_named_module(model, head_name, head)

	_replace_named_parameter(model, "cls_token", torch.zeros(1, 1, 4))
	pos_embed = torch.zeros(1, 65, 4)
	pos_embed[0, 0] = torch.tensor([5.0, 6.0, 7.0, 8.0])
	_replace_named_parameter(model, "pos_embed", pos_embed)

	calls = {"count": 0}

	def fake_create_patch_sequence(x, patch_size):
		calls["count"] += 1
		if patch_size != 4:
			raise AssertionError(f"Expected patch_size=4, got {patch_size}.")

		batch_size = x.shape[0]
		patches = torch.zeros(batch_size, 64, 48, device=x.device, dtype=x.dtype)
		patches[:, 0, :4] = torch.tensor([10.0, 20.0, 30.0, 40.0], device=x.device, dtype=x.dtype)
		patches[:, 1, :4] = torch.tensor([1.0, 2.0, 3.0, 4.0], device=x.device, dtype=x.dtype)
		return patches

	original_create_patch_sequence = vit.create_patch_sequence
	vit.create_patch_sequence = fake_create_patch_sequence

	x = torch.randn(2, 3, 32, 32)

	try:
		logits = model(x)
	except Exception as error:
		print(f"FAIL: ViT.forward raised an error ({error}).")
		return False
	finally:
		vit.create_patch_sequence = original_create_patch_sequence

	if calls["count"] != 1:
		print("FAIL: ViT.forward should call create_patch_sequence exactly once.")
		return False

	if logits.shape != (2, 2):
		print(f"FAIL: ViT.forward returned shape {logits.shape}, expected (2, 2).")
		return False

	if "encoder_input" not in captured:
		print("FAIL: The encoder stack was not called during the forward pass.")
		return False

	encoder_input = captured["encoder_input"]
	if encoder_input.shape != (2, 65, 4):
		print(f"FAIL: Encoder input has shape {encoder_input.shape}, expected (2, 65, 4).")
		return False

	expected_first_token = torch.tensor([5.0, 6.0, 7.0, 8.0], device=encoder_input.device)
	expected_first_patch = torch.tensor([10.0, 20.0, 30.0, 40.0], device=encoder_input.device)

	if not torch.allclose(encoder_input[:, 0, :], expected_first_token.expand_as(encoder_input[:, 0, :])):
		print("FAIL: ViT.forward does not prepend the CLS token or add positional embeddings correctly.")
		return False

	if not torch.allclose(encoder_input[:, 1, :], expected_first_patch.expand_as(encoder_input[:, 1, :])):
		print("FAIL: ViT.forward does not project patches or place them after the CLS token correctly.")
		return False

	expected_logits = torch.tensor([[6.0, 7.0], [6.0, 7.0]], device=logits.device, dtype=logits.dtype)
	if not torch.allclose(logits, expected_logits):
		print(
			"FAIL: ViT.forward did not compute the expected logits from the CLS token path."
		)
		return False

	print("PASS: ViT forward pass is correct.")
	return True


def test_vit_init():

	"""
	Runs the ViT initialization tests.
	"""
	
	return test_vit_init_architecture()


def test_vit_forward():

	"""
	Runs the ViT forward-pass tests.
	"""

	return test_vit_forward_computation()
