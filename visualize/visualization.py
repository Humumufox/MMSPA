import os
import sys
import cv2
import numpy as np
import pandas as pd
import torch
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from sklearn.manifold import TSNE
from torchvision import transforms
from pathlib import Path

# Add project root to Python path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.build import build_models
from setup import config
from models.mps import format_reverse
from utils.data_loader import build_transforms
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, XGradCAM, AblationCAM, LayerCAM, EigenCAM, EigenGradCAM, GradCAMElementWise
import matplotlib.pyplot as plt

# Use existing checkpoint directory for visualization outputs
if hasattr(config.model, 'resume') and config.model.resume:
    checkpoint_path = Path(config.model.resume)
    if checkpoint_path.exists():
        RUN_ROOT = checkpoint_path.parent
    else:
        RUN_ROOT = Path(config.data.log_path)
else:
    RUN_ROOT = Path(config.data.log_path)

print('CWD =', Path.cwd())
print('ROOT =', ROOT)
print('RUN_ROOT =', RUN_ROOT)


def project_path(*parts):
	return str(ROOT.joinpath(*parts))


def run_path(*parts):
	if RUN_ROOT is None:
		return str(Path(config.data.log_path).joinpath(*parts))
	return str(Path(RUN_ROOT).joinpath(*parts))


def open_sample_folder(root):
	file_list = os.listdir(root)
	file_list.sort()
	full_path = []
	for file_name in file_list:
		file_path = os.path.join(root, file_name)
		full_path.append(file_path)
	return full_path


def img_transform(file_list):
	train_transforms, test_transforms = build_transforms(config)
	img_list = []
	for f in file_list:
		with open(f, 'r'):
			img = Image.open(f)
		img = test_transforms(img)
		img_list.append(img)
	img_list = torch.stack(img_list)
	return img_list


def img_visualize(file_list):
	# Deterministic visualization preprocessing: resize + center crop only.
	test_base = [
		transforms.Resize((config.data.img_size, config.data.img_size), InterpolationMode.BICUBIC),
		transforms.CenterCrop(config.data.img_size),
	]
	test_transforms = transforms.Compose([*test_base])
	img_list = []
	for f in file_list:
		with open(f, 'r'):
			img = Image.open(f)
		img = test_transforms(img)
		img = np.array(img)
		img_list.append(img)
		# plt.imshow(img)
		# plt.show()
	img_list = np.stack(img_list)
	return img_list


def build_eval_model(config, num_class=200):
	checkpoint = torch.load(config.model.resume, map_location='cpu')
	state_dicts = {k.replace('module.', ''): v for k, v in checkpoint['model'].items()}
	try:
		state_dicts = {k.replace('_orig_mod.', ''): v for k, v in state_dicts.items()}
	except:
		pass
	model = build_models(config, num_class, load_pre=False)
	model.load_state_dict(state_dicts, strict=False)
	for i in range(4):
		model.block.parts_generation_list[i].assess = True
	model.assess = True
	del checkpoint
	torch.cuda.empty_cache()
	model.eval()
	return model


def class_bar_figure(out, num_classes, softmax=False):
	x = torch.arange(1, num_classes + 1)
	class_name = pd.read_csv(project_path('data', 'ISIC_2018', 'classes.txt'), header=None, sep=' ')
	class_name = class_name.drop(columns=0).values.squeeze()
	if out.ndim == 1:
		out = out.unsqueeze(0)
	for img_idx, logits in enumerate(out):
		index = torch.argmax(logits, -1)
		pred_class = class_name[index]
		print(f'The Prediction Class for image {img_idx} is: {pred_class}')
		if softmax:
			logits = F.softmax(logits, -1)
			plt.ylim((0, 1))
		logits_np = logits.detach().cpu().numpy() if torch.is_tensor(logits) else logits
		plt.figure(figsize=(20, 5))
		plt.bar(x, logits_np, width=0.9)
		plt.grid(True, alpha=0.5)
		plt.xlim((0, num_classes))
		plt.title(f'{pred_class} (Image {img_idx})')
		plt.annotate(f'{index + 1}', xy=(index + 1, logits_np[index]))
		plt.tight_layout()
		out_path = run_path(f'prediction_result/img_{img_idx}_{pred_class}.pdf')
		print('Saving prediction figure to', out_path)
		os.makedirs(run_path('prediction_result'), exist_ok=True)
		plt.savefig(out_path)
		plt.show()


def part_generation_pos(model):
	for t in range(4):
		part_pos = model.block.parts_generation_list[t].part_pos
		part_pos = format_reverse(part_pos).squeeze()
		part_pos = part_pos.cpu().detach().numpy()
		stage_dir = run_path(f'part_generation_pos/stage_{t + 1}')
		print('Saving part generation maps to', stage_dir)
		os.makedirs(stage_dir, exist_ok=True)
		for i, map in enumerate(part_pos):
			map = cv2.resize(map, (384, 384), interpolation=cv2.INTER_NEAREST)
			plt.imsave(run_path(f'part_generation_pos/stage_{t + 1}/map_{i:03}.jpg'), map)


def part_attention_pos(model):
	for t in range(4):
		part_pos = model.block.mpsa_list[t].atten_pos
		head_mean = part_pos.mean(1)
		head_mean = format_reverse(head_mean).squeeze()
		head_mean = head_mean.cpu().detach().numpy()
		head_dir = run_path(f'part_attention_pos/headmean_stage_{t + 1}')
		print('Saving part attention headmean maps to', head_dir)
		os.makedirs(head_dir, exist_ok=True)
		for i, map in enumerate(head_mean):
			map = cv2.resize(map, (384, 384), interpolation=cv2.INTER_NEAREST)
			plt.imsave(run_path(f'part_attention_pos/headmean_stage_{t + 1}/map_{i:03}.jpg'), map)
		parts_mean = part_pos.mean(-1).transpose(-2, -1)
		parts_mean = format_reverse(parts_mean).squeeze()
		parts_mean = parts_mean.cpu().detach().numpy()
		part_dir = run_path(f'part_attention_pos/pratmean_stage_{t + 1}')
		print('Saving part attention partmean maps to', part_dir)
		os.makedirs(part_dir, exist_ok=True)
		for i, map in enumerate(parts_mean):
			map = cv2.resize(map, (384, 384), interpolation=cv2.INTER_NEAREST)
			plt.imsave(run_path(f'part_attention_pos/pratmean_stage_{t + 1}/map_{i:03}.jpg'), map)


def sample_map_save(file_list, saved_map):
	imgs = img_visualize(file_list)
	for b in range(imgs.shape[0]):
		for t, stage_map in enumerate(saved_map):
			sample_map = torch.load(stage_map).transpose(-2, -1)
			sample_map = format_reverse(sample_map).cpu().detach().numpy()
			maps = sample_map[b]
			stage_dir = run_path(f'sampling_map/img_{b}/stage_{t + 1}')
			print('Saving sampling maps to', stage_dir)
			os.makedirs(stage_dir, exist_ok=True)
			for i, map in enumerate(maps):
				map = cv2.resize(map, (384, 384), interpolation=cv2.INTER_NEAREST)
				plt.imshow(imgs[b])
				plt.imshow(map, alpha=0.5, cmap='jet')
				plt.axis('off')
				plt.savefig(run_path(f'sampling_map/img_{b}/stage_{t + 1}/map_{i:03}.jpg'), bbox_inches='tight', pad_inches=0)
				plt.cla()


def grad_cam_save(file_list, model):
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = model.eval().to(device)
	target_layers = [model.block.activation]
	imgs = img_visualize(file_list)
	tensors = img_transform(file_list).to(device)
	for b in range(imgs.shape[0]):
		tensor = tensors[b].unsqueeze(0)
		with GradCAM(model=model, target_layers=target_layers, reshape_transform=format_reverse) as cam:
			grayscale_cams = cam(input_tensor=tensor, targets=None)
		plt.imshow(imgs[b])
		plt.imshow(grayscale_cams[0], alpha=0.5, cmap='jet')
		plt.axis('off')
		os.makedirs(run_path(f'feature_weights/img_{b}'), exist_ok=True)
		plt.savefig(run_path(f'feature_weights/img_{b}/tgcam.jpg'), bbox_inches='tight', pad_inches=0)
		plt.cla()


def feature_weights_save(file_list, save_weights, model, gap=True):
	imgs = img_visualize(file_list)
	print(f'Feature Weights coefficient of four layer{model.pooling_weights.squeeze()}')
	weights_path = os.path.join(save_weights, 'weights.pt')
	x_path = os.path.join(save_weights, 'x.pt')
	gcam_path = os.path.join(save_weights, 'gradcam.pt')
	if not os.path.exists(weights_path):
		print(f'weights.pt not found at {weights_path}, skipping feature weights visualization')
		return
	if not os.path.exists(x_path):
		print(f'x.pt not found at {x_path}, skipping feature weights visualization')
		return
	feature_weights = torch.load(weights_path)
	x = torch.load(x_path)
	gcam = torch.load(gcam_path) if os.path.exists(gcam_path) else None
	if gcam is None:
		print(f'gradcam.pt not found at {gcam_path}, skipping gradcam overlay')
	for b in range(imgs.shape[0]):
		feature_weights_b = feature_weights.squeeze(-1).permute(1, 0, 2)
		if gap:
			xm = x.mean(1).squeeze(1)
		else:
			xm, _ = x.max(1)
		fw = feature_weights_b[b].reshape(4, 12, 12).cpu().detach().numpy()
		fm = xm[b].reshape(12, 12).cpu().detach().numpy()
		gc = gcam[b].reshape(12, 12).cpu().detach().numpy() if gcam is not None else None

		img_dir = run_path(f'feature_weights/img_{b}')
		print('Saving feature weights to', img_dir)
		os.makedirs(img_dir, exist_ok=True)
		feature_map = cv2.resize(fm, (384, 384), interpolation=cv2.INTER_NEAREST)
		plt.imshow(imgs[b])
		plt.imshow(feature_map, alpha=0.5)
		plt.axis('off')
		plt.savefig(run_path(f'feature_weights/img_{b}/feature_map.jpg'), bbox_inches='tight', pad_inches=0)
		plt.cla()

		if gcam is not None and gc is not None:
			gradcam = cv2.resize(gc, (384, 384))
			plt.imshow(imgs[b])
			plt.imshow(gradcam, alpha=0.5, cmap='jet')
			plt.axis('off')
			plt.savefig(run_path(f'feature_weights/img_{b}/gcam.jpg'), bbox_inches='tight', pad_inches=0)
			plt.cla()

		for i in range(4):
			plt.imshow(imgs[b])
			nmap = cv2.resize(fw[i], (384, 384), interpolation=cv2.INTER_NEAREST)
			plt.imshow(nmap, alpha=0.5)
			plt.axis('off')
			plt.savefig(run_path(f'feature_weights/img_{b}/map_{i + 1}.jpg'), bbox_inches='tight', pad_inches=0)
			plt.cla()

		plt.imshow(imgs[b])
		sum_fw = (model.pooling_weights.reshape(4, 1, 1).cpu().detach().numpy() * fw).sum(0)
		tmap = cv2.resize(sum_fw, (384, 384), interpolation=cv2.INTER_NEAREST)
		plt.imshow(tmap, alpha=0.5)
		plt.axis('off')
		plt.savefig(run_path(f'feature_weights/img_{b}/sum map.jpg'), bbox_inches='tight', pad_inches=0)
		plt.cla()


def gauss(data):
	data = data.float()
	return torch.exp(-1 * data**2)


def center_norm(datas, center=True):
	if center:
		datas = datas - datas.mean(-2, keepdim=True)
	datas = datas / torch.norm(datas, dim=-1, keepdim=True)
	return datas


def tSNE(dataset, num_class):
	features = torch.load(project_path('saved_features', f'{dataset}_f.pth')).cpu()
	labels = torch.load(project_path('saved_features', f'{dataset}_l.pth')).cpu()

	onehot = torch.zeros((features.shape[0], num_class))
	onehot.scatter_(1, labels.unsqueeze(1), 1)
	print(onehot.shape)
	print(features.shape)
	max_entropy = onehot - F.softmax(features.float(), -1)
	res = F.relu(max_entropy).sum(-1)

	class_difficult = torch.zeros(num_class)
	for i in range(num_class):
		class_difficult[i] = res[labels == i].mean()
	print(torch.topk(class_difficult, 50, -1))

	print(features.shape)
	difficult_class = torch.arange(num_class)
	tsne = TSNE(n_components=2, perplexity=10)
	embedded_features = tsne.fit_transform(features)
	colors = plt.cm.rainbow(np.linspace(0, 1, num_class))
	b = torch.randperm(num_class)
	colors = colors[b]
	marker_size = 5
	plt.figure(figsize=(10, 8))
	for i, i_c in enumerate(difficult_class):
		plt.scatter(embedded_features[labels == i_c, 0],
		            embedded_features[labels == i_c, 1],
		            color=colors[i],
		            label=str(i_c), s=marker_size)

	plt.legend().set_visible(False)
	plt.axis('off')
	plt.savefig(project_path(f'tsne_{dataset}_fff.pdf'), bbox_inches='tight', pad_inches=0)
	plt.show()


def model_prediction(file_list, model, label=None):
	model.eval()
	rgb_img = img_visualize(file_list)
	img = img_transform(file_list)
	out = model(img)
	out = out.squeeze().cpu().detach()
	return out


if __name__ == '__main__':
	dataset = 'isic_2018'
	num_class = {'isic_2018': 7}
	file_list = open_sample_folder(project_path('sample_img', dataset))
	model = build_eval_model(config, num_class[dataset])
	label = None
	model.eval()
	out = model_prediction(file_list, model, label)
	print('outlogits done')

	# 类别预测Logits/概率
	class_bar_figure(out, num_class[dataset], softmax=False)
	print('logits figure done')

	# 部件采样注意力的位置编码
	part_attention_pos(model)
	print('part attention pos done')

	# 部件生成的位置编码
	part_generation_pos(model)
	print('part generation pos done')

	# 根据输入生成的采样图
	sampling_map_dir = run_path('sampling_map', 'map_file')
	if os.path.isdir(sampling_map_dir):
		sample_map_save(file_list, open_sample_folder(sampling_map_dir))
		print('sampling map done')
	else:
		print(f'sampling map skipped: {sampling_map_dir} not found')

	# 特征权重与特征图，也会有类激活图
	feature_weights_dir = run_path('feature_weights')
	if os.path.isdir(feature_weights_dir):
		feature_weights_save(file_list, feature_weights_dir, model, True)
		print('feature weights done')
	else:
		print(f'feature weights skipped: {feature_weights_dir} not found')

	# 单独梯度类激活图
	grad_cam_save(file_list, model)
	print('grad cam done')
