
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

# 病名映射
DISEASE_NAMES = {
    'AKIEC': 'Actinic Keratoses',
    'BCC': 'Basal Cell Carcinoma',
    'BKL': 'Benign Keratosis',
    'DF': 'Dermatofibroma',
    'MEL': 'Melanoma',
    'NV': 'Melanocytic Nevi',
    'VASC': 'Vascular Lesions'
}

def get_disease_name(filename):
    """从文件名中提取病名"""
    for key in DISEASE_NAMES.keys():
        if key in filename:
            return key, DISEASE_NAMES[key]
    return None, None

def draw_vertical_text_rotated(canvas, text, position, font, fill='black'):
    """绘制旋转90度的纵向文字"""
    x, y = position
    
    # 创建一个临时图片来绘制文字
    bbox = canvas.getdraw().textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    temp_img = Image.new('RGBA', (text_width + 20, text_height + 20), (255, 255, 255, 0))
    temp_draw = ImageDraw.Draw(temp_img)
    temp_draw.text((10, 10), text, fill=fill, font=font)
    
    # 旋转90度（逆时针）
    rotated_img = temp_img.rotate(90, expand=True)
    
    # 粘贴到主画布
    canvas.paste(rotated_img, (x, y), rotated_img)

def get_rotated_text_size(draw, text, font):
    """获取旋转后文字的尺寸"""
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    # 旋转90度后，宽高互换
    return (text_height, text_width)

def main():
    # 设置路径
    base_dir = Path(__file__).resolve().parents[1]
    image_dir = base_dir / 'visualizedata' / 'image'
    mpsa_dir = base_dir / 'visualizedata' / 'mpsa'
    ours_dir = base_dir / 'visualizedata' / 'ours'
    output_dir = base_dir  # 直接保存在项目根目录
    
    # 获取所有图片文件并排序
    image_files = sorted(list(image_dir.glob('*.jpg')))
    
    # 先读取一张热力图来获取尺寸
    sample_heatmap = Image.open(str(next(mpsa_dir.glob('*.jpg'))))
    heatmap_width, heatmap_height = sample_heatmap.size
    print(f"Heatmap size: {heatmap_width}x{heatmap_height}")
    
    # 设置画布参数
    label_width = 100  # 左侧标签宽度（更小，让文字离图片更近）
    bottom_label_height = 80  # 底部标签高度（需要容纳两行文字）
    gap = 0  # 纵向图片之间无间隙
    column_gap = 30  # 列之间的间距
    
    # 计算画布大小
    total_height = heatmap_height * len(image_files) + gap * (len(image_files) - 1) + bottom_label_height
    total_width = label_width + heatmap_width * 3 + column_gap * 2
    
    # 创建白色画布
    canvas = Image.new('RGB', (total_width, total_height), 'white')
    draw = ImageDraw.Draw(canvas)
    
    # 尝试加载字体
    try:
        font_large = ImageFont.truetype('arial.ttf', 38)
        font_medium = ImageFont.truetype('arial.ttf', 36)
        font_small = ImageFont.truetype('arial.ttf', 34)
    except:
        font_large = ImageFont.load_default()
        font_medium = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    # 处理每一张图片
    for idx, img_path in enumerate(image_files):
        # 获取病名
        disease_code, disease_name = get_disease_name(img_path.name)
        if not disease_code:
            continue
            
        # 计算当前行的y位置（无间隙）
        y_pos = idx * heatmap_height
        
        # 1. 处理原始图片
        orig_img = Image.open(str(img_path))
        # 调整原始图片大小
        orig_img_resized = orig_img.resize((heatmap_width, heatmap_height), Image.Resampling.LANCZOS)
        canvas.paste(orig_img_resized, (label_width, y_pos))
        
        # 2. 处理MPSA热力图
        mpsa_path = mpsa_dir / f"{img_path.stem}_tgcam.jpg"
        if mpsa_path.exists():
            mpsa_img = Image.open(str(mpsa_path))
            canvas.paste(mpsa_img, (label_width + heatmap_width + column_gap, y_pos))
        
        # 3. 处理Ours热力图
        ours_path = ours_dir / f"{img_path.stem}_tgcam.jpg"
        if ours_path.exists():
            ours_img = Image.open(str(ours_path))
            canvas.paste(ours_img, (label_width + 2 * (heatmap_width + column_gap), y_pos))
        
        # 4. 绘制左侧病名标签（旋转90度纵向排列，垂直居中）
        # 获取旋转后的文字尺寸
        bbox = draw.textbbox((0, 0), disease_name, font=font_medium)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        # 旋转后宽高互换
        rotated_width = text_height
        rotated_height = text_width
        
        # 计算位置：垂直和水平都居中
        text_x = (label_width - rotated_width) // 2
        text_y = y_pos + (heatmap_height - rotated_height) // 2
        
        # 绘制旋转文字
        # 创建临时图片绘制文字
        temp_img = Image.new('RGBA', (text_width + 20, text_height + 20), (255, 255, 255, 0))
        temp_draw = ImageDraw.Draw(temp_img)
        temp_draw.text((10, 10), disease_name, fill='black', font=font_medium)
        
        # 旋转90度（逆时针）
        rotated_img = temp_img.rotate(90, expand=True)
        
        # 粘贴到主画布
        canvas.paste(rotated_img, (text_x, text_y), rotated_img)
    
    # 绘制底部标签
    bottom_y = total_height - bottom_label_height + 10
    
    # Origin标签
    label_text = 'Origin'
    bbox = draw.textbbox((0, 0), label_text, font=font_large)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    text_x = label_width + (heatmap_width - text_width) // 2
    # Origin只需要一行
    draw.text((text_x, bottom_y + (bottom_label_height - text_height) // 2 - 5), label_text, fill='black', font=font_large)
    
    # Grad-CAM(MPSA)标签 - 分两行
    # 第一行：Grad-CAM
    label_top = 'Grad-CAM'
    bbox_top = draw.textbbox((0, 0), label_top, font=font_medium)
    text_width_top = bbox_top[2] - bbox_top[0]
    text_height_top = bbox_top[3] - bbox_top[1]
    
    # 第二行：(MPSA)
    label_bottom = '(MPSA)'
    bbox_bottom = draw.textbbox((0, 0), label_bottom, font=font_medium)
    text_width_bottom = bbox_bottom[2] - bbox_bottom[0]
    text_height_bottom = bbox_bottom[3] - bbox_bottom[1]
    
    # 居中位置
    center_x = label_width + heatmap_width + column_gap + heatmap_width // 2
    # 两行之间的间距
    line_gap = 5
    total_text_height = text_height_top + line_gap + text_height_bottom
    start_y = bottom_y + (bottom_label_height - total_text_height) // 2 - 5
    
    # 绘制第一行
    draw.text((center_x - text_width_top // 2, start_y), label_top, fill='black', font=font_medium)
    # 绘制第二行
    draw.text((center_x - text_width_bottom // 2, start_y + text_height_top + line_gap), label_bottom, fill='black', font=font_medium)
    
    # Grad-CAM(Ours)标签 - 分两行
    # 第二行：(Ours)
    label_bottom_ours = '(Ours)'
    bbox_bottom_ours = draw.textbbox((0, 0), label_bottom_ours, font=font_medium)
    text_width_bottom_ours = bbox_bottom_ours[2] - bbox_bottom_ours[0]
    
    # 居中位置
    center_x_ours = label_width + 2 * (heatmap_width + column_gap) + heatmap_width // 2
    
    # 绘制第一行
    draw.text((center_x_ours - text_width_top // 2, start_y), label_top, fill='black', font=font_medium)
    # 绘制第二行
    draw.text((center_x_ours - text_width_bottom_ours // 2, start_y + text_height_top + line_gap), label_bottom_ours, fill='black', font=font_medium)
    
    # 不绘制分隔线，保持简洁
    
    # 保存结果
    output_path = output_dir / 'comparison_figure.png'
    canvas.save(output_path, dpi=(300, 300))
    print(f"Comparison figure saved to: {output_path}")
    
    # 同时保存一份高质量PDF
    output_path_pdf = output_dir / 'comparison_figure.pdf'
    canvas.save(output_path_pdf, 'PDF', resolution=300)
    print(f"Comparison figure (PDF) saved to: {output_path_pdf}")
    
    # 显示结果
    plt.figure(figsize=(20, 16))
    plt.imshow(canvas)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()

