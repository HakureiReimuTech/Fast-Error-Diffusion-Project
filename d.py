import cv2
import numpy as np
import argparse

def halftone_dithering(input_path, output_path):
    # 读取图像并转换为灰度图
    gray = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError("无法读取图像，请检查文件路径和格式")
    
    # 转换为浮点型以便进行误差计算
    img = gray.astype(np.float32)
    height, width = img.shape
    
    # 遍历每个像素
    for y in range(height):
        for x in range(width):
            old_pixel = img[y, x]
            new_pixel = 255.0 if old_pixel >= 128.0 else 0.0
            img[y, x] = new_pixel
            quant_error = old_pixel - new_pixel
            
            # Floyd-Steinberg误差扩散
            if x + 1 < width:
                img[y, x+1] += quant_error * 7/16
            if y + 1 < height:
                if x - 1 >= 0:
                    img[y+1, x-1] += quant_error * 3/16
                img[y+1, x] += quant_error * 5/16
                if x + 1 < width:
                    img[y+1, x+1] += quant_error * 1/16
    
    # 转换回uint8类型并保存
    binary = img.astype(np.uint8)
    if not cv2.imwrite(output_path, binary):
        raise ValueError("无法保存图像，请检查输出路径和权限")

if __name__ == "__main__":
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='灰度图像半色调转换')
    parser.add_argument('input', help='输入图像路径（支持JPG/PNG等格式）')
    parser.add_argument('output', help='输出图像路径（支持PNG/JPG等格式）')
    args = parser.parse_args()

    try:
        halftone_dithering(args.input, args.output)
        print(f"转换成功！结果已保存至 {args.output}")
    except Exception as e:
        print(f"错误发生: {str(e)}")
