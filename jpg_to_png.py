import os
import shutil

png_path = "./img_datas/train/PNGImages"  # png格式图片所在文件夹的路径
jpg_path = "./img_datas/train/JPEGImages"  # jpg格式图片存放文件夹的路径
file_walk = os.listdir(jpg_path)
fileNum = 0         # png文件夹下所有文件个数计数
png_fileNum = 0     # png文件夹下png图片个数计数

for filePath in file_walk:
    fileNum += 1
    protion = os.path.splitext(filePath)

    if protion[1].lower() == '.jpg':  # 判断文件后缀是否为png
        if png_fileNum == 0:  # 当png文件夹中有png图片
            # 判断是否存在jpg文件夹，存在则清空文件夹，不存在就建立文件夹
            if os.path.exists(png_path):
                shutil.rmtree(png_path)
                os.mkdir(png_path)
                print("jpg文件夹内原文件已清除")
            else:
                os.mkdir(png_path)
                print("jpg文件夹已创建")
        png_fileNum += 1
        print("正在处理：" + filePath)

        # 复制转存png图片为jpg格式到jpg文件夹中
        shutil.copyfile(os.path.join(jpg_path, filePath), os.path.join(png_path, protion[0] + '.png'))

print('\n文件夹内共有' + str(fileNum) + '个文件，其中png格式文件有' + str(png_fileNum) + '个，已全部完成转换，存至jpg文件夹内')
# stop = input("\n\n请按回车键退出！")    # 暂停查看信息
