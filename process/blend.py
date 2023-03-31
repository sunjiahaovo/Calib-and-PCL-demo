from PIL import Image
import imageio
import cv2
import numpy as np
import copy

def blend_two_images():
    img1 = Image.open("data1/demo/0.jpg")
    img1 = img1.convert('RGBA') # 根据需求是否转换颜色空间
    print(img1.size)
    # img1 = img1.resize((1280,720)) # 注意，是2个括号
    # print(img1.size)

    img2 = Image.open("data1/demo/100.jpg")
    img2 = img2.convert('RGBA')
    print(img2.size)
    # img2 = img2.resize((1280,720)) # 注意，是2个括号
    # print(img2.size)

    r, g, b, alpha = img2.split()
    alpha = alpha.point(lambda i: i > 0 and 204)

    img = Image.blend(img2, img1, 0.1)

    img.show()
    img.save("blend1122.png") # 注意jpg和png，否则 OSError: cannot write mode RGBA as JPEG

    return img

if __name__ == "__main__":
    
    # img1 = cv2.imread("data1/demo/0.jpg")
    # # img1 = img1.convert('RGBA') # 根据需求是否转换颜色空间
    # print(img1[0:368,0:150,:].shape)
    
    # img2 = cv2.imread("data1/demo/100.jpg")
    # img1[100:368,0:150,:] = img2[100:368,0:150,:]
    # cv2.imshow("img",img1)
    # cv2.waitKey(0)
    
    # blend_two_images()
    
    boxes = np.loadtxt("box.txt").astype(np.uint32)
    print(boxes)
    img = imageio.imread("data1/demo/0.jpg")
    img2 = imageio.imread("data1/demo/50.jpg")
    img3 = imageio.imread("data1/demo/200.jpg")
    img4 = imageio.imread("data1/demo/300.jpg")
    img5 = imageio.imread("data1/demo/350.jpg")
    img6 = imageio.imread("data1/demo/550.jpg")
    img7 = imageio.imread("data1/demo/650.jpg")
    
    
    img[boxes[1,0]:boxes[1,1],boxes[1,2]:boxes[1,3],:] =  copy.deepcopy(img2[boxes[1,0]:boxes[1,1],boxes[1,2]:boxes[1,3]])
    img2[boxes[0,0]:boxes[0,1],boxes[0,2]:boxes[0,3],:] =  copy.deepcopy(img[boxes[0,0]:boxes[0,1],boxes[0,2]:boxes[0,3]])
    img = Image.fromarray(img.astype('uint8')).convert('RGB')
    img2 = Image.fromarray(img2.astype('uint8')).convert('RGB')
    img = Image.blend(img2, img, 0.5)
    
    
    img = np.array(img)
    img[boxes[2,0]:boxes[2,1],boxes[2,2]:boxes[2,3],:] =  copy.deepcopy(img3[boxes[2,0]:boxes[2,1],boxes[2,2]:boxes[2,3]])
    img3[boxes[0,0]:boxes[0,1],boxes[0,2]:boxes[0,3],:] =  copy.deepcopy(img[boxes[0,0]:boxes[0,1],boxes[0,2]:boxes[0,3]])
    img3[boxes[1,0]:boxes[1,1],boxes[1,2]:boxes[1,3],:] =  copy.deepcopy(img[boxes[1,0]:boxes[1,1],boxes[1,2]:boxes[1,3]])
    img = Image.fromarray(img.astype('uint8')).convert('RGB')
    img3 = Image.fromarray(img3.astype('uint8')).convert('RGB')
    img = Image.blend(img3, img, 0.5)
    
    
    
    img = np.array(img)
    img[boxes[3,0]:boxes[3,1],boxes[3,2]:boxes[3,3],:] =  copy.deepcopy(img4[boxes[3,0]:boxes[3,1],boxes[3,2]:boxes[3,3]])
    img4[boxes[0,0]:boxes[0,1],boxes[0,2]:boxes[0,3],:] =  copy.deepcopy(img[boxes[0,0]:boxes[0,1],boxes[0,2]:boxes[0,3]])
    img4[boxes[1,0]:boxes[1,1],boxes[1,2]:boxes[1,3],:] =  copy.deepcopy(img[boxes[1,0]:boxes[1,1],boxes[1,2]:boxes[1,3]])
    img4[boxes[2,0]:boxes[2,1],boxes[2,2]:boxes[2,3],:] =  copy.deepcopy(img[boxes[2,0]:boxes[2,1],boxes[2,2]:boxes[2,3]])
    img = Image.fromarray(img.astype('uint8')).convert('RGB')
    img4 = Image.fromarray(img4.astype('uint8')).convert('RGB')
    img = Image.blend(img4, img, 0.5)
    
    
    img = np.array(img)
    img[boxes[4,0]:boxes[4,1],boxes[4,2]:boxes[4,3],:] =  copy.deepcopy(img5[boxes[4,0]:boxes[4,1],boxes[4,2]:boxes[4,3]])
    img5[boxes[0,0]:boxes[0,1],boxes[0,2]:boxes[0,3],:] =  copy.deepcopy(img[boxes[0,0]:boxes[0,1],boxes[0,2]:boxes[0,3]])
    img5[boxes[1,0]:boxes[1,1],boxes[1,2]:boxes[1,3],:] =  copy.deepcopy(img[boxes[1,0]:boxes[1,1],boxes[1,2]:boxes[1,3]])
    img5[boxes[2,0]:boxes[2,1],boxes[2,2]:boxes[2,3],:] =  copy.deepcopy(img[boxes[2,0]:boxes[2,1],boxes[2,2]:boxes[2,3]])
    img5[boxes[3,0]:boxes[3,1],boxes[3,2]:boxes[3,3],:] =  copy.deepcopy(img[boxes[3,0]:boxes[3,1],boxes[3,2]:boxes[3,3]])
    img = Image.fromarray(img.astype('uint8')).convert('RGB')
    img5 = Image.fromarray(img5.astype('uint8')).convert('RGB')
    img = Image.blend(img5, img, 0.5)
    
    
    img = np.array(img)
    img[boxes[5,0]:boxes[5,1],boxes[5,2]:boxes[5,3],:] =  copy.deepcopy(img6[boxes[5,0]:boxes[5,1],boxes[5,2]:boxes[5,3]])
    img6[boxes[0,0]:boxes[0,1],boxes[0,2]:boxes[0,3],:] =  copy.deepcopy(img[boxes[0,0]:boxes[0,1],boxes[0,2]:boxes[0,3]])
    img6[boxes[1,0]:boxes[1,1],boxes[1,2]:boxes[1,3],:] =  copy.deepcopy(img[boxes[1,0]:boxes[1,1],boxes[1,2]:boxes[1,3]])
    img6[boxes[2,0]:boxes[2,1],boxes[2,2]:boxes[2,3],:] =  copy.deepcopy(img[boxes[2,0]:boxes[2,1],boxes[2,2]:boxes[2,3]])
    img6[boxes[3,0]:boxes[3,1],boxes[3,2]:boxes[3,3],:] =  copy.deepcopy(img[boxes[3,0]:boxes[3,1],boxes[3,2]:boxes[3,3]])
    img6[boxes[4,0]:boxes[4,1],boxes[4,2]:boxes[4,3],:] =  copy.deepcopy(img[boxes[4,0]:boxes[4,1],boxes[4,2]:boxes[4,3]])
    img = Image.fromarray(img.astype('uint8')).convert('RGB')
    img6 = Image.fromarray(img6.astype('uint8')).convert('RGB')
    img = Image.blend(img6, img, 0.5)
    
    
    img = np.array(img)
    img[boxes[6,0]:boxes[6,1],boxes[6,2]:boxes[6,3],:] =  copy.deepcopy(img7[boxes[6,0]:boxes[6,1],boxes[6,2]:boxes[6,3]])
    img7[boxes[0,0]:boxes[0,1],boxes[0,2]:boxes[0,3],:] =  copy.deepcopy(img[boxes[0,0]:boxes[0,1],boxes[0,2]:boxes[0,3]])
    img7[boxes[1,0]:boxes[1,1],boxes[1,2]:boxes[1,3],:] =  copy.deepcopy(img[boxes[1,0]:boxes[1,1],boxes[1,2]:boxes[1,3]])
    img7[boxes[2,0]:boxes[2,1],boxes[2,2]:boxes[2,3],:] =  copy.deepcopy(img[boxes[2,0]:boxes[2,1],boxes[2,2]:boxes[2,3]])
    img7[boxes[3,0]:boxes[3,1],boxes[3,2]:boxes[3,3],:] =  copy.deepcopy(img[boxes[3,0]:boxes[3,1],boxes[3,2]:boxes[3,3]])
    img7[boxes[4,0]:boxes[4,1],boxes[4,2]:boxes[4,3],:] =  copy.deepcopy(img[boxes[4,0]:boxes[4,1],boxes[4,2]:boxes[4,3]])
    img7[boxes[5,0]:boxes[5,1],boxes[5,2]:boxes[5,3],:] =  copy.deepcopy(img[boxes[5,0]:boxes[5,1],boxes[5,2]:boxes[5,3]])
    img = Image.fromarray(img.astype('uint8')).convert('RGB')
    img7 = Image.fromarray(img7.astype('uint8')).convert('RGB')
    img = Image.blend(img7, img, 0.5)
    
    
    # img.show()
    img = np.array(img)
    imageio.imwrite("img.png",img)
    
   
    
    
    # img = Image.blend(img4, img, 0.6)
    # img = Image.blend(img5, img, 0.65)
    # img.show()
