import imgaug.augmenters as iaa
import cv2
import os


def read_images(location, filelist):
    imgs = []
    for i in range(len(filelist)):
        img = cv2.imread(location + filelist[i])
        if img is not None:
            imgs.append(img)
    return imgs


def save_images(location, imgname, imgaug, num):
    for i in range(len(imgaug)):
        cv2.imwrite('%s%s\\%s_aug%d.jpg'%(location, imgname[:imgname.find("_")], imgname[:imgname.find("_")], i * 15 + num + 1), imgaug[i])


def imageaug(img):
    seq1 = iaa.MotionBlur(k=(10, 30))
    seq2 = iaa.PerspectiveTransform(scale=(0.01, 0.15))
    seq3 = iaa.MultiplyAndAddToBrightness(mul=(0.5, 1.5), add=(-30, 30))
    seq4 = iaa.Fliplr(1)
    seq5 = iaa.Flipud(1)
    seq6 = iaa.Affine(rotate=(-1, -89))
    seq7 = iaa.Affine(rotate=(1, 89))
    seq8 = iaa.Affine(rotate=(-91, -179))
    seq9 = iaa.Affine(rotate=(91, 179))
    seq10 = iaa.Affine(shear=(-16, 16))
    seq11 = iaa.LinearContrast((0.4, 0.7), per_channel=True)
    seq12 = iaa.PiecewiseAffine(scale=(0.01, 0.05))
    seq13 = iaa.BilateralBlur(d=(3, 10), sigma_color=(10, 250), sigma_space=(10, 250))
    seq14 = iaa.MedianBlur(k=(3, 5))
    seq15 = iaa.CropAndPad(percent=(-0.25, 0.25))
    
    img1 = seq1.augment_image(img)
    img2 = seq2.augment_image(img)
    img3 = seq3.augment_image(img)
    img4 = seq4.augment_image(img)
    img5 = seq5.augment_image(img)
    img6 = seq6.augment_image(img)
    img7 = seq7.augment_image(img)
    img8 = seq8.augment_image(img)
    img9 = seq9.augment_image(img)
    img10 = seq10.augment_image(img)
    img11 = seq11.augment_image(img)
    img12 = seq12.augment_image(img)
    img13 = seq13.augment_image(img)
    img14 = seq14.augment_image(img)
    img15 = seq15.augment_image(img)

    imgiaa = [img1, img2, img3, img4, img5, img6, img7, img8, img9, img10, img11, img12, img13, img14, img15]
    return imgiaa


def main():
    location_read = "./image_aug\\"
    location_save = "./image\\"
    filelist = os.listdir(location_read)

    images_origin = read_images(location_read, filelist)

    for j in range(2):
        for i in range(len(images_origin)):
            image_aug = imageaug(images_origin[i])
            if i%15 == 0:
                print("save complete #%d-%d"%(j+1, i//15 + 1))
            num = j * 15 * 15 + (i%15)
            save_images(location_save, filelist[i].replace(".jpg", ""), image_aug, num)
        print("save complete all #%d"%(j+1))


if __name__ == "__main__":
    main()