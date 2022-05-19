import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


transform_train = A.Compose([
    A.RandomResizedCrop(width=224, height=224, scale=[0.75, 1.0], p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    # ToTensorV2()
])

transform_test = A.Compose([
    # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    # ToTensorV2()
])

transform_replay = A.ReplayCompose([
    A.RandomResizedCrop(width=224, height=224, scale=[0.75, 1.0], p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    # ToTensorV2()
])

transform_same = A.Compose([
    A.RandomResizedCrop(width=224, height=224, scale=[0.75, 1.0], p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    # ToTensorV2()
    ],
    additional_targets={f'img{i}': 'image' for i in range(1, 32)}
)

img_folder_path = 'visualize/augmentations/'
img_face = cv2.imread(img_folder_path+'face.png')
img_body = cv2.imread(img_folder_path+'body.png')

for i in range(5):
    # # face
    # transformed_img_train = transform_train(image=img_face)['image']
    # transformed_img_test = transform_test(image=img_face)['image']
    # cv2.imwrite(img_folder_path+f'face_train_{i+1}.png', transformed_img_train)
    # # cv2.imwrite(img_folder_path+f'face_test_{i+1}.png', transformed_img_test)
    
    # # body
    # transformed_img_train = transform_train(image=img_body)['image']
    # transformed_img_test = transform_test(image=img_body)['image']
    # cv2.imwrite(img_folder_path+f'body_train_{i+1}.png', transformed_img_train)
    # # cv2.imwrite(img_folder_path+f'body_test_{i+1}.png', transformed_img_test)

    # # replay
    # if i == 0:
    #     T_replay = transform_replay(image=img_face)
    # transformed_img_replay = A.ReplayCompose.replay(T_replay['replay'], image=img_face)['image']
    # cv2.imwrite(img_folder_path+f'replay_{i+1}.png', transformed_img_replay)

    # same
    transformed_imgs_same = transform_same(
        image=img_body, img1=img_body,  img2=img_body,  img3=img_body,
        img4=img_body,  img5=img_body,  img6=img_body,  img7=img_body,
        img8=img_body,  img9=img_body,  img10=img_body, img11=img_body,
        img12=img_body, img13=img_body, img14=img_body, img15=img_body,
        img16=img_body, img17=img_body, img18=img_body, img19=img_body,
        img20=img_body, img21=img_body, img22=img_body, img23=img_body,
        img24=img_body, img25=img_body, img26=img_body, img27=img_body,
        img28=img_body, img29=img_body, img30=img_body, img31=img_body
    )
    cv2.imwrite(img_folder_path+f'body_same_{i+1}_{0}.png', transformed_imgs_same['image'])
    for j in range(1, 32):
        cv2.imwrite(img_folder_path+f'body_same_{i+1}_{j}.png', transformed_imgs_same[f'img{j}'])

    transformed_imgs_same = transform_same(
        image=img_face, img1=img_face,  img2=img_face,  img3=img_face
    )
    cv2.imwrite(img_folder_path+f'face_same_{i+1}_{0}.png', transformed_imgs_same['image'])
    for j in range(1, 4):
        cv2.imwrite(img_folder_path+f'face_same_{i+1}_{j}.png', transformed_imgs_same[f'img{j}'])

    break
