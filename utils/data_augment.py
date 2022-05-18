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
    additional_targets={f'image{i+2}': 'image' for i in range(31)} # image2 - image32
)

img_folder_path = 'visualize/augmentations/'
img_face = cv2.imread(img_folder_path+'face.png')
img_body = cv2.imread(img_folder_path+'body.png')

for i in range(5):
    # face
    transformed_img_train = transform_train(image=img_face)['image']
    transformed_img_test = transform_test(image=img_face)['image']
    cv2.imwrite(img_folder_path+f'face_train_{i+1}.png', transformed_img_train)
    # cv2.imwrite(img_folder_path+f'face_test_{i+1}.png', transformed_img_test)
    
    # body
    transformed_img_train = transform_train(image=img_body)['image']
    transformed_img_test = transform_test(image=img_body)['image']
    cv2.imwrite(img_folder_path+f'body_train_{i+1}.png', transformed_img_train)
    # cv2.imwrite(img_folder_path+f'body_test_{i+1}.png', transformed_img_test)

    # replay
    if i == 0:
        T_replay = transform_replay(image=img_face)
    transformed_img_replay = A.ReplayCompose.replay(T_replay['replay'], image=img_face)['image']
    cv2.imwrite(img_folder_path+f'replay_{i+1}.png', transformed_img_replay)

# same
transformed_imgs_same = transform_same(image=img_body, image2=img_body, image3=img_body)
cv2.imwrite(img_folder_path+f'same_{1}.png', transformed_imgs_same['image'])
for i in range(2):
    cv2.imwrite(img_folder_path+f'same_{i+2}.png', transformed_imgs_same[f'image{i+2}'])
