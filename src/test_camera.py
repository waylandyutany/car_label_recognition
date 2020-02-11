import cv2, argparse, os
from datetime import datetime
from image_acquisition.camera_image import CameraImage

cam_res = {'720p':(1280,720),
           '1080p':(1920,1080)}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--cam-device-index', type=int, default=0, help='')
    parser.add_argument('--cam-resolution', default='720p', type=str, choices=[key for key in cam_res], help='')
    parser.add_argument('--out-folder', type=str, default='data/camera_images', help='')
    args = parser.parse_args()

    (width,height) = cam_res[args.cam_resolution]
    cam_image = CameraImage(width, height, args.cam_device_index)
    output_folder = os.path.join(args.out_folder, "{}_{}".format(datetime.now().strftime("%Y_%m_%d_%H_%M_%S"), args.cam_resolution))

    print("Camera resolution   : '{}' : {}".format(args.cam_resolution, cam_image.gray_image.shape))
    print("Camera device index : {}".format(args.cam_device_index))
    print("Output folder       : '{}'".format(output_folder))
    print("Press 'ESC' to exit.")
    print("Press 'SPACE' to store picture.")

    index = 0
    while True:
        gray_image = cam_image.gray_image
        cv2.imshow('cam_grey_image', gray_image)

        key_value = cv2.waitKey(1)
        if key_value == 27:
            break
        elif key_value == 32:
            if not os.path.isdir(output_folder):
                os.makedirs(output_folder)
            file_path = os.path.join(output_folder, "{}_{:03d}.jpg".format(datetime.now().strftime("%Y_%m_%d_%H_%M_%S"), index))
            cv2.imwrite(file_path, gray_image)
            print(file_path)
            index += 1

    del cam_image
    cv2.destroyAllWindows()
