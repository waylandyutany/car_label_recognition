import cv2, glob2, os

from core.letter_recognitor import LetterRecognitor
from core.labels_regions import LabelsRegions
from image_acquisition.file_image import FileImage

#letter_recognitor = LetterRecognitor.load_or_create("data/recognitors", (64, 128))
letter_recognitor = LetterRecognitor.load_or_create("data/recognitors", (32, 64))

def draw_labels_to_image(labels, image, rec_color, tx_color):
    for label in labels:
        for cbb in label.cbbs:
            (x, y, w, h) = cbb.bbox
            cv2.putText(image, cbb.param, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, tx_color, 2, cv2.LINE_AA)
            cv2.rectangle(image, (x, y), (x+w, y+h), rec_color, 2)

def draw_contours_to_image(contours, image, rec_color):
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x+w, y+h), rec_color, 2)

def sec_to_ms(sec):
    return int(sec*1000)

################################################################################
def main():
    file_names = glob2.glob("data/spz_pictures/*.jpg")
    file_index = len(file_names) - 1

    print("Using optimized CV2." if cv2.useOptimized() else "Using not-optimized CV2!")
    print("Press ESC to quit.")
    while True:
        file_path = file_names[file_index]
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        labels_regions = LabelsRegions(FileImage(file_path), file_path, letter_recognitor)
        labels = labels_regions.detected_labels()

        label_text = ", ".join((label.text for label in labels))
        print(label_text)
        print("Time CTS({}ms), FLT({}ms), LBS({}ms), TOT({}ms)".format(
                        sec_to_ms(labels_regions.time_contours_detection), 
                        sec_to_ms(labels_regions.time_contours_filtering), 
                        sec_to_ms(labels_regions.time_labels_detection),
                        sec_to_ms(labels_regions.time_total_detection)))

        #labels_regions.save_labels_recognitions("data/to_categorize", file_name, labels, (64,128))
        #labels_regions.save_labels_recognitions("data/to_categorize", file_name, labels, (32,64))
        #labels_regions.save_labels_recognitions("data/to_categorize", file_name, labels, (20,40))

        image = cv2.imread(file_path)
        draw_labels_to_image(labels, image, (255,255,255), (255,255,255))
        #draw_contours_to_image(labels_regions.filtered_countours(), image, (255,255,255))
        #draw_contours_to_image(labels_regions.all_countours(), image, (255,255,255))

        
        cv2.imshow('dbg_image', labels_regions.construct_debug_image(labels_regions.filtered_contours()))
        cv2.imshow('src_image', image)
        key_value = cv2.waitKey(0)

        if key_value == 27:#Esc
            break
        elif key_value == 52:#Left(4)
            pass
        elif key_value == 54:#Right(6)
            pass
        elif key_value == 56:#Up(8)
            file_index = (file_index + 1) % len(file_names)
        elif key_value == 50:#Down(2)
            file_index = file_index - 1
            file_index = len(file_names) - 1 if file_index < 0 else file_index

    cv2.destroyAllWindows()

main()
#letter_recognitor.learn_from_samples("data/to_learn")
#letter_recognitor.save("data/my_model.h5")
