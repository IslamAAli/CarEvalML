import Config

def baseline_majority_guess(ds_data):
    classes_data = ds_data.car_class
    total_sample = ds_data.car_class.count()
    most_probable_class = classes_data.value_counts().argmax() +1
    most_probable_class_count = classes_data[classes_data == most_probable_class].shape[0]

    # print classes count
    print('\n\n[0-0] Base Line Report (Majority Guess):')
    print('====================================================================================')
    print('1-> unacc, 2-> acc, 3-> good, 4->vgood')
    print(classes_data.value_counts())
    print('=> Baseline Accuracy = ', most_probable_class_count/total_sample)

    return