from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
from keras.models import Model
import numpy as np
import time
import cv2


def collect_demo(path, num_patch, aux_dim, action_dim):

    for i in xrange(num_patch):
        path_patch = path + str(i) + "/"
        demo_name = path_patch + "demo.txt"
        demo_raw = open(demo_name, 'r').readlines()
        state_name = path_patch + "states.txt"
        state_raw = open(state_name, 'r').readlines()

        pa = np.zeros(6, dtype=np.float32)

        print "Loading patch %d ..." % i
        for j in xrange(0, len(demo_raw)):
            action_data = np.array(demo_raw[j].strip().split(" ")).astype(np.float32)
            state_data  = np.array(state_raw[j].strip().split(" ")).astype(np.float32)

            aux = np.expand_dims([state_data[-3], state_data[-1]], axis=0).astype(np.float32)
            action = np.expand_dims(action_data[:], axis=0).astype(np.float32)
            
            img_path = path_patch + str(j) + ".jpg"
            img = image.load_img(img_path)
            img = image.img_to_array(img)
            img = cv2.resize(img, (256, 256))
            #img = img[40:, :, :]

            '''
            if j < 130 and i == 1:
                img_cv2 = cv2.imread(img_path)
                img_cv2 = cv2.resize(img_cv2, (200, 150))
                img_cv2 = img_cv2[40:, :, :]
                cv2.imshow('image', cv2.cvtColor(img, cv2.COLOR_RGB2BGR)/255.0)
                cv2.waitKey(0)
            '''
            img = np.expand_dims(img, axis=0).astype(np.uint8)


            if j == 0:
                auxs_tmp = aux
                actions_tmp = action
                imgs_tmp = img
            else:
                auxs_tmp = np.concatenate((auxs_tmp, aux), axis=0)
                actions_tmp = np.concatenate((actions_tmp, action), axis=0)
                imgs_tmp = np.concatenate((imgs_tmp, img), axis=0)

        if i == 0:
            auxs = auxs_tmp
            actions = actions_tmp
            imgs = imgs_tmp
        else:
            auxs = np.concatenate((auxs, auxs_tmp), axis=0)
            actions = np.concatenate((actions, actions_tmp), axis=0)
            imgs = np.concatenate((imgs, imgs_tmp), axis=0)

        print "Current total:", imgs.shape, auxs.shape, actions.shape

    print "Images:", imgs.shape, "Auxs:", auxs.shape, "Actions:", actions.shape

    return imgs, auxs, actions


def normalize(x):
    x[:, 0:4] /= 200.
    return x


def main():
    aux_dim = 66
    action_dim = 3
    num_patch = 240
    #demo_path = "/home/yunzhu/Desktop/human_low_case_1/demo_"
    demo_path = "/home/zhiyang/Desktop/intention/reacher/rl_demo/demo_"

    imgs, auxs, actions = collect_demo(demo_path, num_patch, aux_dim, action_dim)
    auxs = normalize(auxs)

    np.savez_compressed("/home/zhiyang/Desktop/intention/reacher/rl_demo/demo.npz",
                        imgs=imgs, auxs=auxs, actions=actions)
    print "Finished."


if __name__ == "__main__":
    main()
