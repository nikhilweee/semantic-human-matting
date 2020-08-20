# Adapted from https://github.com/lizhengwei1992/Semantic_Human_Matting/blob/master/data/gen_trimap.py

import cv2
import numpy as np
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Create trimaps from mattes.')
    parser.add_argument('--matte-dir', type=str, default='mattes', help="mattes directory")
    parser.add_argument('--save-dir', type=str, default='trimaps', help="where trimap result save to")
    parser.add_argument('--list', type=str, default='images.txt', help="list of image names")
    parser.add_argument('--size', type=int, default=10, help="kernel size")
    args = parser.parse_args()
    print(args)
    return args

def erode_dilate(matte, struc="ELLIPSE", size=(10, 10)):
    if struc == "RECT":
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, size)
    elif struc == "CORSS":
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, size)
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size)

    # Binarize image
    _, matte = cv2.threshold(matte, 127, 1, cv2.THRESH_BINARY)

    #matte = matte.astype(np.float32)
    # matte = matte / 255
    #matte = matte.astype(np.uint8)

    # val in 0 or 255

    dilated = cv2.dilate(matte, kernel, iterations=1) * 255
    eroded = cv2.erode(matte, kernel, iterations=1) * 255

    count1 = len(np.where(matte >= 0)[0])
    count2 = len(np.where(matte == 0)[0])
    count3 = len(np.where(matte == 1)[0])
    #print("all:{} bg:{} fg:{}".format(count1, count2, count3))
    assert(count1 == count2 + count3)

    count1 = len(np.where(dilated >= 0)[0])
    count2 = len(np.where(dilated == 0)[0])
    count3 = len(np.where(dilated == 255)[0])
    #print("all:{} bg:{} fg:{}".format(count1, count2, count3))
    assert(count1 == count2 + count3)

    count1 = len(np.where(eroded >= 0)[0])
    count2 = len(np.where(eroded == 0)[0])
    count3 = len(np.where(eroded == 255)[0])
    #print("all:{} bg:{} fg:{}".format(count1, count2, count3))
    assert(count1 == count2 + count3)

    result = dilated.copy()
    #result[((dilated == 255) & (matte == 0))] = 128
    result[((dilated == 255) & (eroded == 0))] = 128

    return result

def main():
    args = get_args()
    with open(args.list, 'r') as f:
        names = f.readlines()

    print(f"Images Count: {len(names)}")
    for name in names:
        matte_name = args.matte_dir + "/" + name.strip()[:-4] + ".png"
        print(f"Reading {matte_name}")
        trimap_name = args.save_dir + "/" + name.strip()[:-4] + ".png"
        matte = cv2.imread(matte_name, cv2.IMREAD_UNCHANGED)
        trimap = erode_dilate(matte, size=(args.size, args.size))

        print(f"Writing {trimap_name}")
        cv2.imwrite(trimap_name, trimap)

if __name__ == "__main__":
    main()


