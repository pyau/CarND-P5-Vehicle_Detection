import os
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip
from IPython.display import HTML

#def make_list_of_images():
basedir = 'vehicles/'
image_types = os.listdir(basedir)
cars = []
for imtype in image_types:
    cars.extend(glob.glob(basedir+imtype+'/*'))

print('Number of Vehicle Images found:', len(cars))
with open("cars.txt", "w") as f:
    for fn in cars:
        f.write(fn+'\n')

basedir = 'non-vehicles/'
image_types = os.listdir(basedir)
notcars = []
for imtype in image_types:
    notcars.extend(glob.glob(basedir+imtype+'/*'))

print('Number of Non-Vehicle Images found:', len(notcars))
with open("notcars.txt", "w") as f:
    for fn in notcars:
        f.write(fn+'\n')

def get_hog_features(img, orient, pix_per_cell, cell_per_block,
    vis=False, feature_vec=True):
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell,pix_per_cell),
            cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, visualise=vis,
            feature_vector=feature_vec)
        return features, hog_image
    else:
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell,pix_per_cell),
            cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, visualise=vis,
            feature_vector=feature_vec)
        return features

def bin_spatial(img, size=(32, 32)):
    color1 = cv2.resize(img[:,:,0], size).ravel()
    color2 = cv2.resize(img[:,:,1], size).ravel()
    color3 = cv2.resize(img[:,:,2], size).ravel()
    return np.hstack((color1, color2, color3))

def color_hist(img, nbins=32): #bins_range=(0,256)
    channel1_hist = np.histogram(img[:,:,0], bins=nbins)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins)
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    return hist_features

def extract_features(imgs, color_space='RGB', spatial_size=(32,32), hist_bins=32, orient=9,
    pix_per_cell=8, cell_per_block=2, hog_channel=0, spatial_feat=True, hist_feat=True, hog_feat=True):
    features = []
    for file in imgs:
        file_features = []
        image = mpimg.imread(file)
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(image)

        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        if hist_feat == True:
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
        if hog_feat == True:
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:,:,channel],
                        orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient,
                    pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    return features

def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], xy_window=(64, 64),
    xy_overlap=(0.5,0.5)):
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    nx_pix_per_step = np.int(xy_window[0]*(1-xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1-xy_overlap[1]))
    nx_windows = np.int(xspan/nx_pix_per_step) - 1
    ny_windows = np.int(yspan/ny_pix_per_step) - 1

    window_list = []
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            startx = xs * nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys * ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]

            window_list.append(((startx, starty), (endx, endy)))
    return window_list

def draw_boxes(img, bboxes, color=(0,0, 255), thick=6):
    imcopy = np.copy(img)
    for bbox in bboxes:
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    return imcopy

def single_img_features(img, color_space='RGB', spatial_size=(32,32), hist_bins=32,
    orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0, spatial_feat=True,
    hist_feat=True, hog_feat=True, vis=False):
    img_features = []
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(img)

    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        img_features.append(spatial_features)
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        img_features.append(hist_features)
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:,:,channel],
                    orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)
        else:
            if vis == True:
                hog_features, hog_image = get_hog_features(feature_image[:,:,hog_channel], orient,
                    pix_per_cell, cell_per_block, vis=True, feature_vec=True)
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient,
                    pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        img_features.append(hog_features)
    if vis == True:
        return np.concatenate(img_features), hog_image, spatial_features, hist_features
    else:
        return np.concatenate(img_features)

def search_windows(img, windows, clf, scaler, color_space='RGB',
    spatial_size=(32,32), hist_bins=32, hist_range=(0,256), orient=9,
    pix_per_cell=8, cell_per_block=2, hog_channel=0, spatial_feat=True,
    hist_feat=True, hog_feat=True):
    on_windows=[]
    for window in windows:
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64,64))
        features = single_img_features(test_img, color_space=color_space,
            spatial_size=spatial_size, hist_bins=hist_bins, orient=orient,
            pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
            hog_channel=hog_channel, spatial_feat=spatial_feat,
            hist_feat=hist_feat, hog_feat=hog_feat)
        test_features = scaler.transform(np.array(features).reshape(1,-1))
        prediction = clf.predict(test_features)
        if prediction == 1:
            on_windows.append(window)
    return on_windows

def visualize(fig, rows, cols, imgs, titles):
    for i, img in enumerate(imgs):
        plt.subplot(rows, cols, i+1)
        plt.title(i+1)
        img_dims = len(img.shape)
        if img_dims < 3:
            plt.imshow(img, cmap='hot')
            plt.title(titles[i])
        else:
            plt.imshow(img)
            plt.title(titles[i])
#    plt.show()


##make_list_of_images()
#------
car_ind = np.random.randint(0, len(cars))
notcar_ind = np.random.randint(0, len(notcars))

car_image=  mpimg.imread(cars[car_ind])
notcar_image = mpimg.imread(notcars[notcar_ind])

color_space = 'RGB'
orient = 6
pix_per_cell = 8
cell_per_block = 2
hog_channel = 0     #0,1,2,"ALL"
spatial_size = (16,16)
hist_bins = 16
spatial_feat = True
hist_feat = True
hog_feat = True

#------
def extract_car_notcar_features():
    car_features, car_hog_image, car_spatial_features, car_hist_features = single_img_features(car_image,
        color_space=color_space, spatial_size=spatial_size, hist_bins=hist_bins, orient=orient, 
        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel,
        spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat, vis=True)
    notcar_features, notcar_hog_image, notcar_spatial_features, notcar_hist_features = single_img_features(notcar_image,
        color_space=color_space, spatial_size=spatial_size, hist_bins=hist_bins, orient=orient,
        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel,
        spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat, vis=True)

    images=[car_image, car_hog_image, notcar_image, notcar_hog_image]
    titles = ['car image', 'car HOG image', 'notcar image', 'notcar HOG image']
    fig = plt.figure(figsize=(16,4))
    visualize(fig, 2, 4, images, titles)
    plt.subplot(2, 4, 5)
    plt.plot(car_spatial_features)
    plt.title('Car spatial features')
    plt.subplot(2, 4, 6)
    plt.plot(car_hist_features)
    plt.title('Car hist features')
    plt.subplot(2, 4, 7)
    plt.plot(notcar_spatial_features)
    plt.title('Not car spatial features')
    plt.subplot(2, 4, 8)
    plt.plot(notcar_hist_features)
    plt.title('Not car hist features')
    plt.show()

#extract_car_notcar_features()
#------

color_space = 'YCrCb'
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = "ALL"     #0,1,2,"ALL"
spatial_size = (32,32)
hist_bins = 32
spatial_feat = True
hist_feat = True
hog_feat = True


def training():
    t = time.time()
    n_samples = 1000
    random_idxs = np.random.randint(0, len(cars), n_samples)
    test_cars = cars#np.array(cars)[random_idxs]#cars
    test_notcars = notcars#np.array(notcars)[random_idxs]#notcars

    car_features = extract_features(test_cars, color_space=color_space,
        spatial_size=spatial_size, hist_bins=hist_bins, orient=orient,
        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
        hog_channel=hog_channel, spatial_feat=spatial_feat,
        hist_feat=hist_feat, hog_feat=hog_feat)
    notcar_features = extract_features(test_notcars, color_space=color_space,
        spatial_size=spatial_size, hist_bins=hist_bins, orient=orient,
        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
        hog_channel=hog_channel, spatial_feat=spatial_feat,
        hist_feat=hist_feat, hog_feat=hog_feat)

    print(time.time()-t, 'Seconds to compute features...')

    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    X_scaler = StandardScaler().fit(X)
    scaled_X = X_scaler.transform(X)

    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y,
        test_size=0.1, random_state=rand_state)

    print('Using:', orient, 'orientations,', pix_per_cell, 'pixels per cell,',
        cell_per_block, 'cells per block,', hist_bins,'histogram bins, and',
        spatial_size, 'spatial sampling')
    print('Feature vector length:', len(X_train[0]))

    svc = LinearSVC()
    t=time.time()
    svc.fit(X_train, y_train)
    print(round(time.time()-t, 2), 'Seconds to train SVC...')
    print('Test Accuract of SVC = ', round(svc.score(X_test, y_test), 4))
    return svc, X_scaler

#svc, X_scaler = training()
#svc_pickle = {}
#svc_pickle["svc"] = svc
#svc_pickle["X_scaler"] = X_scaler
#pickle.dump(svc_pickle, open("./svc.p", "wb"))

svc_pickle = pickle.load(open("./svc_32_32_32.p", "rb"))
svc = svc_pickle["svc"]
X_scaler = svc_pickle["X_scaler"]

searchpath='test_images/*'
example_images = glob.glob(searchpath)

def sliding_windows():
    images = []
    titles = []
    y_start_stop = [400, 656]
    overlap = 0.5
    for img_src in example_images:
        t1 = time.time()
        img = mpimg.imread(img_src)
        draw_img = np.copy(img)
        img = img.astype(np.float32)/255    # reading jpeg, trained with png
        print(np.min(img), np.max(img))

        windows = slide_window(img, x_start_stop=[None, None], y_start_stop=y_start_stop,
            xy_window=(96,96), xy_overlap=(overlap, overlap))

        hot_windows = search_windows(img, windows, svc, X_scaler, color_space=color_space,
            spatial_size=spatial_size, hist_bins=hist_bins, orient=orient,
            pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel,
            spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)

        window_img = draw_boxes(draw_img, hot_windows, color=(0,0,255), thick=6)
        images.append(window_img)
        titles.append('')
        print(time.time()-t1, 'seconds to process one image searching', len(windows), 'windows')
    fig = plt.figure(figsize=(12,18))#, dpi=300)
    visualize(fig, 5, 2, images, titles)
    plt.show()

#sliding_windows()

def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)

def compute_hog_for_whole_image_and_subsample():
    out_images = []
    out_maps = []
    out_titles = []
    out_boxes = []
    ystart = 400
    ystop = 656
    scale = 1.5

    for img_src in example_images:
        img_boxes = []
        t = time.time()
        count = 0
        img = mpimg.imread(img_src)
        draw_img = np.copy(img)
        heatmap = np.zeros_like(img[:,:,0])
        img = img.astype(np.float32)/255

        img_tosearch = img[ystart:ystop, :,:]
        ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
        if scale != 1:
            imshape = ctrans_tosearch.shape
            # if want diff scale, instead of diff size window, resize image 
            ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

        ch1 = ctrans_tosearch[:,:,0]
        ch2 = ctrans_tosearch[:,:,1]
        ch3 = ctrans_tosearch[:,:,2]

        nxblocks = (ch1.shape[1] // pix_per_cell) - 1
        nyblocks = (ch1.shape[0] // pix_per_cell) - 1
        nfeat_per_block = orient*cell_per_block**2
        window = 64
        nblocks_per_window = (window // pix_per_cell) - 1
        cells_per_step = 2  # instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step

        hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

        for xb in range(nxsteps):
            for yb in range(nysteps):
                count += 1
                ypos = yb * cells_per_step
                xpos = xb * cells_per_step
                hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

                xleft = xpos*pix_per_cell
                ytop = ypos*pix_per_cell

                subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))

                spatial_features = bin_spatial(subimg, size=spatial_size)
                hist_features = color_hist(subimg, nbins=hist_bins)

                test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
                test_prediction = svc.predict(test_features)

                if test_prediction == 1:
                    xbox_left = np.int(xleft*scale)
                    ytop_draw = np.int(ytop*scale)
                    win_draw = np.int(window*scale)
                    cv2.rectangle(draw_img, (xbox_left, ytop_draw+ystart), (xbox_left+win_draw, ytop_draw+win_draw+ystart), (0, 0, 255), 6)
                    img_boxes.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw, ytop_draw+win_draw+ystart)))
                    heatmap[ytop_draw+ystart:ytop_draw+win_draw+ystart, xbox_left:xbox_left+win_draw] += 1

        print(time.time()-t, 'seconds to run, total windows = ', count)

        out_images.append(draw_img)
        out_titles.append(img_src[-12:])
        out_titles.append(img_src[-12:])
        #heatmap = 255*heatmap/np.max(heatmap)
        out_images.append(heatmap)
        out_maps.append(heatmap)
        out_boxes.append(img_boxes)

    fig = plt.figure(figsize=(16,8))
    visualize(fig, 4,4,out_images, out_titles)
    plt.show()

#compute_hog_for_whole_image_and_subsample()

def find_cars(img, scale):
    draw_img = np.copy(img)
    heatmap = np.zeros_like(img[:,:,0])
    img = img.astype(np.float32)/255

    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = convert_color(img_tosearch, conv = 'RGB2YCrCb')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        # if want diff scale, instead of diff size window, resize image 
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    nxblocks = (ch1.shape[1] // pix_per_cell) - 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - 1
    nfeat_per_block = orient*cell_per_block**2
    window = 64
    nblocks_per_window = (window // pix_per_cell) - 1
    cells_per_step = 2  # instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step
            yend = ypos+nblocks_per_window
            xend = xpos+nblocks_per_window
            hog_feat1 = hog1[ypos:yend, xpos:xend].ravel()
            hog_feat2 = hog2[ypos:yend, xpos:xend].ravel()
            hog_feat3 = hog3[ypos:yend, xpos:xend].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))

            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                cv2.rectangle(draw_img, (xbox_left, ytop_draw+ystart), (xbox_left+win_draw, ytop_draw+win_draw+ystart), (0, 0, 255), 4)
                #img_boxes.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw, ytop_draw+win_draw+ystart)))
                heatmap[ytop_draw+ystart:ytop_draw+win_draw+ystart, xbox_left:xbox_left+win_draw] += 1

    return draw_img, heatmap


def apply_threshold(heatmap, threshold):
    heatmap[heatmap <= threshold] = 0
    return heatmap

def draw_labeled_bboxes(img, labels):
    for car_number in range(1, labels[1]+1):
        nonzero = (labels[0] == car_number).nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 4)
    return img

ystart = 360
ystop = 616
def find_labels():
    out_images = []
    out_maps = []
    out_titles = []
    scale = 1.5

    for img_src in example_images:
        img = mpimg.imread(img_src)
        out_img, heat_map = find_cars(img, scale)
        labels = label(heat_map)
        draw_img = draw_labeled_bboxes(np.copy(img), labels)
        out_images.append(draw_img)
        out_images.append(heat_map)
        out_titles.append(img_src[-12:])
        out_titles.append(img_src[-12:])

    fig = plt.figure(figsize=(12,24))
    visualize(fig, 4,4, out_images, out_titles)
    plt.show()
#find_labels()

heat_maps = []
def integrate_heat_maps(new_map):
    heat_maps.append(new_map)
    if len(heat_maps) > 30:
        heat_maps.pop(0)
    sum_map = np.zeros(new_map.shape)
    for map in heat_maps:
        sum_map = np.add(sum_map, map)
    return sum_map

def process_image(img):
    out_img1, heat_map1 = find_cars(img, 1.0)
    #out_img2, heat_map2 = find_cars(img, 1.5)
    out_img3, heat_map3 = find_cars(img, 2.0)
    heat_map = np.add(heat_map1, heat_map3)
    #heat_map = np.add(heat_map, heat_map3)

    # integrate across multiple frames
    integrate_heat_map = integrate_heat_maps(heat_map)
    integrate_heat_map = apply_threshold(integrate_heat_map, 17)
    labels = label(integrate_heat_map)
    draw_img = draw_labeled_bboxes(np.copy(img), labels)

    # draw out heat map
    #heat_map_rgb = np.zeros((heat_map.shape[0], heat_map.shape[1], 3), 'uint8')
    #heat_map_rgb[...,0] = integrate_heat_map*10
    #heat_map_rgb[...,1] = integrate_heat_map*10
    #heat_map_rgb[...,2] = integrate_heat_map*10
    return draw_img

def process_video():
    #test_output = 'test.mp4'
    #clip = VideoFileClip("test_video.mp4")
    test_output = 'project.mp4'
    clip = VideoFileClip("project_video.mp4")#.subclip(39,43)
    test_clip = clip.fl_image(process_image)
    test_clip.write_videofile(test_output, audio=False)
    #once scale 2, once scale 1, add all detections in the same heatmap

process_video()
