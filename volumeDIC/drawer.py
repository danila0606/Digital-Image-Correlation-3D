import numpy as np
import cv2

def draw_image_disps(image_path, points_ref, points_def, scale = 1., text = None, filename = None) :
  
    assert (points_ref.shape) == (points_def.shape), 'The shape of reference points array must be the same as the shape of deformed points array!'

    image = cv2.imread(image_path)
	
    if text is not None :
        image = cv2.putText(image, text, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),4)
		 
    for pt in points_ref:
        if not np.isnan(pt[0]) and not np.isnan(pt[1]):
            x = int(pt[0])
            y = int(pt[1])
            image = cv2.circle(image, (x, y), 4, (0, 255, 255), -1)
				
    for i, pt0 in enumerate(points_ref):
        pt1 = points_def[i]
        if np.isnan(pt0[0])==False and np.isnan(pt0[1])==False and np.isnan(pt1[0])==False and np.isnan(pt1[1])==False :
            disp_x = (pt1[0]-pt0[0])*scale
            disp_y = (pt1[1]-pt0[1])*scale
            image = cv2.line(image, (int(pt0[0]), int(pt0[1])), (int(pt0[0]+disp_x), int(pt0[1]+disp_y)), (255, 120, 255) , 2)
				
    if filename is not None:
        cv2.imwrite(filename, image)
        return

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', image.shape[1], image.shape[0])
    cv2.imshow('image',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()