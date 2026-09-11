import numpy as np
from scipy import spatial


def get_rectangle_bounds(event_bounds, angle, im_length, one_rot=None, two_rot=None, three_rot=None, four_rot=None):
    '''
    AIM: return xmin, xmax, ymin, ymax of rectangle
    '''
    #use user-drawn rectangle in order to define xmin, xmax; ymin, ymax. if no rectangle drawn, then default to image width for x and some fraction of the height for y.
    try:
        #for the case where the rectangle is not rotated...
        if angle == 0:
            xmin = min(int(event_bounds[0]), int(event_bounds[2]))
            xmax = max(int(event_bounds[0]), int(event_bounds[2]))
            ymin = min(int(event_bounds[1]), int(event_bounds[3]))
            ymax = max(int(event_bounds[1]), int(event_bounds[3]))
        #rectangle IS rotated
        else:
            xvertices = np.array([one_rot[0], two_rot[0], three_rot[0], four_rot[0]])
            yvertices = np.array([one_rot[1], two_rot[1], three_rot[1], four_rot[1]])
            xmin = np.min(xvertives)
            xmax = np.max(xvertices)
            ymin = np.min(yvertices)
            ymax = np.max(yvertices)
    except:
        print('Defaulting to image parameters for xmin, xmax; ymin, ymax.')
        xmin = 0
        xmax = im_length
        ymin = int(im_length/2 - (0.20*im_length))
        ymax = int(im_length/2 + (0.20*im_length))
    
    return xmin, xmax, ymin, ymax
    

#from https://stackoverflow.com/questions/34372480/rotate-point-about-another-point-in-degrees-python
def rotate(point, angle, center=(0,0)):
    '''
    AIM: rotate a point by angle (degree) about the center coordinate.
    point - point to be rotated
    angle - the angle (in degrees) by which to rotate the point
    '''
    angle_rad = angle * np.pi / 180.
    
    xnew = (np.cos(angle_rad) * (point[0] - center[0]) - np.sin(angle_rad) * (point[1] - center[1]) + center[0])
    ynew = (np.sin(angle_rad) * (point[0] - center[0]) + np.cos(angle_rad) * (point[1] - center[1]) + center[1])
    
    return round(xnew, 2), round(ynew, 2)


#extract x and y vertex coordinates, and the slope of the lines connecting these points
#this function "returns" a dictionary of (x,y) vertices, and the slopes of the rectangle perimeter lines
#NOTE: I choose np.max(x)-np.min(x) as the number of elements comprising each equation. seems fine.
def get_xym(event_bounds, angle):
    '''
    AIM: RETURN (ROTATED) RECTANGLE!
    event_bounds: flattened array of coordinates for the "click" events defining the initial rectangle
    '''
    #event_bounds contains the list [x1,y1,x2,y2]
    p1 = [event_bounds[0],event_bounds[1]]   #coordinates of first click event
    p2 = [event_bounds[2],event_bounds[3]]   #coordinates of second click event

    if angle%90 != 0:      #if angle is not divisible by 90, can rotate using this algorithm. 

        n_spaces = int(np.abs(p1[0] - p2[0]))   #number of 'pixels' between x coordinates

        (xc,yc) = ((p1[0]+p2[0])/2, (p1[1]+p2[1])/2)
        one_rot = rotate(p1, angle, center = (xc,yc))
        two_rot = rotate(p2, angle, center = (xc,yc))
        three_rot = rotate((p1[0],p2[1]), angle, center = (xc,yc))
        four_rot = rotate((p2[0],p1[1]), angle, center = (xc,yc))

        x1 = np.linspace(one_rot[0],three_rot[0],n_spaces)
        m1 = (one_rot[1] - three_rot[1])/(one_rot[0] - three_rot[0])
        y1 = three_rot[1] + m1*(x1 - three_rot[0])

        x2 = np.linspace(one_rot[0],four_rot[0],n_spaces)
        m2 = (one_rot[1] - four_rot[1])/(one_rot[0] - four_rot[0])
        y2 = four_rot[1] + m2*(x2 - four_rot[0])

        x3 = np.linspace(two_rot[0],three_rot[0],n_spaces)
        m3 = (two_rot[1] - three_rot[1])/(two_rot[0] - three_rot[0])
        y3 = two_rot[1] + m3*(x3 - two_rot[0])

        x4 = np.linspace(two_rot[0],four_rot[0],n_spaces)
        m4 = (two_rot[1] - four_rot[1])/(two_rot[0] - four_rot[0])
        y4 = two_rot[1] + m4*(x4 - two_rot[0])

        x_rot = [x1,x2,x3,x4]
        y_rot = [y1,y2,y3,y4]
        m_rot = [m1,m2,m3,m4]
        n_spaces = n_spaces

        one_rot = one_rot
        two_rot = two_rot
        three_rot = three_rot
        four_rot = four_rot

    elif (angle/90)%2 == 0:  #if angle is divisible by 90 but is 0, 180, 360, ..., no change to rectangle

        n_spaces = int(np.abs(p1[0] - p2[0]))   #number of 'pixels' between x coordinates

        x1 = np.zeros(50)+p1[0]
        y1 = np.linspace(p2[1],p1[1],n_spaces)

        x2 = np.linspace(p1[0],p2[0],n_spaces)
        y2 = np.zeros(50)+p2[1]

        x3 = np.linspace(p2[0],p1[0],n_spaces)
        y3 = np.zeros(50)+p1[1]

        x4 = np.zeros(50)+p2[0]
        y4 = np.linspace(p1[1],p2[1],n_spaces)

        x_rot = [x1,x2,x3,x4]
        y_rot = [y1,y2,y3,y4]
        m_rot = [0,0,0,0]
        n_spaces = n_spaces

        one_rot = p1
        two_rot = p2
        three_rot = (p1[0],p2[1])
        four_rot = (p2[0],p1[1])
        
    return {'x_rot': x_rot,
           'y_rot': y_rot,
           'm_rot': m_rot,
           'one_rot': one_rot,
           'two_rot': two_rot,
           'three_rot': three_rot,
           'four_rot': four_rot,
           'n_spaces': n_spaces}


def sample_rotated_rectangle(image, event_bounds, angle, image_alt=None):

    '''
    AIM: return list of average pixel values, the coordinates for all lines within the rectangle, and the image_alt average pixel values (if applicable).
    * more specifically, this function will sample a rotated rectangle through an image.
    '''
    
    xym_dict = get_xym(event_bounds, angle)   #defines and initiates x_rot, y_rot, m_rot
    x_rot = xym_dict['x_rot']
    y_rot = xym_dict['y_rot']
    m_rot = xym_dict['m_rot']
    n_spaces = xym_dict['n_spaces']

    #create lists
    list_to_mean = []
    mean_list = []   #only need to initialize once
    all_line_coords = []   #also only need to initialize once --> will give x,y coordinates for every line (hence the variable name).

    calc_alt = image_alt is not None
    mean_list_alt = [] if calc_alt else None
    list_to_mean_alt = [] if calc_alt else None
    
    for i in range(n_spaces):  #for the entire n_spaces extent: 
                               #find x range of between same-index points on opposing sides of the 
                               #rectangle, determine the equation variables to 
                               #connect these elements within this desired x range, 
                               #then find this line's mean pixel value. 
                               #proceed to next set of elements, etc.

        #points from either x4,y4 (index=3) or x1,y1 (index=0)
        #any angle which is NOT 0,180,360,etc.
        if angle%90 != 0:
            xpoints = np.linspace(x_rot[3][i],x_rot[0][-(i+1)],n_spaces)
            b = y_rot[0][-(i+1)] - (m_rot[2]*x_rot[0][-(i+1)])
            ypoints = m_rot[2] * xpoints + b

        #0,180,360,etc.
        if (angle/90)%2 == 0:
            xpoints = np.linspace(x_rot[3][i],x_rot[0][-(i+1)],n_spaces)
            b =y_rot[0][-(i+1)] - (m_rot[2]*x_rot[0][-(i+1)])
            ypoints = func(xpoints,m_rot[2],b)

        #convert xpoint, ypoint to arrays, round all elements to 2 decimal places, convert back to lists
        all_line_coords.append(list(zip(np.ndarray.tolist(np.round(np.asarray(xpoints),3)),
                                        np.ndarray.tolist(np.round(np.asarray(ypoints),3)))))

        for n in range(len(ypoints)):
            #from the full data grid x, isolate all of values occupying the rows (xpoints) in 
            #the column ypoints[n]
            list_to_mean.append(image[int(ypoints[n])][int(xpoints[n])])

            if calc_alt:
                list_to_mean_alt.append(image_alt[int(ypoints[n])][int(xpoints[n])])

        mean_list.append(np.mean(list_to_mean))
        list_to_mean = []

        if calc_alt:
            mean_list_alt.append(np.mean(list_to_mean_alt))
            list_to_mean_alt = []

    #check if all_line_coords arranged from left to right
    #if not, sort it (flip last list to first, etc.) and reverse mean_list accordingly
    #first define coordinates of first and second "starting coordinates"
    first_coor = all_line_coords[0][0]
    second_coor = all_line_coords[1][0]

    #isolate the x values
    first_x = first_coor[0]
    second_x = second_coor[0]

    #if the first x coordinate is greater than the second, then all set. 
    #otherwise, lists arranged from right to left. fix.
    #must also flip mean_list so that the values remain matched with the correct lines
    if first_x<second_x:
        all_line_coords.sort()
        mean_list.reverse()

        if calc_alt:
            mean_list_alt.reverse()
            
    return {'mean_list':mean_list,
            'all_line_coords':all_line_coords,
            'mean_list_alt':mean_list_alt}


#it may not be the most efficient function, as it calculates the distances between every line coordinate and the given (x,y); however, I am not clever enough to conjure up an alternative solution presently.
def find_closest_bar(all_line_coords, x, y):

    #initiate distances list --> for a given (x,y), which point in every line in self.all_line_coords
    #is closest to (x,y)? this distance will be placed in the distances list.
    distances=[]

    coord=(x,y)

    for line in all_line_coords:
        tree = spatial.KDTree(line)
        result=tree.query([coord])
        distances.append(result[0])

    closest_line_index = np.where(np.asarray(distances)==np.min(distances))[0][0]
    
    return closest_line_index


def sample_vertical_rectangle(image, xmin, xmax, ymin, ymax, image_alt=None):
    '''
    AIM: sample a non-rotated rectangle and return mean strip values and the coordinates associated with each strip.
    * xmin, xmax, ymin, ymax : int
        Rectangle bounds.
    * image_alt : np.ndarray or None
        Optional comparison image.
    '''

    cropped_data = image[ymin:ymax, xmin:xmax]

    mean_list = []

    x_coords = np.arange(xmin, xmax, 1)
    y_coords = np.arange(ymin, ymax, 1)

    all_line_coords = []

    for i in range(xmax - xmin):
        x = np.zeros(len(y_coords)) + x_coords[i]
        y = y_coords
        all_line_coords.append(list(zip(np.round(x, 3).tolist(), np.round(y, 3).tolist())))

    vertical_lines = [cropped_data[:, i] for i in range(xmax - xmin)]

    for line in vertical_lines:
        line = np.asarray(line)
        mean_list.append(np.mean(line[line != 0.]))

    mean_list_alt = None

    if image_alt is not None:
        cropped_data_alt = image_alt[ymin:ymax, xmin:xmax]
        mean_list_alt = []
        vertical_lines_alt = [cropped_data_alt[:, i] for i in range(xmax - xmin)]

        for line in vertical_lines_alt:
            line = np.asarray(line)
            mean_list_alt.append(np.mean(line[line != 0.]))

    return {'mean_list': mean_list, 'all_line_coords': all_line_coords, 'mean_list_alt': mean_list_alt}