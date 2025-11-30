import copy

def pre_pocessing_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    base_x, base_y = 0,0
    relative_points = []

    for i, c in enumerate(temp_landmark_list):
        if i == 0:
            base_x, base_y = c.x, c.y
            break
    for c in temp_landmark_list:
        new_x = c.x - base_x
        new_y = c.y - base_y

        relative_points.append(new_x)
        relative_points.append(new_y)
    
    max_value = max(list(map(abs, relative_points)))

    def normalize(n):
        return n/max_value
    
    if max_value > 0 :
        normalized_points = list(map(normalize, relative_points))
        return normalized_points
    else :
        return relative_points