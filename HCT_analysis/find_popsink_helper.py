import numpy as np
from calculate_pos_and_dir import get_directions_to_position, get_relative_directions_to_position 
from calculate_occupancy import get_directional_occupancy

def get_relative_direction_occupancy_by_plat(pos_xy, hd,  candidate_sinks):
    '''
    APPENDED FOR POP SINK CODE
    output is a y, x, y, x, n_bins array.
    The first y and x are the position bins, and the second y and x are the consink positions.     
    '''
        

    x_sink_pos = candidate_sinks['x']
    y_sink_pos = candidate_sinks['y']


    # create relative directional occupancy by position array
    n_dir_bins=12 # 12 bins of 30 degrees each
    n_x_sinks = len(x_sink_pos)
    n_y_sinks = len(y_sink_pos)

    reldir_occ_by_plat = np.zeros((61, n_y_sinks, n_x_sinks, n_dir_bins))
    
    # get x and y data

    x = pos_xy['x']
    y = pos_xy['y']


    for p in range(61):
        x_positions = x[p]
        y_positions = y[p]
        positions = {'x': x_positions, 'y': y_positions}

        # get the head directions and durations for these indices
        hd_temp = hd[p]


        # loop through possible consink positions
        for i2, x_sink in enumerate(x_sink_pos):
            for j2, y_sink in enumerate(y_sink_pos):
            
                # get directions to sink                    
                directions = get_directions_to_position([x_sink, y_sink], positions)
                
                # get the relative direction
                relative_direction = get_relative_directions_to_position(directions, hd_temp)
                durations_temp = np.ones(len(relative_direction)) #NOTE: Jake's code used durations data from DLC.
                #We don't use that, each frame is equally long, so we replace all durations with 1 (meaning each frame has length 1)

                # get the directional occupancy for these indices
                directional_occupancy, direction_bins = \
                    get_directional_occupancy(relative_direction, durations_temp, n_bins=12)

                # add the directional occupancy to positional_occupancy_temp
                reldir_occ_by_plat[p, j2, i2, :] = directional_occupancy

    return reldir_occ_by_plat