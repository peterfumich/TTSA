# IMPORT STATEMENTS ARE THERE FOR CURRENT OR FUTURE DEPENDENCY.
import numpy as np
import pandas as pd
import requests, json
import time
import gudhi
from gudhi.representations import Landscape
from matplotlib import pyplot as plt
import csv
import os
import math
import sklearn
import scipy
###############################################
time_steps = ['1m', '5m', '15m', '30m', '1hr', '1day']
# THIS PROCEDURE CALCULATES THE SET OF MOVING AVERAGES  Input prices must be listed with most recent value first.
def Create_Moving_Average(prices, window_depth,time_step):#time_step in seconds
    set_of_moving_averages = [prices[0]]
    try:
        for x in range(1, window_depth):  # range(1,int(60/time_step_int)): Moving
            set_of_moving_averages.append((set_of_moving_averages[-1] * x + prices[x]) / (x + 1))
        return (set_of_moving_averages)
    except:
        return (set_of_moving_averages)
# THIS PROCEDURE CALCULATES THE SET OF SLOPES OF THE MOVING AVERAGES. Input prices must be listed with most recent value first.
def Create_Slopes(prices, delta, window_depth,time_step):
    set_of_slopes = [(prices[0] - prices[1])/(delta)]
    try:
        for y in range(1, window_depth):  # range(1,int(60/time_step_int)):
            set_of_slopes.append((prices[0] - prices[y + 1]) / (delta * (y + 1)))

        return (set_of_slopes)
    except:
        return (set_of_slopes)
# Creates the space which will be the INPUT for the Persistence Homology Procedure. OUTPUT CONSISTS OF POINTS OF THE FORM (DISTANCE BETWEEN TWO ELEMENTS OF THE MOVING AVERAGE SET, DISTANCE BETWEEN TWO ELEMENTS OF THE SLOPES SET)
def Create_Averaging_Arrays(MA_set, Slope_set):
    length = min(len(MA_set),len(Slope_set))
    MA_set = MA_set[0:length]
    Slope_set = Slope_set[0:length]
    precision = 10
    index = 0
    average_diff = []
    slope_diff = []
    for large_average in MA_set:
        for small_average_index in range(index):
            small_average = MA_set[small_average_index]
            #print(small_average-large_average)
            average_diff.append(round(small_average - large_average, precision))
        index += 1
    index = 0
    for large_slope in Slope_set:
        for small_slope_index in range(index):

            small_slope = Slope_set[small_slope_index]
            slope_diff.append(round(large_slope - small_slope, precision))
        index += 1
    output = []
    for x, avg in enumerate(average_diff):
        output.append(avg)
        output.append(slope_diff[x])
    return([average_diff,slope_diff,output])#return ([average_diff, slope_diff])
def Transform_MSPC_To_S1(points, method):
    # complex = []
    # for point in points:
    #     complex.append(point[0]+1j*point[1])
    # return(complex)
    compute_distance = True
    angles = []
    if method == 0:
        for point in points:
            if point[0] == 0:
                theta = np.pi / 2 * np.sign(point[1])
            else:
                theta = np.arctan(point[1] / point[0])
            angles.append(theta)
            np.round(angles, 3)
    elif method == 1:
        X = points[0]
        Y = points[1]
        for n,x in enumerate(X):
            if x == 0:
                theta = np.pi/2*np.sign(Y[n])
            else:
                theta = np.arctan(Y[n]/x)

            if compute_distance == True:
                if x<0:
                    theta+=np.pi
                elif theta<0:
                    theta+=2*np.pi


            angles.append([theta,0])
    return(angles)
###############################################
###############################################
#Center Stuff
###############################################
###############################################
def Convert_vector_of_angles_to_distance_matrix(angles):
    distance_matrix = np.zeros((len(angles),len(angles))) #scipy.spatial.distance_matrix(angles)

    # print(len(angles))
    for i in range(len(angles)):
        for j in range(i+1,len(angles)):
            # print(angles[i])
            # print(angles[j])
            angular_distance_coordinate_vector = []
            for k in range(len(angles[i])):
                # print(angles[i][k])
                angular_distance_coordinate_vector.append(sklearn.metrics.pairwise.haversine_distances([angles[i][k]],[angles[j][k]])[0])
            # print('here')
            # print(angular_distance_coordinate_vector)
            distance_matrix[i,j] = np.linalg.norm(angular_distance_coordinate_vector,ord=1)
            # print(distance_matrix)
    #
    # print('there')
    # print(distance_matrix)

    distance_matrix = distance_matrix[np.triu_indices(len(distance_matrix),k=1)]
    distance_matrix = 1/(np.pi*len(angles[0]))*distance_matrix

    #distance_matrix = np.round(distance_matrix,3)

    # print(distance_matrix)
    output = [[]]
    temp = []
    count = 1
    length = 1
    # print('start')
    for n in range(len(distance_matrix)):
        if n<count:
            temp.append(distance_matrix[n])
        else:
            output.append(temp)
            temp = [distance_matrix[n]]
            length+=1
            count=n+length
    output.append(temp)
    #print(output)
    distance_matrix = output
    return(distance_matrix)

# PROCEDURE CALCULATES THE CENTER OF MASS OF THE MOVING AVERAGE SPACE. INPUT IS Create_Averaging_Arrays OUTPUT IS [HORIZONTAL COMPONENT OF CENTER OF MASS, VERTICAL COMPONENT OF CENTER OF MASS]
def Center_of_Mass(avg_diff, slope_diff):
    X = 0
    for x in avg_diff:
        X += x
    avg_center = X / len(avg_diff)
    Y = 0
    for y in slope_diff:
        Y += y
    slope_center = Y / len(slope_diff)
    print('center of mass')
    print(avg_diff)

    print(np.mean(np.array(avg_diff)))
    print(np.mean(np.array(slope_diff)))
    return ([avg_center, slope_center])
# OUTPUT : CENTER
#
def Confidence_Weighted_Center_of_Mass(avg_diff, slope_diff):
    X = 0
    Y = 0
    for i in range(len(avg_diff)):
        try:
            weight = abs(slope_diff[i] / (avg_diff[i] ** 2 + slope_diff[i] ** 2) ** (1 / 2))
        except:
            weight = 0
        X += avg_diff[i] * weight
        Y += slope_diff[i] * weight
    avg_center = X / len(avg_diff)
    slope_center = Y / len(slope_diff)
    return ([avg_center, slope_center])
    # cwcm = [0,0]
    # for i in range(len(avg_diff)):
    #     angle = np.arctan(slope_diff[i]/avg_diff[i])
    #     confidence_value = np.cos(angle)**2
    #     cwcm = [cwcm[0]+avg_diff[i]*confidence_value,cwcm[1]+slope_diff[i]*confidence_value]
    # cwcm = [x/len(avg_diff) for x in cwcm]
    # return(cwcm)
# OUTPUT : CENTER
#
def Boundary_Box_Center(averages, slopes):
    avg_min = np.amin(averages)
    avg_max = np.amax(averages)
    slope_min = np.amin(slopes)
    slope_max = np.amax(slopes)
    return ([(avg_max + avg_min) / 2, (slope_min + slope_max) / 2])
# OUTPUT : CENTER
#
def Center_Evolution(centers,new_center):
    # Calculate angle given complex representation of coordinates.
    angle_of_center= np.angle([new_center[0]+new_center[1]*1j])
    if 0<=angle_of_center and angle_of_center<1/2*np.pi:
        center_state = [1,1]
    elif 1/2*np.pi<=angle_of_center and angle_of_center<np.pi:
        center_state = [-1,1]
    elif -np.pi<=angle_of_center and angle_of_center<-1/2*np.pi:
        angle_of_center += 2*np.pi
        center_state = [-1,-1]
    elif -1/2*np.pi<=angle_of_center<0:
        angle_of_center += 2 * np.pi
        center_state = [1,-1]
    else:
        center_state = [0,0]

    d_center = np.subtract(new_center,centers[-1])
    d_center_prev = np.subtract(centers[-1],centers[-2])
    dd_center = np.subtract(d_center,d_center_prev)
    cm_evolution_vector = np.add(d_center,1/2*dd_center)
    #expected_positions = np.add(new_center_set,cm_evolution_vectors)
    length_of_evolution_vector = np.linalg.norm(cm_evolution_vector)
    angle_of_evolution_vector = np.angle(cm_evolution_vector[0]+cm_evolution_vector[1]*1j)
    if angle_of_evolution_vector<0:
        angle_of_evolution_vector+=2*np.pi
    type_vector_angle = np.subtract(angle_of_evolution_vector,angle_of_center)
    if type_vector_angle<0:
        type_vector_angle+=2*np.pi
    return([center_state,type_vector_angle,length_of_evolution_vector])
#OUTPUT : "return([center_state,type_vector_angle,length_of_evolution_vector])"
#
#TOPOLOGICAL PROCEDURES#
#
def Diagram_to_Array(diag):
    out_array = []
    for value in diag:
        out_array.append([value[1][0], value[1][1]])
    # print(out_array)
    return (out_array)
def Diagram_to_Dimension_Arrays(diag):
    out_diags = []
    for i in range(2):
        dim_diag = []
        for x in diag:
            if x[0] == i:
                dim_diag.append(x[1])
        out_diags.append(dim_diag)
    return(out_diags)
def Simplex_Tree(Points, max_edg_len, minimum_persistence, sparse, max_dimension, method):
    print("Building Simplex Tree...")
    # witnesses = Points
    # landmarks = gudhi.pick_n_random_points(points=witnesses, nb_points=10)
    # witness_complex = gudhi.EuclideanWitnessComplex(witnesses=witnesses, landmarks=landmarks)
    # simplex_tree = witness_complex.create_simplex_tree(max_alpha_square=.01, limit_dimension=3)

    # rips_complex = gudhi.RipsComplex(points=Points,
    #                                  max_edge_length=max_edg_len, sparse = sparse)#Sparse = .1
    if method == 'distances':
        rips_complex = gudhi.RipsComplex(distance_matrix=Points,
                                         max_edge_length=max_edg_len, sparse = sparse)#Sparse = .1
    elif method== 'points':
        rips_complex = gudhi.RipsComplex(points=Points,
                                         max_edge_length=max_edg_len, sparse=sparse)  # Sparse = .1
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dimension)
    #simplex_tree.collapse_edges()
    # result_str = 'Rips complex is of dimension ' + repr(simplex_tree.dimension()) + ' - ' + \
    #     repr(simplex_tree.num_simplices()) + ' simplices - ' + \
    #     repr(simplex_tree.num_vertices()) + ' vertices.'
    # print(result_str)
    #os.system("say 'Persistence'")
    print("Simplex Tree Built, computing persistence")
    simplex_tree.compute_persistence(min_persistence=minimum_persistence)


    # see what happens with just persistence()
    # print(simplex_tree.persistence_intervals_in_dimension(1))
    #LANDSCAPE
    LS = gudhi.representations.Landscape(resolution=2000)
    #SH = gudhi.representations.Silhouette(resolution=1000)#, weight=lambda x: np.power(x[1] - x[0], 1))
    #PI = gudhi.representations.PersistenceImage(bandwidth=1e-4, weight=lambda x: x[1] ** 2, im_range=[-.02, .02, -.01, .01], resolution=[1000, 1000])
    landscape_dimension = 0
    if simplex_tree.persistence_intervals_in_dimension(1) != []:
        print("Cycles Detected")
        array = simplex_tree.persistence_intervals_in_dimension(1)
        #print(array)
        out_array = []
        for x in array:
            if not math.isinf(x[1]):
                out_array.append(x)
        #print(out_array)
        if out_array != []:
            #L1 = PI.fit_transform([simplex_tree.persistence_intervals_in_dimension(1)])
            #L1 = SH.fit_transform([np.array(out_array)])

            L1 = LS.fit_transform([np.array(out_array)])
            landscape_dimension = 1
        else:
            L1 = []
    elif simplex_tree.persistence_intervals_in_dimension(0) != []:
        print("No cycles, but connected components")
        array = simplex_tree.persistence_intervals_in_dimension(0)
        out_array = []
        for x in array:
            if not math.isinf(x[1]):
                out_array.append(x)
        #print(out_array)
        if out_array != []:
            L1 = LS.fit_transform([np.array(out_array)])
            #L1 = SH.fit_transform([np.array(out_array)])
            #L1 = PI.fit_transform([np.array(out_array)])
        else:
            L1 = []
        landscape_dimension = 0
    else:
        print("Singular")
        out_array = []
        L1 = []
        landscape_dimension = 'none'
    #silhouette,

    if out_array != []:
        new_betti_numbers = simplex_tree.betti_numbers()
    else:
        print("Singular")
        new_betti_numbers = [1]

    #print(new_betti_numbers)

    #tree = simplex_tree.persistence()
    #print(tree)
    #print(simplex_tree)

    #diag = simplex_tree.persistence(min_persistence=minimum_persistence)

    return (simplex_tree, L1, new_betti_numbers, landscape_dimension)
#OUTPUT : Persistence Diagram, "return(diag)"
def Plot_Persistence_Diagram(diag, landscape, T, directory):
    print("PLOTTING")
    #plt.subplot(1,2,1)
    plot_barcode = True
    plot_diagram = True
    plot_landscape = True
    landscape_type = 'landscape'
    if plot_barcode == True:

        gudhi.plot_persistence_barcode(diag.persistence())
        plt.savefig(os.path.join(directory, "Persistence Barcode" + str(T)+ ".png"), dpi=300)

        print("Barcode Built")
        plt.clf()
        plt.close()
    if plot_diagram == True:
        gudhi.plot_persistence_diagram(diag.persistence(), legend = True)
        plt.savefig(os.path.join(directory, "Persistence Diagram" +str(T) +  ".png"), dpi=300)
        print("Persistence Diagram Built")
        plt.clf()
        plt.close()
    if plot_landscape ==True:
        L = landscape
    if landscape_type == 'img':#Persistence Image
        plt.imshow(np.flip(np.reshape(L[0], [1000, 1000]), 0))
        plt.title("Persistence Image")
        plt.savefig(os.path.join(directory, "Persistence Image" + str(T) + ".png"), dpi=300)

    elif landscape_type== 'landscape':#Persistence Landscape
        if L!= []:
            plt.plot(L[0][:1000])
            plt.plot(L[0][1000:2000])
            plt.plot(L[0][2000:3000])
            plt.savefig(os.path.join(directory,"Persistence Landscape" + str(T)+ ".png"), dpi=300)
    elif landscape_type == 'sil':#Persistence Silhouette
        if L!= []:
            plt.plot(L[0])
            plt.title("Silhouette")
            plt.savefig(os.path.join(directory, "Persistence Sil" + str(T) + ".png"), dpi=300)

    plt.clf()
    plt.close()
    #     plt.title("Landscape")
    # for i in range(2):
    #     L = landscape[i]
    #     plt.subplot(1,2,i+1)
    #
    #     plt.plot(L[0][:1000])
    #     plt.plot(L[0][1000:2000])
    #     plt.plot(L[0][2000:3000])
    #     plt.title("Landscape")

    # plt.subplot(1,3,1)
    # gudhi.plot_persistence_density(diag,
    #                                max_intervals=0, dimension=0, legend=True)
    # plt.subplot(1,3,2)
    # gudhi.plot_persistence_density(diag,
    #                                max_intervals=0, dimension=1)
    # plt.subplot(2,3,3)
    # gudhi.plot_persistence_density(diag,
    #                                 max_intervals=0, dimension=1, legend=True)
    # plt.savefig(os.path.join(directory, str(T) + "Persistence Heat Diagram" + ".png"), dpi=300)
    # print("Barcode Density")
    # # plt.show()
    # plt.close()
    #betti_curve = gudhi.representations.vector_methods.BettiCurve(resolution=100)(Diagram_to_Array(diag))
    #dim_1_diag, dim_2_diag = Diagram_to_Dimension_Arrays(diag)
    #print(dim_1_diag)
    #print(dim_2_diag)
    #input()

#OUTPUT : PLOT

#OUTPUT : ARRAY. Converts persistence diagram to an array.
#OUTPUT : UNUSED
def Calculate_Persistence(old_diagram, market_space, max_dist, min_pers):
    tree = Simplex_Tree(market_space[0], market_space[1], max_dist, min_pers)
    diagram = Diagram_to_Array(tree)
    if old_diagram != False:
        persistence_distance = gudhi.bottleneck_distance(old_diagram, diagram)
        #print(persistence_distance)
        # try:
        #     persistence_distance =  gudhi.bottleneck_distance(old_diagram, diagram)
        # except:
        #     return(0,diagram)
    else:
        persistence_distance = 0
    return(persistence_distance,diagram)
#OUTPUT : "return(persistence_distance, persistence_diagram)"
# Other
def Stats(array):
    sum = 0
    # print(array)
    max = 0
    for x in array:
        # print(x)
        if float(x) > max:
            max = float(x)
        sum += float(x)
    mean = sum / 50
    median = array[25]
    return (sum, mean, median, max)
def SW_Homology(sw_angles, index, parameters,plot_bool):
    #Convert Data to Distance Matrix:

    data = Convert_vector_of_angles_to_distance_matrix(sw_angles)

    diag, topological_summary, bett_numbers, landscape_dimension = Simplex_Tree(data,
                                                                                    parameters['SW_max_persistence'],
                                                                                    parameters['SW_min_persistence'],
                                                                                    parameters['SW_sparse'], 10,'distances')  # 2*max(abs(x_list)),min(abs(y_list)))
    if landscape_dimension==0:
        topological_summary=[0]

    if plot_bool == True:
        if landscape_dimension == 1:
            Plot_Persistence_Diagram(diag, topological_summary, "", index)

        else:
            Plot_Persistence_Diagram(diag, topological_summary, "", index)

    return (diag, topological_summary, landscape_dimension)
def Mean_Landscape(landscapes):
    #print(landscapes)
    #print(landscapes)
    # zero_landscape = False
    #
    # if landscapes[0] == 0:
    #     zero_landscape=True
    #
    # if zero_landscape==False:
    try:
        dist_matrix = scipy.spatial.distance_matrix(np.array(landscapes),np.array(landscapes))
        row_sums = []
        for row in dist_matrix:
            row_sums.append(np.sum(row))
        row_sums = np.array(row_sums)
        #print(row_sums)
        min_norm_index = np.argmin(row_sums)
        #print('here')
        #print(min_norm_index)
        mean_landscape = landscapes[min_norm_index]
        #print('mean_landscape')
        #print(mean_landscape)
    #else:
    except:
        mean_landscape = [0]
    return(mean_landscape)
def Var_Landscape(landscapes,mean_landscape):

    for land in landscapes:
        (land-mean_landscape)**2