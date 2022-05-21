import os
import tkinter as tk
import numpy as np
from matplotlib import pyplot as plt
import TDA
import scipy
import gemini_analyze
from gudhi.representations import WassersteinDistance
import multiprocessing
# 1. Generate Time Series Data
#       return(time_series(array), graph)
# 2. Generate MAPC and basic MAPC stats
#       return(mapc_stats(dict), graph)
# 3. Generate SW(Each point in SW is a MAPC) & Calculate
#   a. Mean Landscape, and mean Lp norm, for some p.
#   b. * After enough landscapes are generated, calculate probability new landscape was sampled from distribution.
#       return(probability(float(0,1)), dimension 1 norm(float(0,infty))).
#
root_directory = os.getcwd()
id = 'Sandbox'
directory = os.path.join(root_directory, id)
if not os.path.exists(directory):
    os.makedirs(directory)
id_num = str(np.random.randint(1000,9999))
sandbox_directory = os.path.join(directory,id_num)
while os.path.exists(sandbox_directory):
    print('directory already exists')
    id_num = str(np.random.randint(1000, 9999))
    sandbox_directory = os.path.join(directory, id_num)
directory = sandbox_directory
os.makedirs(sandbox_directory)
time_series_dict = {'data':[]}
render = None
parameters_f = {'time_series_method': 1, 'time_series_domain': '[0,1]', 'number_of_points': 500,
                'function': 'x', 'ts_plot': True, 'iterations': 100, 'ma_range': 10, 'number_of_samples_from_MSPC': 0,
                "MSPC_max_persistence": 1, "MSPC_min_persistence": .0001, "MSPC_sparse": .0001, 'mapc_plot':True, "SW_step": 10,
                "SW_Length": 20,
                "SW_min_persistence": .1, "SW_max_persistence": 1, "SW_sparse": .001}
SW = []
landscapes = []
landscape_norms = []
last_diagram_dict = {'diag':False}
wasserstein_distances = []
mean_wasserstein_distance = [0]
var_wasserstein_distance = [0]
test_landscapes = []
test_landscape_norms = []
sample = {'sample':False}
sample_landscape_norms = []
mean_landscape = [0]
test_dL1_probs = []
test_wasserstein_distances = []
test_wasserstein_prob = []

def write_ts_graph():
    # get image and display
    image = tk.PhotoImage(file = os.path.join(directory,"time_series.png"))
    tsimageLabel.configure(image = image)
    tsimageLabel.image = image
def reset_ts_graph():
    plt.clf()
    plt.savefig(os.path.join(directory,'time_series'))
    write_ts_graph()
def write_mapc_graph():
    # get image and display
    image = tk.PhotoImage(file = os.path.join(directory,"MAPC.png"))
    mapcimageLabel.configure(image = image)
    mapcimageLabel.image = image
def reset_mapc_graph():
    plt.clf()
    plt.savefig(os.path.join(directory,'MAPC'))
    write_mapc_graph()
def Time_Series(method, domain, number_of_points, function, plot_bool):
    Y = []
    if method == "F(x)":#Continous Functions
        x = np.linspace(eval(domain)[0], eval(domain)[1], num=number_of_points)
        X = 1*x
        Y = eval(function)
    elif method == "Random walk: [mean, stdv]":#Random Walk
        Y_0 = 1
        Y = [Y_0]
        X = np.linspace(eval(domain)[0], eval(domain)[1], num=number_of_points)
        for x in X:
            std_random = np.random.normal(eval(function)[0],eval(function)[1])
            Y.append(Y[-1]+std_random)
    elif method == "Gemini Data: [timestep,symbol]":
        X = np.linspace(eval(domain)[0], eval(domain)[1], num=number_of_points)

        Y = gemini_analyze.Price_Data(str(eval(ts_function_entry.get())[0]),str(eval(ts_function_entry.get())[1])+"usd")[0][0:number_of_points]
        Y =Y[::-1]
    if plot_bool == True:
        plot_save = True
        if plot_save == True:

            plt.plot(Y)
        # # plt.show()
            plt.savefig(os.path.join(directory,'time_series'))
    time_series_dict['data'] = Y
    #print(time_series_dict)
    write_ts_graph()
    #return(Y)
def Calculate_Prob():
    norm = float(L1_norm.get())
    mean_norm = float(mean_L1_norm.get())
    var_norm = float(var_L1_norm.get())
    # print(norm)
    # print(mean_norm)
    # print(var_norm)
    if norm<mean_norm:
        p = scipy.stats.norm(loc = mean_norm, scale=var_norm**(1/2)).cdf(norm)
        # print('here')
        # print(p)
        #(norm-mean_norm)/var_norm
    elif norm > mean_norm:
        p= 1-scipy.stats.norm(mean_norm, var_norm ** (1 / 2)).cdf(norm)
    else:
        p = .5
    # print(p)
    L1_prob.set(p)
def Gen_MAPC():
    parameters_f['SW_Length'] = int(np.floor(float(numpoints_entry.get())/(float(ma_range_entry.get())))-2)

    parameters_f['mapc_plot'] = True
    out_angles = []
    #for k in range(30):
    #
    #print('k=',k)
    for i in range(parameters_f['SW_Length']+1):
        # print('i=',i)
        w = int(ma_range_entry.get())
        ma = TDA.Create_Moving_Average(time_series_dict['data'][-1*(i+2)*w:-1*(i+1)*w][::-1],w,0)
        dma = TDA.Create_Slopes(time_series_dict['data'][-1*(i+2)*w:-1*(i+1)*w][::-1],1,w,0)
        MSPC_x, MSPC_y, MSPC_linked = TDA.Create_Averaging_Arrays(ma,dma)
        if parameters_f['mapc_plot'] == True:
                #parameters_f['mapc_plot'] = False
                plt.clf()
                plt.scatter(x=MSPC_x,y=MSPC_y)
                plt.savefig(os.path.join(directory,'MAPC'))
                write_mapc_graph()
        MSPC_angles = TDA.Transform_MSPC_To_S1([MSPC_x, MSPC_y], 1)

        for angle in MSPC_angles:
            out_angles.append(angle[0])
        #print(out_angles)
        cm = TDA.Center_of_Mass(MSPC_x,MSPC_y)
        if cm[0]>0:
            #print('here')
            #print(cm[0])
            mean_angle.set(np.arctan(cm[1]/cm[0]))
        elif cm[0]<0:
            mean_angle.set(np.pi+np.arctan(cm[1] / cm[0]))
        elif cm[1]==0:
            mean_angle.set(0)
        else:
            mean_angle.set(np.sign(cm[1])*np.pi/2)
        #print(mean_angle.get())
        #mean_angle.set(str(np.mean(out_angles)))
        SW.append(MSPC_angles)
        if len(SW) > parameters_f['SW_Length']:
            parameters_f['SW_min_persistence'] = float(minpers_entry.get())
            sw_diag, sw_top_summary, sw_landscape_dimension = TDA.SW_Homology(SW, directory, parameters_f,plot_bool=False)
            if last_diagram_dict['diag'] != False:
                W_1 = WassersteinDistance(order=1, internal_p=1, mode="hera", delta=1)
                Wdistance= W_1(sw_diag.persistence_intervals_in_dimension(1), last_diagram_dict['diag'].persistence_intervals_in_dimension(1))
                wasserstein_distances.append(Wdistance)
            #Compute Wasserstein Distance between sw_diag and last_diag
            last_diagram_dict['diag'] = sw_diag

            if np.array(sw_top_summary).size == 0:
                landscape_norms.append(0)
                L1_norm.set(0)

            else:
                landscapes.append(sw_top_summary[0])
                local_land_norm = np.linalg.norm(sw_top_summary,ord=1)
                landscape_norms.append(local_land_norm)
                # print(sw_top_summary)
                L1_norm.set(local_land_norm)
            SW.clear()

    #wasserstein_distance =
    #mean_L1_norm.set(np.linalg.norm([mean_landscape],ord=1))#mean_L1_norm.set(np.linalg.norm([mean_landscape],ord=1))
    if np.array(sw_top_summary).size==0:# == []:
        mean_L1_norm.set(0)
        var_L1_norm.set(0)
        mean_landscape.append([])
    else:
        mean_landscape.clear()
        mean_landscape.append(TDA.Mean_Landscape(landscapes))
        mean_L1_norm.set(np.mean(landscape_norms))
        var_L1_norm.set(np.var(landscape_norms))
    # print(np.linalg.norm([mean_landscape],ord=1))
    Calculate_Prob()
def Gen_Sample():
    parameters_f['SW_Length'] = int(np.floor(float(numpoints_entry.get())/(float(ma_range_entry.get())))-2)
    parameters_f['mapc_plot'] = True
    out_angles = []
    for i in range(parameters_f['SW_Length']+1):
        w = int(ma_range_entry.get())
        ma = TDA.Create_Moving_Average(time_series_dict['data'][-1*(i+2)*w:-1*(i+1)*w][::-1],w,0)
        dma = TDA.Create_Slopes(time_series_dict['data'][-1*(i+2)*w:-1*(i+1)*w][::-1],1,w,0)
        MSPC_x, MSPC_y, MSPC_linked = TDA.Create_Averaging_Arrays(ma,dma)
        if parameters_f['mapc_plot'] == True:
                parameters_f['mapc_plot'] = False
                plt.clf()
                plt.scatter(MSPC_x,MSPC_y)
                plt.savefig(os.path.join(directory,'MAPC'))
                write_mapc_graph()
        MSPC_angles = TDA.Transform_MSPC_To_S1([MSPC_x, MSPC_y], 1)
        for angle in MSPC_angles:
            out_angles.append(angle[0])
        cm = TDA.Center_of_Mass(MSPC_x,MSPC_y)
        if cm[0]==0:
            mean_angle.set(np.sign(cm[1])*np.pi/2)
        elif cm[0]>0:
            mean_angle.set(np.arctan(cm[1]/cm[0]))
        else:
            mean_angle.set(np.pi+np.arctan(cm[1] / cm[0]))
        SW.append(MSPC_angles)
        if len(SW) > parameters_f['SW_Length']:
            parameters_f['SW_min_persistence'] = float(minpers_entry.get())
            sw_diag, sw_top_summary, sw_landscape_dimension = TDA.SW_Homology(SW, directory, parameters_f,plot_bool=False)
            W_1 = WassersteinDistance(order=1, internal_p=1, mode="hera", delta=1)

            Wdistance = W_1(sw_diag.persistence_intervals_in_dimension(1), last_diagram_dict['diag'].persistence_intervals_in_dimension(1))
            test_wasserstein_distances.append(Wdistance)

            if np.array(sw_top_summary).size==0:# == []:
                test_landscape_norms.append(0)
                #L1_norm.set(0)
            else:
                test_landscapes.append(sw_top_summary[0])
                local_land_norm = np.linalg.norm(sw_top_summary,ord=1)

                test_landscape_norms.append(local_land_norm)
            SW.clear()
    dnorms = []
    if np.mean(test_landscape_norms)==0:
        if float(mean_L1_norm.get())==0:
            test_dL1_probs.append(1)
        else:
            test_dL1_probs.append(scipy.stats.norm(loc=float(mean_L1_norm.get()), scale=float(var_L1_norm.get()) ** (1 / 2)).cdf(0))
    elif float(mean_L1_norm.get())==0:
        test_dL1_probs.append(0)
    else:
        for landscape in landscapes:
            dnorms.append(np.linalg.norm(np.array(mean_landscape)-np.array(landscape), ord = 1))
        dnorm_mean = np.mean(dnorms)
        dnorm_var = np.var(dnorms)
        local_dnorm = np.linalg.norm(np.array(mean_landscape)-np.array(sw_top_summary[0]), ord=1)
        p = scipy.stats.norm(loc=dnorm_mean, scale=dnorm_var ** (1 / 2)).cdf(local_dnorm)
        if local_dnorm<=dnorm_mean:
            prob = 2*p
        elif local_dnorm > dnorm_mean:
            prob = 2*(1-p)
        test_dL1_probs.append(prob)

    # CAlCULATE PROBABILITY THE DISTANCE BETWEEN THE OBSERVED LANDSCAPE AND MEAN LANDSCAPE IS GREATER THAN THE MEAN NORM OF LANDSCAPES TO THE MEAN LANDSCAPE
    # mean_L1_norm.set(np.mean(sample_landscape_norms))
    # var_L1_norm.set(np.var(sample_landscape_norms))
#### ##### #### #####
def run_batch():
    test_dL1_probs.clear()
    test_landscapes.clear()
    test_landscape_norms.clear()
    test_wasserstein_distances.clear()
    for k in range(int(batch_size_entry.get())):
        plt.clf()
        Time_Series(str(ts_method.get()), ts_domain_entry.get(),
                    int(numpoints_entry.get()), str(ts_function_entry.get()),
                    parameters_f['ts_plot'])
        if sample['sample'] == False:
            Gen_MAPC()
            mean_wasserstein_distance[0] = np.mean(wasserstein_distances)
            var_wasserstein_distance[0] = np.var(wasserstein_distances)
        else:
            Gen_Sample()
    TESTL1_norm.set(np.mean(test_landscape_norms))
    mean_test_wasserstein_distance = np.mean(test_wasserstein_distances)
    var_test_wasserstein_distance = np.var(test_wasserstein_distances)

    if mean_wasserstein_distance[0] == 0 and mean_test_wasserstein_distance==0:
        wasserprob.set(1)
    elif mean_wasserstein_distance[0]==0 and mean_test_wasserstein_distance!=0:
        wasserprob.set(0)
    else:
        p = scipy.stats.norm(loc=mean_wasserstein_distance[0],
                         scale=var_wasserstein_distance[0] ** (1 / 2)).cdf(mean_test_wasserstein_distance)
        if mean_test_wasserstein_distance<=mean_wasserstein_distance[0]:
            wasserstein_prob =  2*p
        else:
            wasserstein_prob = 2*(1-p)
        wasserprob.set(wasserstein_prob)
    #print(test_dL1_probs)
    dL1_norm_prob.set(np.mean(test_dL1_probs))
def Test():
    sample['sample'] = True
    run_batch()
    sample['sample'] = False
def close():
   sand_window.quit()
def reset():
    SW.clear()
    landscapes.clear()
    landscape_norms.clear()
    wasserstein_distances.clear()
    mean_wasserstein_distance[0] = 0
    var_wasserstein_distance[0] = 0
    test_landscapes.clear()
    test_landscape_norms.clear()
    sample['sample'] = False
    last_diagram_dict['diag'] = False
    mean_landscape.clear()
    test_dL1_probs.clear()
    test_wasserstein_distances.clear()
    test_wasserstein_prob.clear()


# Create Sandbox Window
sand_window = tk.Tk()
sand_window.attributes('-fullscreen', True)
#sand_window.geometry("1200x1200")
# Make Window into a scrolling frame
container = tk.Frame(sand_window)
#canvas = tk.Canvas(container,height=1100,width=1600)
canvas = tk.Canvas(container)
# def _on_mousewheel(event):
#     canvas.yview_scroll(-1*(event.delta/120), "units")
# canvas.bind_all("<MouseWheel>", _on_mousewheel)

scrollbarh = tk.Scrollbar(container, orient="horizontal", command=canvas.xview)
scrollbarv = tk.Scrollbar(container, orient="vertical", command=canvas.yview)

canvas.configure(yscrollcommand=scrollbarv.set)
canvas.configure(xscrollcommand=scrollbarh.set)
scrollbarh.pack(side="bottom", fill="y")
scrollbarv.pack(side="right", fill="y")
container.pack(fill=tk.BOTH, expand=True)
canvas.pack(side="left", fill="both", expand=True)
scrollable_frame = tk.Frame(canvas)
scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(
        scrollregion=canvas.bbox("all")
    )
)
canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")


# Make a label and variable for the time-series method
#### #### #### ####
label_method = tk.StringVar()
label_method.set("Choose Method")
label_method_Dir = tk.Label(scrollable_frame, textvariable=label_method)
label_method_Dir.grid(row=1,column = 0)
ts_method = tk.StringVar()
ts_method.set("Random walk: [mean, stdv]")
ts_method_entry = tk.OptionMenu(scrollable_frame,ts_method,*["F(x)",
                                                            "Random walk: [mean, stdv]",
                                                            "Gemini Data: [timestep,symbol]"])#tk.Entry(scrollable_frame, textvariable='0')
#ts_method_entry.insert(tk.END,'1')
ts_method_entry.grid(row=1,column = 1)

# Make a label and variable for the time-series function
#### #### #### ####
label_function = tk.StringVar()
label_function.set("Choose function")
label_function_Dir = tk.Label(scrollable_frame, textvariable=label_function)
label_function_Dir.grid(row=1,column = 2)
ts_function_entry = tk.Entry(scrollable_frame, textvariable='1')
ts_function_entry.insert(tk.END,'[0,1]')
ts_function_entry.grid(row=1,column = 3)
# Make a label and variable for the time-series domain
#### #### #### ####
label_domain = tk.StringVar()
label_domain.set("Choose a domain")
label_domain_Dir = tk.Label(scrollable_frame, textvariable=label_domain)
label_domain_Dir.grid(row=1,column = 4)
ts_domain_entry = tk.Entry(scrollable_frame, textvariable='2')
ts_domain_entry.insert(tk.END,'[0,1]')
ts_domain_entry.grid(row=1,column = 5)

# Create place for the image of the graph
#### #### #### ####
tsimageLabel = tk.Label(scrollable_frame)
tsimageLabel.grid(row=3,column = 0)
# Label and variable for the moving average range

#### #### #### ####
label_ma_range = tk.IntVar()
label_ma_range.set("Moving Average Range")
label_ma_range_Dir = tk.Label(scrollable_frame, textvariable=label_ma_range)
label_ma_range_Dir.grid(row=4,column = 0)
ma_range_entry = tk.Entry(scrollable_frame, textvariable='3')
ma_range_entry.insert(tk.END,'10')
ma_range_entry.grid(row=4,column = 1)
#batch_size_entry
label_batch_size_entry = tk.IntVar()
label_batch_size_entry.set("batch size")
label_batch_size_entry_Dir = tk.Label(scrollable_frame, textvariable=label_batch_size_entry)
label_batch_size_entry_Dir.grid(row=4,column = 2)
batch_size_entry = tk.Entry(scrollable_frame, textvariable='4')
batch_size_entry.insert(tk.END,'10')
batch_size_entry.grid(row=4,column = 3)
#
label_numpoints = tk.IntVar()
label_numpoints.set("Number of Points")
label_numpoints_Dir = tk.Label(scrollable_frame, textvariable=label_numpoints)
label_numpoints_Dir.grid(row=4,column = 4)
numpoints_entry = tk.Entry(scrollable_frame, textvariable='5')
numpoints_entry.insert(tk.END,'100')
numpoints_entry.grid(row=4,column = 5)
#
label_minpers = tk.IntVar()
label_minpers.set("Minimum Persistence")
label_minpers_Dir = tk.Label(scrollable_frame, textvariable=label_minpers)
label_minpers_Dir.grid(row=4,column = 6)
minpers_entry = tk.Entry(scrollable_frame, textvariable='6')
minpers_entry.insert(tk.END,'.1')
minpers_entry.grid(row=4,column = 7)

# Placeholder to display the MAPC graph
#### #### #### ####
mapcimageLabel = tk.Label(scrollable_frame)
mapcimageLabel.grid(row=5,column = 0)
# Button to generate the MAPC
#### #### #### ####
button_gen_mapc = tk.Button(scrollable_frame, text="Generate MAPC",
                            command=Gen_MAPC)
button_gen_mapc.grid(row=0,column = 4)
#
button_quit = tk.Button(scrollable_frame, text="QUIT",
                        command=close)
button_quit.grid(row=0,column = 0)
#### #### #### ####
button_reset = tk.Button(scrollable_frame, text="OOPS reset DATA(rerun to generate new frames)",
                        command=reset)
button_reset.grid(row=10,column = 0)
# Button to display MAPC graph
#### #### #### ####
mapc_graph = tk.Button(scrollable_frame,
                  command=write_mapc_graph)
mapc_graph.grid(row=5,column = 1)
# Reset Graph
#### #### #### ####
reset_mapc_graph = tk.Button(scrollable_frame,text="Clear MAPC Graph",
                  command=reset_mapc_graph)
reset_mapc_graph.grid(row=0,column = 5)
#Make a label and variable for a button to generate the time series
#### #### #### ####
button_run = tk.Button(scrollable_frame, text="Run Batch",
                       command=run_batch)
button_run.grid(row=0,column = 1)
#### #### #### #####
button_gen_ts = tk.Button(scrollable_frame, text="Generate Time Series",
                       command=lambda: Time_Series(str(ts_method.get()),ts_domain_entry.get(),
                                                   int(numpoints_entry.get()),str(ts_function_entry.get()),
                                                   parameters_f['ts_plot']))
button_gen_ts.grid(row=0,column = 2)
# Make a button to generate the time-series graph and display it
#### #### #### ####
reset_ts_graph = tk.Button(scrollable_frame,text="Clear Graph",
                  command=reset_ts_graph)
reset_ts_graph.grid(row=0,column = 3)

Test = tk.Button(scrollable_frame,text="TEST",
                  command=Test)
Test.grid(row=10,column = 1)
# REPORTS
# Label and variable to display the mean angle of the MAPC
#### #### #### ####
label_mean_angle = tk.StringVar()
label_mean_angle.set("Mean Angle")
label_mean_angle_Dir = tk.Label(scrollable_frame, textvariable=label_mean_angle)
label_mean_angle_Dir.grid(row=7,column = 0)
mean_angle = tk.StringVar()
mean_angle_value = tk.Label(scrollable_frame, textvariable=mean_angle)
mean_angle_value.grid(row=7,column = 1)
#### #### #### ####
label_L1_norm = tk.StringVar()
label_L1_norm.set("Dimension 1 Landscape Distance")
label_L1_norm_Dir = tk.Label(scrollable_frame, textvariable=label_L1_norm)
label_L1_norm_Dir.grid(row=8,column = 0)
L1_norm = tk.StringVar()
L1_norm_value = tk.Label(scrollable_frame, textvariable=L1_norm)
L1_norm_value.grid(row=8,column = 1)
#### #### #### ####
label_TESTL1_norm = tk.StringVar()
label_TESTL1_norm.set("TEST MEAN Dimension 1 Landscape Distance")
label_TESTL1_norm_Dir = tk.Label(scrollable_frame, textvariable=label_TESTL1_norm)
label_TESTL1_norm_Dir.grid(row=8,column = 2)
TESTL1_norm = tk.StringVar()
TESTL1_norm_value = tk.Label(scrollable_frame, textvariable=TESTL1_norm)
TESTL1_norm_value.grid(row=8,column = 3)
#### #### #### #####
label_mean_L1_norm = tk.StringVar()
label_mean_L1_norm.set("MEAN Dimension 1 Landscape Distance")
label_mean_L1_norm_Dir = tk.Label(scrollable_frame, textvariable=label_mean_L1_norm)
label_mean_L1_norm_Dir.grid(row=9,column = 0)
mean_L1_norm = tk.StringVar()
mean_L1_norm_value = tk.Label(scrollable_frame, textvariable=mean_L1_norm)
mean_L1_norm_value.grid(row=9,column = 1)
#### #### #### #####
label_var_L1_norm = tk.StringVar()
label_var_L1_norm.set("VARIANCE Dimension 1 Landscape Distance")
label_var_L1_norm_Dir = tk.Label(scrollable_frame, textvariable=label_var_L1_norm)
label_var_L1_norm_Dir.grid(row=9,column = 2)
var_L1_norm = tk.StringVar()
var_L1_norm_value = tk.Label(scrollable_frame, textvariable=var_L1_norm)
var_L1_norm_value.grid(row=9,column = 3)
#### #### #### #####
#### #### #### ####
label_L1_prob = tk.StringVar()
label_L1_prob.set("Probability of observing a landscape norm greater than, or less than, the given norm(whichever is smaller)")
label_L1_prob_Dir = tk.Label(scrollable_frame, textvariable=label_L1_prob)
label_L1_prob_Dir.grid(row=9,column = 4)
L1_prob = tk.StringVar()
L1_prob_value = tk.Label(scrollable_frame, textvariable=L1_prob)
L1_prob_value.grid(row=9,column = 5)
#### #### #### ####
label_dL1_norm_prob = tk.StringVar()
label_dL1_norm_prob.set("Prob of TEST landscape")
label_dL1_norm_prob_Dir = tk.Label(scrollable_frame, textvariable=label_dL1_norm_prob)
label_dL1_norm_prob_Dir.grid(row=10,column = 4)
dL1_norm_prob = tk.StringVar()
dL1_norm_prob_value = tk.Label(scrollable_frame, textvariable=dL1_norm_prob)
dL1_norm_prob_value.grid(row=10,column = 5)
#### #### #### ####
# label_L1_Tprob = tk.StringVar()
# label_L1_Tprob.set("T TEST")
# label_L1_Tprob_Dir = tk.Label(scrollable_frame, textvariable=label_L1_Tprob)
# label_L1_Tprob_Dir.grid(row=10,column = 4)
# L1_Tprob = tk.StringVar()
# L1_Tprob_value = tk.Label(scrollable_frame, textvariable=L1_Tprob)
# L1_Tprob_value.grid(row=10,column = 5)

label_wasserprob = tk.StringVar()
label_wasserprob.set("Prob of TEST Diagrams to LAST TRAIN Diagram")
label_wasserprob_Dir = tk.Label(scrollable_frame, textvariable=label_wasserprob)
label_wasserprob_Dir.grid(row=10,column = 2)
wasserprob = tk.StringVar()
wasserprob_value = tk.Label(scrollable_frame, textvariable=wasserprob)
wasserprob_value.grid(row=10,column = 3)
sand_window.mainloop()


